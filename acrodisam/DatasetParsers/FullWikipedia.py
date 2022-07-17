'''
Wikipedia dump parser

Created on Apr 22, 2019

Before executing this code please follow the next steps:

use wget 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2'
to download the latest english wikipedia dump file

Download and setup WikiExtractor
http://medialab.di.unipi.it/wiki/Wikipedia_Extractor

Execute Wikipedia Extractor with the downloaded dump file as argument
E.g., python3.6 WikiExtractor.py  ~/Downloads/enwiki-latest-pages-articles.xml.bz2

Move the WikiExtractor output to:
{project root folder}/acrodisam_app/data/FullWikipedia/

Execute this script

@author: jpereira
'''

import os
import platform

import pickle
import functools
from Logger import logging

from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from sqlitedict import SqliteDict


import multiprocessing as mp
from multiprocessing.managers import BaseManager

import multiprocessing_logging

from AcroExpExtractors.AcroExpExtractor_Yet_Another_Improvement2 import AcroExpExtractor_Yet_Another_Improvement
from AcroExpExtractors.AcroExpExtractor_Original_Schwartz_Hearst import AcroExpExtractor_Original_Schwartz_Hearst

from helper import getDatasetGeneratedFilesPath,\
    get_acronym_db_path, getArticleDBPath, getArticleAcronymDBPath,\
    getCrossValidationFolds, getTrainTestData,\
    get_raw_article_db_path, get_preprocessed_article_db_path
from string_constants import FULL_WIKIPEDIA_DATASET, FOLDER_DATA
from DataCreators.ArticleAcronymDB import create_article_acronym_db_from_acronym_db
from text_preparation import full_text_preprocessing, transform_text_with_expansions_tokens
from DataCreators.AcronymDB import add_expansions_to_acronym_db
from AcroExpExtractors.AcroExpExtractor_AAAI_Schwartz_Hearst import AcroExpExtractor_AAAI_Schwartz_Hearst

logger = logging.getLogger(__name__)

text_preprocessing = full_text_preprocessing


class ParserFullWikipedia():
    def __init__(self, in_expander=None):
        self.wikiExtractorOutputPath = FOLDER_DATA+ "/FullWikipedia"

        if in_expander == "MadDog":
            from AcroExpExtractors.AcroExpExtractor_MadDog import AcroExpExtractor_MadDog
            self.default_extractor = AcroExpExtractor_MadDog
            self.dataset = FULL_WIKIPEDIA_DATASET + "_MadDog"
        else:
            #self.default_extractor = AcroExpExtractor_Original_Schwartz_Hearst
            self.default_extractor = AcroExpExtractor_AAAI_Schwartz_Hearst
            self.dataset = FULL_WIKIPEDIA_DATASET

    def processWikiDoc(self, doc_id, text, acroExp, acronymDB):
        
        try:
            acro_exp_dict = acroExp.get_acronym_expansion_pairs(text)
        except:
            logger.exception("Fatal error in acroExp.get_acronym_expansion_pairs for docId: "+doc_id+" text: " + text)
            return None
            
        if acro_exp_dict and len(acro_exp_dict) > 0:
            
            raw_text, expansions_without_spaces, acro_exp_not_found = transform_text_with_expansions_tokens(text, acro_exp_dict)
            for acro_exp in acro_exp_not_found:
                acro_not_found = acro_exp[0]
                exp_not_found = acro_exp[1]
                acro_exp_dict.pop(acro_not_found)
                logger.warning("Extracted acronym %s and expansion %s not found in text for article %s, text: %s", acro_not_found, exp_not_found, str(doc_id), raw_text)
            
            if len(expansions_without_spaces) < 1:
                logger.warning("Skipped article %s, no expansion/acronym found.", doc_id)
                return None
            
            pre_processed_text = text_preprocessing(raw_text, expansions_without_spaces)

            add_expansions_to_acronym_db(acronymDB, doc_id, acro_exp_dict)

            logger.debug("Finished processing document "+doc_id + " found " + str(len(acro_exp_dict)) + " acronyms with expansion.")
            
            return raw_text, pre_processed_text
            
        logger.debug("Finished processing document: "+doc_id + " no acronyms with expansion were found.")
        return None

    def processWikiFile(self, filePath, acroExp, acronym_db=None, raw_article_db=None, preprocessed_article_db=None, previous_processed_ids={}):
        if acronym_db is None:
            acronym_db = {}
            
        if raw_article_db is None:
            raw_article_db = {}
            
        if preprocessed_article_db is None:
            preprocessed_article_db = {}
            
        logger.debug("Processing file: "+filePath)
        with open(filePath) as file:
            try:
                soupOut = BeautifulSoup(markup=file, features="lxml")
                for doc in soupOut.findAll(name="doc"):
                    attributes = doc.attrs
                    docId = attributes["id"]
                    
                    if docId in previous_processed_ids:
                        continue
                    
                    docUrl = attributes["url"]
                    docTitle = attributes["title"]
                    
                    text = doc.text
                    process_wiki_doc_out = self.processWikiDoc(docId, text, acroExp, acronym_db)
                    
                    # Text is returned only if acronyms and expansions are found
                    if process_wiki_doc_out is not None:
                        raw_article_db[docId] = process_wiki_doc_out[0]
                        preprocessed_article_db[docId] = process_wiki_doc_out[1]

                logger.debug("Finished processing file: %s", filePath)
            except:
                logger.exception("Error processing file: %s", filePath)               
                            
        return acronym_db, raw_article_db, preprocessed_article_db
                    
    def _mergeDicts(self, dictList):
        newDict = {}
    
        for d in dictList:
            for key, value in d.items():
                newDictValue = newDict.setdefault(key, [])
                newDictValue.extend(value)
        return newDict

    def _extendAcronymDB(self, baseAcronymDB, resultAcronymDB):
        for key, value in resultAcronymDB.items():
            newDictValue = baseAcronymDB.setdefault(key, [])
            newDictValue.extend(value)
            #when using SQLite we have to make sure the value is assigned
            baseAcronymDB[key] = newDictValue
            

    def multiProcessWikiFiles(self, filePathsList, acroExp, processes_number, acronym_db, raw_article_db, preprocessed_article_db, previous_processed_ids):
        
        
        tasksNum = len(filePathsList)
        partialFunc = functools.partial(self.processWikiFile, acroExp=acroExp, previous_processed_ids=previous_processed_ids)
        with ProcessPoolExecutor(processes_number) as process_pool:
            with tqdm(total=tasksNum) as pbar:
                for i, r in tqdm(enumerate(process_pool.map(partialFunc, filePathsList, chunksize=1))):
                    resultAcronymDB = r[0]
                    self._extendAcronymDB(acronym_db, resultAcronymDB)
                    
                    result_raw_article_db = r[1]
                    raw_article_db.update(result_raw_article_db)
                    result_preprocessed_article_db = r[2]
                    preprocessed_article_db.update(result_preprocessed_article_db)
                    pbar.update()         
                    
        return acronym_db, raw_article_db, preprocessed_article_db


    def _processWikiFolderAux(self, filePathsList, acroExp, processes_number, acronym_db, raw_article_db, preprocessed_article_db):
        previous_processed_ids = set(preprocessed_article_db.keys())
        
        if processes_number > 1:
            return self.multiProcessWikiFiles(filePathsList, acroExp, processes_number, acronym_db=acronym_db, raw_article_db=raw_article_db, preprocessed_article_db=preprocessed_article_db, previous_processed_ids=previous_processed_ids)
        else:
            tasksNum = len(filePathsList)
            with tqdm(total=tasksNum) as pbar:
                for filePath in tqdm(enumerate(filePathsList)):
                    self.processWikiFile(filePath[1], acroExp, acronym_db=acronym_db, raw_article_db=raw_article_db, preprocessed_article_db=preprocessed_article_db, previous_processed_ids=previous_processed_ids)
                    pbar.update()

    def processWikiFolder(self,
                         startdir,
                         acroExp,
                         processes_number=1,
                         storageType=None):
        filePathsList = []
        
        
        directories = os.listdir(startdir)
        for wikiDir in directories:
            fullPathWikiDir = os.path.join(startdir, wikiDir)
            if os.path.isdir(fullPathWikiDir):
                for file in os.listdir(fullPathWikiDir):
                    filePath = os.path.join(fullPathWikiDir, file)
                    filePathsList.append(filePath)
        
        if storageType == "SQLite":
            with SqliteDict(get_raw_article_db_path(self.dataset),
                            flag='c',
                            autocommit=True) as raw_article_db, \
                            SqliteDict(get_preprocessed_article_db_path(self.dataset),
                            flag='c',
                            autocommit=True) as preprocessed_article_db, \
                            SqliteDict(get_acronym_db_path(self.dataset),
                            flag='c',
                            autocommit=True) as acronym_db:
                self._processWikiFolderAux(filePathsList, acroExp, processes_number, acronym_db=acronym_db, raw_article_db=raw_article_db, preprocessed_article_db=preprocessed_article_db)
        else:
            acronym_db= {}
            raw_article_db= {}
            preprocessed_article_db = {}
            self._processWikiFolderAux(filePathsList, acroExp, processes_number, acronym_db=acronym_db, raw_article_db=raw_article_db, preprocessed_article_db=preprocessed_article_db)

        return acronym_db, raw_article_db, preprocessed_article_db



    def make_dbs(self,
                processes_number=8,
                storageType="SQLite"):
        
        foldsNum = 5
        ourExtractor = False
        if ourExtractor:
            acroExpExtractor = AcroExpExtractor_Yet_Another_Improvement
        else:
            acroExpExtractor = self.default_extractor
            
        if processes_number > 1:
            if platform.system() != "Linux":
                mp.set_start_method('spawn')
            BaseManager.register('SimpleClass', acroExpExtractor)
            manager = BaseManager()
            manager.start()
            inst = manager.SimpleClass()
        else:
            inst = acroExpExtractor()
        
        acronym_db, raw_article_db, preprocessed_article_db = self.processWikiFolder(self.wikiExtractorOutputPath,
                                                      inst,
                                                      processes_number=processes_number,
                                                      storageType=storageType)
        logger.debug("Finished processing Wikipedia")

        if storageType == "SQLite":
        
            with SqliteDict(get_acronym_db_path(self.dataset),
                            flag='r',
                            autocommit=True) as acronymDB:

                articleIDToAcronymExpansions = create_article_acronym_db_from_acronym_db(
                    acronymDB)
            logger.debug("Storing into SQLite")
                    
            with SqliteDict(getArticleAcronymDBPath(self.dataset),
                            flag='n',
                            autocommit=True) as articleAcroExpan:
                for article,  acroExp in articleIDToAcronymExpansions.items():
                    articleAcroExpan[article] = acroExp
        else:
        #shuffledArticleDB = createShuffledArticleDB(articleDB)
            articleIDToAcronymExpansions = create_article_acronym_db_from_acronym_db(
                    acronymDB)
            logger.debug("Storing into pickle")
            pickle.dump(raw_article_db, open(get_raw_article_db_path(self.dataset), "wb"), protocol=2)
            pickle.dump(preprocessed_article_db, open(get_preprocessed_article_db_path(self.dataset), "wb"), protocol=2)
            pickle.dump(acronym_db, open(get_acronym_db_path(self.dataset), "wb"), protocol=2)
            pickle.dump(articleIDToAcronymExpansions, open(
                getArticleAcronymDBPath(self.dataset), "wb"), protocol=2)
        
        #pickle.dump(shuffledArticleDB, open(    
        #    getArticleDBShuffledPath(FULL_WIKIPEDIA_DATASET), "wb"), protocol=2)

        logger.debug("Generate Folds")
        generatedFilesFolder = getDatasetGeneratedFilesPath(self.dataset)

        if storageType == "SQLite":
        
            with SqliteDict(getArticleDBPath(self.dataset),
                        flag='r',
                        autocommit=True) as articleDB:
                articleDBKeys = set(articleDB.keys())
                
        else:
            articleDBKeys = articleDB.keys()

        newTrain, newTest = getTrainTestData(articleDBKeys, 0.70)
        pickle.dump(newTrain, open(generatedFilesFolder + 'train_articles.pickle', "wb"), protocol=2)
        pickle.dump(newTest, open(generatedFilesFolder + 'test_articles.pickle', "wb"), protocol=2)
        
        newFolds = getCrossValidationFolds(newTrain, foldsNum)

        foldsFilePath = generatedFilesFolder + str(foldsNum) + "-cross-validation.pickle"
        pickle.dump(newFolds, open(foldsFilePath, "wb"), protocol=2)

if __name__ == "__main__":
    parser = ParserFullWikipedia()
    parser.make_dbs()
