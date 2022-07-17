
import os
import pickle
import random
import re

from bs4 import BeautifulSoup
from sqlitedict import SqliteDict

from AcroExpExtractors.AcroExpExtractor_Original_Schwartz_Hearst import (
    AcroExpExtractor_Original_Schwartz_Hearst,
)
from DataCreators.AcronymDB import add_expansions_to_acronym_db
from DataCreators.ArticleAcronymDB import create_article_acronym_db_from_acronym_db
from DatasetParsers import FullWikipedia
from Logger import logging
from helper import (
    getDatasetGeneratedFilesPath,
    get_acronym_db_path,
    getArticleAcronymDBPath,
    getCrossValidationFolds,
    getTrainTestData,
    get_raw_article_db_path,
    get_preprocessed_article_db_path,
)
from string_constants import REUTERS_DATASET, folder_reuters_articles
from text_preparation import transform_text_with_expansions_tokens


#from AcronymExpanders.Expander_fromText_v2 import Expander_fromText_v2
#from AcronymExpanders.Expander_fromText_v3 import Expander_fromText_v3
logger = logging.getLogger(__name__)
#from AcronymExtractors.AcronymExtractor_v3_small import AcronymExtractor_v3_small
#from AcronymExtractors.AcronymExtractor_v4 import AcronymExtractor_v4


def _removeMarkup(text):
    textWithoutMarkup = re.sub(u"\\<e\\>", u"", text)
    textWithoutMarkup = re.sub(u"\\<\\/e\\>", u"", textWithoutMarkup)
    return textWithoutMarkup

def _fixArticlesParagraphs(text):
    #Reuters news articles are formated with may line breaks,
    # the paragraphs start with some spaces so we only need
    # line breaks instead of those spaces
    return text.replace('\n', ' ').replace('     ', '\n')

def _create_article_and_acronym_db():
    # acronymExtractor = AcronymExtractor_v3_small()
    # acronymExpander = Expander_fromText_v2()
   
    # Replaced by Leahs:
    #acronymExtractor = AcronymExtractor_v4()
    #acronymExpander = Expander_fromText_v3()
    
    #acroExpExtractor = AcroExpExtractor_Yet_Another_Improvement()
    acroExpExtractor = AcroExpExtractor_Original_Schwartz_Hearst()
    acroExp = acroExpExtractor.get_acronym_expansion_pairs
   
    raw_article_db = {}
    preprocessed_article_db = {}
    acronymDB = {}

    for fileName in os.listdir(folder_reuters_articles):
        if not fileName.endswith('.sgm'):
            continue
        filePath = os.path.join(folder_reuters_articles, fileName)
        print(filePath)
        
        f = open(filePath, 'r', errors='ignore')
        
        data= f.read()
        # BODY tag is not parsed by BeautifulSoup, needs to be replaced by something else 
        data = re.sub('<BODY', '<CONTENT', data)
        soup = BeautifulSoup(data,"lxml")
        reuters_list = soup.findAll('reuters')
        for reuters in reuters_list:
            if not reuters.content:
                continue
            article_id = reuters['newid']
            title = reuters.title.text
            body = reuters.content.text
            fixed_body = _fixArticlesParagraphs(body)

            articleText = title + '\n' + fixed_body
            
            acro_exp_dict = acroExp(articleText)
            if acro_exp_dict and len(acro_exp_dict) > 0:
                raw_text, expansions_without_spaces, acro_exp_not_found = transform_text_with_expansions_tokens(articleText, acro_exp_dict)
                for acro_exp in acro_exp_not_found:
                    acro_not_found = acro_exp[0]
                    exp_not_found = acro_exp[1]
                    acro_exp_dict.pop(acro_not_found)
                    logger.warning("Extracted acronym %s and expansion %s not found in text for article %s, text: %s", acro_not_found, exp_not_found, str(article_id), raw_text)
                
                if len(expansions_without_spaces) < 1:
                    logger.warning("Skipped article %s, no expansion/acronym found.", article_id)
                    continue
                
                pre_processed_text = FullWikipedia.text_preprocessing(raw_text, expansions_without_spaces)
                add_expansions_to_acronym_db(acronymDB, article_id, acro_exp_dict)

                raw_article_db[article_id] = raw_text
                preprocessed_article_db[article_id] = pre_processed_text

    return acronymDB, raw_article_db, preprocessed_article_db

def saveToSQLite(oldDict, path):
    with SqliteDict(path,
                     flag='n',
                     autocommit=True) as sqlDict:
        for key, value in oldDict.items():
            sqlDict[key] = value

def make_dbs(createFolds=False, storageType="SQLite"):
    foldsNum = 5
    
    acronymDB, raw_article_db, preprocessed_article_db = _create_article_and_acronym_db()

    articleIDToAcronymExpansions = create_article_acronym_db_from_acronym_db(
        acronymDB)

    if storageType == "SQLite":
        #logger.debug("Storing into SQLite")

        saveToSQLite(raw_article_db, get_raw_article_db_path(REUTERS_DATASET))
        saveToSQLite(preprocessed_article_db, get_preprocessed_article_db_path(REUTERS_DATASET))
        saveToSQLite(acronymDB, get_acronym_db_path(REUTERS_DATASET))
        saveToSQLite(articleIDToAcronymExpansions, getArticleAcronymDBPath(REUTERS_DATASET))
        
    else:
        #shuffledArticleDB = _createShuffledArticleDB(articleDB)
    
        pickle.dump(raw_article_db, open(get_raw_article_db_path(REUTERS_DATASET), "wb"), protocol=2)
        pickle.dump(preprocessed_article_db, open(get_preprocessed_article_db_path(REUTERS_DATASET), "wb"), protocol=2)
        pickle.dump(acronymDB, open(get_acronym_db_path(REUTERS_DATASET), "wb"), protocol=2)
        pickle.dump(articleIDToAcronymExpansions, open(
            getArticleAcronymDBPath(REUTERS_DATASET), "wb"), protocol=2)

    #pickle.dump(shuffledArticleDB, open(
    #    file_reuters_articleDB_shuffled, "wb"), protocol=2)

    if createFolds:
        generatedFilesFolder = getDatasetGeneratedFilesPath(REUTERS_DATASET)
        
        # New train, test and folds
        keys = list(raw_article_db.keys())
        random.shuffle(keys)
        
        newTrain, newTest = getTrainTestData(keys, 0.70)
        pickle.dump(newTrain, open(generatedFilesFolder + 'train_articles.pickle', "wb"), protocol=2)
        pickle.dump(newTest, open(generatedFilesFolder + 'test_articles.pickle', "wb"), protocol=2)
        
        newFolds = getCrossValidationFolds(newTrain, foldsNum)
    
        foldsFilePath = generatedFilesFolder + str(foldsNum) + "-cross-validation_new.pickle"
        pickle.dump(newFolds, open(foldsFilePath, "wb"), protocol=2)


def _classToIndex(cls):
    return int(cls[1:]) - 1


def _fileNameToAcronym(fileName):
    return fileName.split("_")[0]

if __name__ == "__main__":
    make_dbs()
