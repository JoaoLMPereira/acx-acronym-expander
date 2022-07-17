"""
convert <e> to ( or ""
convert </e> to ) or ""
regex extend: BLM (Bloom's syndrome protein)
"""
import os
import pickle
import re

import arff

from DataCreators.ArticleAcronymDB import create_article_acronym_db_from_acronym_db
from DataCreators.ArticleDB import get_preprocessed_article_db
from Logger import logging
from helper import (
    getDatasetPath,
    getTrainTestData,
    getCrossValidationFolds,
    get_acronym_db_path,
    getArticleAcronymDBPath,
    get_preprocessed_article_db_path,
    get_raw_article_db_path,
    getDatasetGeneratedFilesPath,
)
from string_constants import folder_msh_arff, MSH_SOA_DATASET, MSH_ORGIN_DATASET
from text_preparation import transform_text_with_exp_tokens, text_word_tokenization


logger = logging.getLogger(__name__)


def _removeMarkup(text):
    textWithoutMarkup = re.sub(u"\\<e\\>", u"", text)
    textWithoutMarkup = re.sub(u"\\<\\/e\\>", u"", textWithoutMarkup)
    return textWithoutMarkup


def _create_article_and_acronym_db():    
    articleDB = {}
    acronymDB = {}
    
    acronyms_file = [line.strip().split('\t') for line in open(getDatasetPath(MSH_SOA_DATASET) + 'acronyms.txt')]
    acronym_code_to_expansion = {acronym_code:expansion for _, acronym_code, expansion in acronyms_file}
    for fileName in os.listdir(folder_msh_arff):
        filePath = os.path.join(folder_msh_arff, fileName)
        try:
            file_reader = arff.load(open(filePath, "rt"))
            # the iterator needs to be called for the self.relation part to be
            # initialized
            lines = list(file_reader['data'])
            # storing all acronyms as uppercase values
            acronym = _fileNameToAcronym(fileName).upper()
            
            article_id_expansions = []
            
            for line in lines:
                pmid = str(line[0])
                text = line[1]
                #cuid = cuids[_classToIndex(line[2])]
                expansion = acronym_code_to_expansion.get(acronym+line[2])
    
    
                if not expansion:
                    logger.info("Skipping pmid %s, no expansion for acronym %s.", pmid, acronym)
                    continue
                
                raw_text = articleDB.get(pmid, _removeMarkup(text))
                if pmid not in articleDB:
                    articleDB[pmid] = _removeMarkup(text)
                else:
                    logger.debug("pmid %s already exists for acronym %s.", pmid, acronym)

                
                expansion = expansion.strip()
                
                text_with_expansion, successs, _ = transform_text_with_exp_tokens(acronym, expansion, raw_text)
                if not successs:
                    logger.error("Acronym %s and expansion %s not found in pmid %s", acronym, expansion, pmid)
                    continue
                
                articleDB[pmid] = text_with_expansion
                article_id_expansions.append([expansion, pmid])
            if (acronym in acronymDB):
                logger.error("acronym already present in acronymDB")
            else:
                acronymDB[acronym] = article_id_expansions
        except Exception:
            logger.exception("File %s failed to load", fileName, exc_info=True)
                

    return acronymDB, articleDB




def make_dbs(create_folds=False):
    folds_num = 5
    
    acronymDB, articleDB = _create_article_and_acronym_db()

    #removed acronymDB = applyManualCorrections(acronymDB)

    articleIDToAcronymExpansions = create_article_acronym_db_from_acronym_db(
        acronymDB)

    pickle.dump(articleDB, open(get_raw_article_db_path(MSH_ORGIN_DATASET), "wb"), protocol=2)
    pickle.dump(acronymDB, open(get_acronym_db_path(MSH_ORGIN_DATASET), "wb"), protocol=2)
    pickle.dump(articleIDToAcronymExpansions, open(
        getArticleAcronymDBPath(MSH_ORGIN_DATASET), "wb"), protocol=2)
    
    if create_folds:
        generatedFilesFolder = getDatasetGeneratedFilesPath(MSH_ORGIN_DATASET)
        
        # New train, test and folds
        train_ids, test_ids = getTrainTestData(articleDB.keys(), 0.70)
        pickle.dump(train_ids, open(generatedFilesFolder + 'train_articles.pickle', "wb"), protocol=2)
        pickle.dump(test_ids, open(generatedFilesFolder + 'test_articles.pickle', "wb"), protocol=2)
        
        newFolds = getCrossValidationFolds(train_ids, folds_num)

        foldsFilePath = generatedFilesFolder + str(folds_num) + "-cross-validation.pickle"
        pickle.dump(newFolds, open(foldsFilePath, "wb"), protocol=2)
    else:
        train_ids = pickle.load(open(generatedFilesFolder + 'train_articles.pickle', "rb"))
        test_ids = pickle.load(open(generatedFilesFolder + 'train_articles.pickle', "rb"))

    preprocessed_artible_db, train_exec_time, test_avg_exec_time = get_preprocessed_article_db(articleDB, articleIDToAcronymExpansions, train_ids, test_ids, text_word_tokenization)
    
    pickle.dump(preprocessed_artible_db, open(get_preprocessed_article_db_path(MSH_ORGIN_DATASET), "wb"), protocol=2)




def _classToIndex(cls):
    return int(cls[1:]) - 1


def _fileNameToAcronym(fileName):
    return fileName.split("_")[0]

if __name__ == "__main__":
    make_dbs(True)
