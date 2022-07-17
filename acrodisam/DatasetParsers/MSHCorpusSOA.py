import re
import random
import logging
import pickle
from helper import getDatasetPath, getDatasetGeneratedFilesPath,\
    get_acronym_db_path, getArticleDBPath, getArticleDBShuffledPath, getArticleAcronymDBPath,\
    getCrossValidationFolds, getTrainTestData
from string_constants import MSH_SOA_DATASET
from collections import OrderedDict
from DataCreators.ArticleAcronymDB import create_article_acronym_db_from_acronym_db
from text_preparation import get_expansion_without_spaces

logger = logging.getLogger(__name__)


def _create_article_and_acronym_db():
    datasetPath = getDatasetPath(MSH_SOA_DATASET)
    
    articleDB = {}
    acronymDB = {}
    trainArticles = []
    testArticles = []
    
    acronyms_file = [line.strip().split('\t') for line in open(datasetPath + 'acronyms.txt')]
        
    train_documents = [line.strip() for line in open(datasetPath + 'training_cleaned.txt','U')]
    train_texts = [[document] + [word for word in document.split()] for document in train_documents]
        
    pmid = 0;
    for text in train_texts:
        rawText = text[0]
        for [acronym, acronym_code, expansion] in acronyms_file:
            if acronym_code in text[1:]:
                expansion = expansion.strip()
                # we do this so the expansion is considered a single token, after a tokenization
                expansionWithoutSpaces = get_expansion_without_spaces(expansion)
                textWithExpansion = rawText.replace(acronym_code.strip(), expansionWithoutSpaces)
                
                if acronym not in acronymDB:
                    acronymDB[acronym] = [];
                acronymDB[acronym].append([expansion, pmid]);
                
                articleDB[pmid] = textWithExpansion
                trainArticles.append(pmid)
                pmid += 1;
                break
    
    test_documents = [line.strip() for line in open(datasetPath + 'test_cleaned.txt','U')]

    for test_text in test_documents:
        
        tmp_acros = re.findall('([\S]*\dTEST\d*)',test_text);
        if not tmp_acros:
            #logger.error("No acronym in: \n" + test_text)
            pmid += 1;
            continue;
        tmp_acro = tmp_acros[0];
        
        acronym_code = tmp_acro[:tmp_acro.index('TEST')]
        
        valid = False
        for [acronym, acronym_code_file, expansion] in acronyms_file:
            if acronym_code == acronym_code_file:
                expansion = expansion.strip()
                # we do this so the expansion is considered a single token, after a tokenization
                expansionWithoutSpaces = get_expansion_without_spaces(expansion)
                textWithExpansion = test_text.replace(tmp_acro.strip(), expansionWithoutSpaces)

                if acronym not in acronymDB:
                    logger.error("acronym not present in acronymDB: " + acronym)
                    acronymDB[acronym] = [];
                acronymDB[acronym].append([expansion, pmid]);
                
                testArticles.append(pmid)
                articleDB[pmid] = textWithExpansion
                pmid += 1;
                valid = True
                break
        if valid == False:
            for [acronym, acronym_code_file, expansion] in acronyms_file:
                valid = False
                if acronym_code_file.startswith(acronym_code[:-1]):
                    expansion = acronym
                    # we do this so the expansion is considered a single token, after a tokenization
                    expansionWithoutSpaces = get_expansion_without_spaces(expansion)
                    textWithExpansion = test_text.replace(tmp_acro.strip(), expansionWithoutSpaces)
    
                    if acronym not in acronymDB:
                        logger.error("acronym not present in acronymDB: " + acronym)
                        acronymDB[acronym] = [];
                    acronymDB[acronym].append([expansion, pmid]);
                    
                    testArticles.append(pmid)
                    articleDB[pmid] = textWithExpansion
                    pmid += 1;
                    valid = True
                    break
        if valid == False:
            articleDB[pmid] = test_text.replace(tmp_acro,acronym_code)
            pmid += 1;
    return acronymDB, articleDB, trainArticles, testArticles


def _createShuffledArticleDB(articleDB):
    items = list(articleDB.items())
    random.Random(1337).shuffle(items)
    shuffledArticleDB = OrderedDict(items)
    return shuffledArticleDB

def make_dbs():
    foldsNum = 5
    acronymDB, articleDB, trainArticles, testArticles = _create_article_and_acronym_db()

    articleIDToAcronymExpansions = create_article_acronym_db_from_acronym_db(
        acronymDB)



    shuffledArticleDB = _createShuffledArticleDB(articleDB)
    pickle.dump(articleDB, open(getArticleDBPath(MSH_SOA_DATASET), "wb"), protocol=2)
    pickle.dump(acronymDB, open(get_acronym_db_path(MSH_SOA_DATASET), "wb"), protocol=2)
    pickle.dump(articleIDToAcronymExpansions, open(
        getArticleAcronymDBPath(MSH_SOA_DATASET), "wb"), protocol=2)
    pickle.dump(shuffledArticleDB, open(    
        getArticleDBShuffledPath(MSH_SOA_DATASET), "wb"), protocol=2)
    
    generatedFilesFolder = getDatasetGeneratedFilesPath(MSH_SOA_DATASET)

    pickle.dump(trainArticles, open(generatedFilesFolder + 'train_articles.pickle', "wb"), protocol=2)
    pickle.dump(testArticles, open(generatedFilesFolder + 'test_articles.pickle', "wb"), protocol=2)

    foldsFilePath = generatedFilesFolder + str(foldsNum) + "-cross-validation.pickle"

    folds = getCrossValidationFolds(trainArticles, foldsNum)
    pickle.dump(folds, open(foldsFilePath, "wb"), protocol=2)
    
    
    # New train, test and folds
    newTrain, newTest = getTrainTestData(articleDB.keys(), 0.70)
    pickle.dump(newTrain, open(generatedFilesFolder + 'train_articles_new.pickle', "wb"), protocol=2)
    pickle.dump(newTest, open(generatedFilesFolder + 'test_articles_new.pickle', "wb"), protocol=2)
    
    newFolds = getCrossValidationFolds(newTrain, foldsNum)

    foldsFilePath = generatedFilesFolder + str(foldsNum) + "-cross-validation_new.pickle"
    pickle.dump(newFolds, open(foldsFilePath, "wb"), protocol=2)
    
if __name__ == "__main__":
    make_dbs()
     


