from concurrent.futures import ProcessPoolExecutor
import csv
import pickle
import re

import regex
from tqdm import tqdm

from DataCreators.ArticleAcronymDB import create_article_acronym_db_from_acronym_db
from DatasetParsers.ScienceWISE_Extractor import arxivIDToFilename
from Logger import logging
from helper import (
    getDatasetGeneratedFilesPath,
    get_acronym_db_path,
    getArticleAcronymDBPath,
    getCrossValidationFolds,
    getTrainTestData,
    ExecutionTimeObserver,
    get_preprocessed_article_db_path,
    get_raw_article_db_path,
)
from string_constants import (
    SCIENCE_WISE_DATASET,
    file_ScienceWise_index_train,
    folder_scienceWise_abstracts,
    file_ScienceWise_index_train_noabbr,
    file_ScienceWise_index_test,
)
from text_preparation import get_expansion_without_spaces, text_word_tokenization


logger = logging.getLogger(__name__)
errorCount = 0


def _addArticlesAndAcronyms(file_ScienceWise, articleDB, acronymDB, preProcessingFunc, trainArticles = []):
    articleIDList = []
    i = 0
    with open(file_ScienceWise, "r") as file:
        
        reader = csv.DictReader(file, delimiter=",", fieldnames=["ARXIV_ID", "ACRONYM", "ID", "DEFINITION"])
        for line in reader:
            articleID = line["ARXIV_ID"]
            acronym = line["ACRONYM"]
            acronymID = line["ID"]
            expansion = re.sub('-','', line["DEFINITION"])
            expansionWithoutSpaces = get_expansion_without_spaces(expansion)
            
            
            if i % 100 == 0:
                logger.info("Processed "+str(i)+" expansions")
            i = i + 1
            

            if(articleID in trainArticles):
                logger.warning("Test article already exists in train set: " + articleID)
                continue
            
            if(articleID not in articleDB):
                filename = arxivIDToFilename(articleID) + ".txt"
                with open(folder_scienceWise_abstracts + filename,"r") as abstractFile:
                    content = abstractFile.read()
                    content = re.sub('-','', content)
            else:
                content =  articleDB[articleID]
                
            prePocessedArticle = preProcessingFunc(content, acronym.strip(), expansion, expansionWithoutSpaces)
            if prePocessedArticle == None:
                logger.warning("Skipped due to error: " + str(line))
                continue
                
            articleDB[articleID] = prePocessedArticle
            
            if articleID not in articleIDList:
                articleIDList.append(articleID)

            if acronym not in acronymDB:
                acronymDB[acronym] = [];
            acronymDB[acronym].append([expansion, articleID]);
    return articleIDList


def _addArticlesAndAcronymsMultiProcess(file_ScienceWise):
    fileInfoDict = {}
    with open(file_ScienceWise, "r") as file:
        i = 0
        reader = csv.DictReader(file, delimiter=",", fieldnames=["ARXIV_ID", "ACRONYM", "ID", "DEFINITION"])
        for line in reader:
            i = i +1
            articleID = line["ARXIV_ID"]
            acronym = line["ACRONYM"]
            acronymID = line["ID"]
            expansion = re.sub('-','', line["DEFINITION"])
            
            if articleID not in fileInfoDict:
                fileInfoDict[articleID] = []
            
            fileInfoDict[articleID].append([acronym, expansion])
            

    results = []
    tasksNum = len(fileInfoDict.items())
    with ProcessPoolExecutor(8) as process_pool:
        with tqdm(total=tasksNum) as pbar:
            for i, r in tqdm(enumerate(process_pool.map(_processArticles, fileInfoDict.items(), chunksize=1))):
                pbar.update()
                results.append(r)

    articleDB = {r[0]:r[1] for r in results}
    
    acronymDBDicts = [r[2] for r in results]
    acronymDB = _mergeDicts(acronymDBDicts)
    
    return articleDB, acronymDB



def _mergeDicts(dictList):
    newDict = {}
 
    for d in dictList:
        for key, value in d.items():
            if key not in newDict:
                newDict[key] = []
            newDict[key] = newDict[key] + value
    return newDict


def _trainPreProcessingFunc(text, acronym, expansion, expansionWithoutSpaces):
    regexExpansion = re.compile("\\b" + re.escape(expansion) + "\\b", re.IGNORECASE)
    errors = len(expansion) // 2
    
    searchR = regexExpansion.search(text)
    if searchR == None:
    
        regexFuzzyExpansion = regex.compile("\\m(?:"+re.escape(expansion)+"){i<=3,d<=3,s<=3,3i+1d+2s<"+str(len(expansion))+"}\\M", regex.IGNORECASE + regex.BESTMATCH)        
        searchR = regexFuzzyExpansion.search(text)
        if searchR == None:
            regexFuzzyExpansion = regex.compile("\\m(?:"+re.escape(expansion)+"){i<="+str(errors)+",d<="+str(errors)+",s<="+str(errors)+",3i+1d+2s<"+str(len(expansion))+"}\\M", regex.IGNORECASE + regex.BESTMATCH)        
            searchR = regexFuzzyExpansion.search(text)

            if searchR == None:
                logger.error("Train Expansion "+ expansion+" not found in text: " + text)
                #The ENHANCEMATCH flag makes fuzzy matching attempt to improve the fit of the next match that it finds.
                #The BESTMATCH flag makes fuzzy matching search for the best match instead of the next match.
                #regexFuzzyExpansion = regex.compile("(?:"+re.escape(expansion)+"){e<=2}", regex.IGNORECASE + regex.BESTMATCH)
                
                #searchR = regexFuzzyExpansion.search(text)
                #if searchR == None:
                #    logger.critical("NO expa found")
                #else:
                #    no_expa = len(searchR)
                #    if no_expa != 1:
                #        logger.critical(no_expa)
                        
                #newTextWithoutSpaces = regexFuzzyExpansion.sub(expansionWithoutSpaces, text)
                #logger.error("New text = " + newTextWithoutSpaces)
                
                return None
            else:
                logger.warning("Train Expansion "+ expansion+" not found, but we found "+searchR.captures()[0]+" instead in text: " + text)
        #else:
        #    logger.warning("Train Expansion "+ expansion+" not found, but we found "+searchR.captures()[0]+" instead in text: " + text)
        if searchR != None:
            if len(searchR.captures()) > 1:
                logger.error("if len(searchR) > 1: Train Expansion "+ expansion)
        
            textExpansionsWithoutSpaces = re.sub(re.escape(searchR.captures()[0]),expansionWithoutSpaces, text)
    else:
        textExpansionsWithoutSpaces = re.sub(re.escape(searchR.group()) ,expansionWithoutSpaces, text)

    
    #textExpansionsWithoutSpaces = regexFuzzyExpansion.sub(expansionWithoutSpaces, text)
    regexAcronym = re.compile(re.escape(acronym).upper()+"s?")

    processedText = regexAcronym.sub(" "+expansionWithoutSpaces+" ", textExpansionsWithoutSpaces)

    if searchR == None and text == processedText:
        logger.critical(">>>>No expansion and no acronym found in text for expansion: " +expansion+" acronym: " + acronym +" text: "+text)
        global errorCount
        errorCount = errorCount + 1
        return None
    return processedText


def _testPreProcessingFunc(text, acronym, expansion, expansionWithoutSpaces):
    regexExpansion = re.compile("\\b" + re.escape(expansion) + "\\b", re.IGNORECASE)
    regexAcronym = re.compile("\\b" + re.escape(acronym).upper()+"s?"+ "\\b")
    
    #textExpansionsWithoutSpaces = regexExpansion.sub(expansionWithoutSpaces, text)
    
    
    searchR = regexExpansion.search(text)
    if searchR == None:
    
        regexFuzzyExpansion = regex.compile("\\m(?:"+re.escape(expansion)+"){i<=2,d<=2,s<=2,3i+1d+2s<"+str(len(expansion)//2)+"}\\M", regex.IGNORECASE + regex.BESTMATCH)        
        searchR = regexFuzzyExpansion.search(text)
        if searchR == None:
            #logger.debug("Test Expansion "+ expansion+" not found in text: " + text)
            textExpansionsWithoutSpaces = text
        else:
            logger.warn("Test Expansion "+ expansion+" not found, but we found "+searchR.captures()[0]+" instead in text: " + text)
            textExpansionsWithoutSpaces = re.sub(re.escape(searchR.captures()[0]),expansionWithoutSpaces, text)
    else:
        textExpansionsWithoutSpaces = re.sub(re.escape(searchR.group()) ,expansionWithoutSpaces, text)

    
    """ We have to add spaces because acromyns may have prefixs and sufixs apart from 's'
    otherwise we may not be able to identify the expansion in text as a token (see _processArticles function)
    also because we tokenize there is no problem in having a sequence of more white spaces as it will reduce to one
    """
    textWithoutAcronym = regexAcronym.sub(" " +expansionWithoutSpaces+" ", textExpansionsWithoutSpaces)
    if text == textWithoutAcronym and searchR == None:
        logger.error("Test Acronym "+ acronym+" not expansion "+expansion+" not found in text: " + text)
        return None
    
    return textWithoutAcronym

def _trainNoAbbrPreProcessingFunc(text, _, expansion, expansionWithoutSpaces):
    regexExpansion = re.compile("\\b" + re.escape(expansion) + "\\b", re.IGNORECASE)
    errors = len(expansion) // 2
    
    searchR = regexExpansion.search(text)
    if searchR == None:
    
        regexFuzzyExpansion = regex.compile("\\m(?:"+re.escape(expansion)+"){i<=3,d<=3,s<=3,3i+1d+2s<"+str(len(expansion))+"}\\M", regex.IGNORECASE + regex.BESTMATCH)        
        searchR = regexFuzzyExpansion.search(text)
        if searchR == None:
            regexFuzzyExpansion = regex.compile("\\m(?:"+re.escape(expansion)+"){i<="+str(errors)+",d<="+str(errors)+",s<="+str(errors)+",3i+1d+2s<"+str(len(expansion))+"}\\M", regex.IGNORECASE + regex.BESTMATCH)        
            searchR = regexFuzzyExpansion.search(text)

            if searchR == None:
                logger.error("Train Expansion "+ expansion+" not found in text: " + text)
                
                return None
            else:
                logger.warning("Train Expansion "+ expansion+" not found, but we found "+searchR.captures()[0]+" instead in text: " + text)
        #else:
        #    logger.warning("Train Expansion "+ expansion+" not found, but we found "+searchR.captures()[0]+" instead in text: " + text)
        if searchR != None:
            if len(searchR.captures()) > 1:
                logger.error("if len(searchR) > 1: Train Expansion "+ expansion)
        
            textExpansionsWithoutSpaces = re.sub(re.escape(searchR.captures()[0]),expansionWithoutSpaces, text)
    else:
        textExpansionsWithoutSpaces = re.sub(re.escape(searchR.group()) ,expansionWithoutSpaces, text)


    if searchR == None:
        logger.critical(">>>>No expansion found in text for expansion: " +expansion+" text: "+text)
        global errorCount
        errorCount = errorCount + 1
        return None
    return textExpansionsWithoutSpaces

def _processArticles(args):
    articleID = args[0]
    listAcroExpan = args[1]
    acronymDB = {}
    filename = arxivIDToFilename(articleID) + ".txt"
    
    with open(folder_scienceWise_abstracts + filename,"r") as abstractFile:
        content = abstractFile.read()
        articleText = re.sub('-','', content)

    for acroExpan in listAcroExpan:
        acronym = acroExpan[0]
        expansion = acroExpan[1]
        expansionWithoutSpaces = get_expansion_without_spaces(expansion)
                
        prePocessedArticle = _trainPreProcessingFunc(articleText, acronym.strip(), expansion, expansionWithoutSpaces)
        if prePocessedArticle == None:
            logger.warning("Skipped due to error: " + str(articleID) + " acronym: " + acronym + " expansion: " + expansion)
            continue
            
        articleText = prePocessedArticle
        
        if acronym not in acronymDB:
            acronymDB[acronym] = []
        acronymDB[acronym].append([expansion, articleID])

    return articleID, articleText, acronymDB

def _create_article_and_acronym_db():
    

    testArticles = []
    """
    articleDB = {}
    acronymDB = {}
    trainArticles = []
    """
    articleDB, acronymDB = _addArticlesAndAcronymsMultiProcess(file_ScienceWise_index_train)
    
    _addArticlesAndAcronyms(file_ScienceWise_index_train_noabbr, articleDB, acronymDB, 
                            _trainNoAbbrPreProcessingFunc)


    trainArticles = list(articleDB.keys())
    global errorCount
    logger.critical("errorcount: " + str(errorCount))
    
    testArticles += _addArticlesAndAcronyms(file_ScienceWise_index_test, articleDB, acronymDB, 
                            _testPreProcessingFunc, trainArticles = trainArticles)
        

    articleIDToAcronymExpansions = create_article_acronym_db_from_acronym_db(
        acronymDB)
    accExecutionTimeObserver = ExecutionTimeObserver()

    raw_article_db = {}
    preprocessed_article_db = {}
        
    for aid, text in articleDB.items():
        if aid not in articleIDToAcronymExpansions:
            logger.critical("Removing article: No acronyms/expansions were found in article: "+aid)
            continue
        
        testExecutionTimeObserver = ExecutionTimeObserver()
        
     
        raw_article_db[aid] = text
        expansions_without_spaces = [get_expansion_without_spaces(exp) for exp in articleIDToAcronymExpansions[aid].values()]
        
        testExecutionTimeObserver.start()            
        preprocessedText = text_word_tokenization(text, expansions_without_spaces)
        testExecutionTimeObserver.stop()
        preprocessed_article_db[aid] = preprocessedText
    
        accExecutionTimeObserver += testExecutionTimeObserver
        
    avgExecutionTime = accExecutionTimeObserver.getTime() / len(preprocessed_article_db)
    logger.info("Average preprocessing execution time per document: " +  str(avgExecutionTime))
    return acronymDB, raw_article_db, preprocessed_article_db, trainArticles, testArticles, articleIDToAcronymExpansions



def make_dbs(createFolds = True):
    foldsNum = 5
    acronymDB,  raw_article_db, preprocessed_article_db, trainArticles, testArticles, articleIDToAcronymExpansions = _create_article_and_acronym_db()

    pickle.dump(raw_article_db, open(get_raw_article_db_path(SCIENCE_WISE_DATASET), "wb"), protocol=2)
    pickle.dump(preprocessed_article_db, open(get_preprocessed_article_db_path(SCIENCE_WISE_DATASET), "wb"), protocol=2)

    #shuffledArticleDB = _createShuffledArticleDB(articleDB)

    pickle.dump(acronymDB, open(get_acronym_db_path(SCIENCE_WISE_DATASET), "wb"), protocol=2)
    pickle.dump(articleIDToAcronymExpansions, open(
        getArticleAcronymDBPath(SCIENCE_WISE_DATASET), "wb"), protocol=2)
    
    if createFolds:
        generatedFilesFolder = getDatasetGeneratedFilesPath(SCIENCE_WISE_DATASET)
    
        pickle.dump(trainArticles, open(generatedFilesFolder + 'train_articles.pickle', "wb"), protocol=2)
        pickle.dump(testArticles, open(generatedFilesFolder + 'test_articles.pickle', "wb"), protocol=2)
    
        foldsFilePath = generatedFilesFolder + str(foldsNum) + "-cross-validation.pickle"
    
        folds = getCrossValidationFolds(trainArticles, foldsNum)
        pickle.dump(folds, open(foldsFilePath, "wb"), protocol=2)
        
        # New train, test and folds
        newTrain, newTest = getTrainTestData(raw_article_db.keys(), 0.70)
        pickle.dump(newTrain, open(generatedFilesFolder + 'train_articles_new.pickle', "wb"), protocol=2)
        pickle.dump(newTest, open(generatedFilesFolder + 'test_articles_new.pickle', "wb"), protocol=2)
        
        newFolds = getCrossValidationFolds(newTrain, foldsNum)

        foldsFilePath = generatedFilesFolder + str(foldsNum) + "-cross-validation_new.pickle"
        pickle.dump(newFolds, open(foldsFilePath, "wb"), protocol=2)

if __name__ == "__main__":
    make_dbs(False)

