"""
collection of functions used to manipulate the articledb dictionary
articledb is a dictionary in the format (article_id: article_text)
"""
import pickle
import csv
import sys
from collections import OrderedDict
import random


from Logger import logging

from sqlitedict import SqliteDict

from string_constants import file_scraped_articles_list, FILE_ARTICLE_DB
from helper import getArticleDBPath, ExecutionTimeObserver 
from text_preparation import toUnicode, get_expansion_without_spaces

logger = logging.getLogger(__name__)


def getArticleDB(datasetName):
    articleDBPath = getArticleDBPath(datasetName)    
    articleDB = load(path=articleDBPath)
    return articleDB

def createFromScrapedArticles():
    logger.info("Creating ArticleDB")
    csv.field_size_limit(sys.maxsize)

    articleDB = {}
    loaded_articles = 0
    for article_file in file_scraped_articles_list:
        # open as csv file with headers
        article_csv = csv.DictReader(open(article_file, "rb"), delimiter=",")

        for row in article_csv:
            article_id = toUnicode(row["article_id"])
            articleDB[article_id] = toUnicode(row["article_text"])
            loaded_articles += 1
            if(loaded_articles % 10000 == 0):
                logger.debug("loaded %d articles", loaded_articles)

    dump(articleDB, path=FILE_ARTICLE_DB)
    logger.info("Dumped ArticleDB successfully")


def dump(articleDB, path):
    pickle.dump(
        articleDB, open(path, "wb"), protocol=2)


def load(path=FILE_ARTICLE_DB, storageType=None):
    """
    Returns: dictionary in the format (article_id: article_text)
    """
    logger.debug("loading articleDB from %s", path)
    if storageType == "SQLite":
        return SqliteDict(path, flag='r')
    else:
        try:
            with open(path, "rb") as article_file:
                articles_dict = pickle.load(article_file)
                return articles_dict
        except pickle.UnpicklingError:
            logger.warning("File at %s is not a pickle file, trying to load sqlite instead.", path)
            return SqliteDict(path, flag='r')

def addArticles(articleDB, articles):
    """
    takes in array of [article_id, article_text] entries
    and adds them to articleDB
    returns articleDB with added articles
    """

    for [article_id, article_text] in articles:
        articleDB[article_id] = article_text

    return articleDB


def createShuffledArticleDB(articleDB):
    items = list(articleDB.items())
    random.Random(1337).shuffle(items)
    shuffledArticleDB = OrderedDict(items)
    return shuffledArticleDB


def get_preprocessed_article_db(raw_article_db, acronym_article_db, train_article_ids, test_article_ids, text_preprocessor):
    train_execution_time_observer = ExecutionTimeObserver()
    test_execution_time_observer = ExecutionTimeObserver()
    
    preprocessed_article_db = {}
    for aid, text in raw_article_db.items():
        expansions_without_spaces = [get_expansion_without_spaces(exp) for exp in acronym_article_db[aid].values()]
        testExecutionTimeObserver = ExecutionTimeObserver()
        testExecutionTimeObserver.start()            
        preprocessedText = text_preprocessor(text, expansions_without_spaces)
        testExecutionTimeObserver.stop()
        preprocessed_article_db[aid] = preprocessedText
    
        if aid in train_article_ids:
            train_execution_time_observer += testExecutionTimeObserver
        elif aid in test_article_ids: 
            test_execution_time_observer += testExecutionTimeObserver
        
    avgExecutionTime = test_execution_time_observer.getTime() / len(test_article_ids)
    
    logger.info("Train Execution time: %s", str(train_execution_time_observer))
    logger.info("Average Test Execution time: %s", str(avgExecutionTime))
    return preprocessed_article_db, train_execution_time_observer, avgExecutionTime

if __name__ == "__main__":
    createFromScrapedArticles()
