"""
Collection of functions used to manipulate the article acronym db dictionary 
with the format (article_id: {acronym: expansion})

"""
import logging
import pickle as pickle
from string_constants import FILE_ARTICLE_ACRONYMDB
from helper import getArticleAcronymDBPath
from sqlitedict import SqliteDict

logger = logging.getLogger(__name__)

def dump(article_info_db):
    pickle.dump(article_info_db, open(FILE_ARTICLE_ACRONYMDB, "wb"), protocol=2)


def load(path=FILE_ARTICLE_ACRONYMDB, storageType=None):
    """
    articleInfoDB is a dictionary in the format:
    (articleID: dictionary of {acronym: expansion})
    """
    logger.debug("loading acronymDB from %s" % path)
    if storageType == "SQLite":
        return SqliteDict(path, flag='r')
    else:
        try:
            return pickle.load(open(path, "rb"))
        except pickle.UnpicklingError:
            logger.warn("File at %s is not a pickle file, trying to load sqlite instead.", path)
            return SqliteDict(path, flag='r')

def getArticleAcronymDB(datasetName):
    articleAcronymDBPath = getArticleAcronymDBPath(datasetName)
    articleAcronymDB = load(path=articleAcronymDBPath)
    return articleAcronymDB


def create_article_acronym_db_from_acronym_db(acronym_db, article_acronym_db=None):
    if article_acronym_db is None:
        article_acronym_db = {}

    for acronym in acronym_db:
        for expansion, articleID in acronym_db[acronym]:
            if articleID not in article_acronym_db:
                acronym_expansions = {}
            else:
                acronym_expansions = article_acronym_db[articleID]
            acronym_expansions[acronym] = expansion
            article_acronym_db[articleID] = acronym_expansions
    return article_acronym_db

