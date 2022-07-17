import os
import functools
import math
import tracemalloc
import gc
from concurrent.futures import ProcessPoolExecutor
from threading import Thread

from sqlitedict import SqliteDict

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

from DataCreators import ArticleDB, ArticleAcronymDB
import logging
import text_preparation
from text_preparation import getTextWithExpansionTokens
import _pickle as pickle
from helper import LDAStructure, getArticleDBPath, getDatasetGeneratedFilesPath, ExecutionTimeObserver,\
 logOSStatus, display_top
from string_constants import file_lda_model_all, file_lda_word_corpus,\
    file_lda_bow_corpus, MSH_SOA_DATASET,\
    file_lda_model_all, file_lda_model,\
    file_lda_articleIDToLDA, file_lda_gensim_dictionary
import tempfile

logger = logging.getLogger(__name__)


class USESAVED:
    none = -1
    word_corpus = 0
    dictionary = 1
    bow_corpus = 2
    lda_model = 3


def load(path=file_lda_model_all):
    """
    Returns: SavedLDAModel object
    """

    logger.info("Loading LDA model from " + path)
    return pickle.load(open(path, "rb"))

# todo: put "_" in front of all private methods


def serialGetWordCorpus(articleDB, word_corpus, expansionTokens = False, articleAcronymDB = None):
    #word_corpus = {}
    articleAcronyms = None
    for article_id, text in articleDB.items():
        if expansionTokens:
            articleAcronyms = articleAcronymDB.get(article_id)
            if not articleAcronyms:
                logger.info("article with no acronyms, article_id: "+ str(article_id))

        word_corpus[article_id] = preProcessText(text, expansionTokens, articleAcronyms)
        
        #text_preparation.getCleanedWords(
        #    text
        #    , stem_words=stem_words
        #    , removeNumbers=removeNumbers)
        if len(word_corpus) % 1000 == 0:
            logger.debug(
                "converted " + str(len(word_corpus)) + " articles to words")
    #return word_corpus


def serialGetBoWCorpus(dictionary, word_corpus_values):
    return [dictionary.doc2bow(words) for words in word_corpus_values]

def preProcessText(text, expansionTokens=False, articleAcronyms=None):
    # We assume that the input text was already preprocessed
    # this is just the minimum for LDA input
    if expansionTokens and articleAcronyms:
        text = getTextWithExpansionTokens(text, articleAcronyms)
    
    return text_preparation.tokenizePreProcessedArticle(text.lower())

# TODO remove?
def parallelGetCleanedWords(article):
    #return article[0], getCleanedWords(article[1]
    #                                   , stem_words=stem_words
    #                                  , removeNumbers=removeNumbers)
    return article[0], preProcessText(article[1])

# TODO remove?
def parallelGetWordCorpus(articleDB, process_pool):
    articles = articleDB.items()
    results = process_pool.map(
        parallelGetCleanedWords, articles, chunksize=chunkSize_getCleanedWords)

    logger.info("Back from multiprocessing, making dict now")
    word_corpus = dict(results)

    return word_corpus


def _doc2bow_alias(dictionary, words):
    """
    Alias for instance method that allows the method to be called in a 
    multiprocessing pool
    see link for details: http://stackoverflow.com/a/29873604/681311
    """
    return dictionary.doc2bow(words)


def parallelGetBoWCorpus(dictionary, word_corpus_values, process_pool):
    bound_instance = functools.partial(_doc2bow_alias, dictionary)

    result = process_pool.map(
        bound_instance, word_corpus_values, chunksize=chunkSize_doc2BoW)

    return result


# TODO remove?
def getWordCorpus(articleDB, process_pool, useSavedTill):
    if(useSavedTill >= USESAVED.word_corpus):
        logger.info("Loading word_corpus from out_file")
        word_corpus = pickle.load(open(file_lda_word_corpus, "rb"))
        return word_corpus, None
    else:
        logger.info("Getting word_corpus from articles")
        word_corpus = parallelGetWordCorpus(
            articleDB, process_pool) if process_pool != None else serialGetWordCorpus(articleDB)

        return word_corpus, None
        """
        logger.info(
            "Saving word_corpus asynchronously, in case the script ahead fails")
        out_file = open(file_lda_word_corpus, "wb")
        word_corpus_dumper = Thread(
            target=pickle.dump, args=(word_corpus, out_file), kwargs={"protocol": 2})
        word_corpus_dumper.start()
        return word_corpus, word_corpus_dumper
        """


def getDictionary(word_corpus, useSavedTill):
    if(useSavedTill >= USESAVED.dictionary):
        logger.info("loading dictionary from file")
        dictionary = Dictionary.load(file_lda_gensim_dictionary)
        return dictionary
    else:
        logger.info("Creating dictionary from corpus")
        dictionary = Dictionary(word_corpus.values())
        return dictionary


def getBoWCorpus(word_corpus, dictionary, process_pool, useSavedTill):
    # TODO convert to sqlite -> doc id, bow
    if(useSavedTill >= USESAVED.bow_corpus):
        logger.info("loading bow_corpus from out_file")
        bow_corpus = pickle.load(open(file_lda_bow_corpus, "rb"))
        return bow_corpus, None
    else:
        logger.info("Creating BoW representations from articles")
        bow_corpus = parallelGetBoWCorpus(dictionary, word_corpus.values(
        ), process_pool) if process_pool != None else serialGetBoWCorpus(dictionary, word_corpus.values())
        return bow_corpus, None


# expansionTokens to process expansion tokens 
def getLdaModel(bow_corpus, dictionary, useSavedTill, num_topics=100, numPasses=1):
    if(useSavedTill >= USESAVED.lda_model):
        logger.info("loading LDA model from file")
        return LdaModel.load(file_lda_model)
    else:
        logger.info("Training LDA model")
        if(num_topics == 'log(nub_distinct_words)+1'):
            num_topics = int(math.log(len(bow_corpus)) + 1)
        else:
            num_topics = int(num_topics)
        
        # TODO bow_corpus.values
        lda_model = LdaModel(ResetableValuesIter(bow_corpus), num_topics=num_topics, id2word=dictionary, passes=numPasses, distributed=False)
        return lda_model

def createArticleIdToLdaDict(bow_corpus, lda_model, article_lda):
    # We don't need this, we use new bow_corpus
    logger.info("Creating article_id -> lda_vector dictionary")
    index = 0
    for article_id, bow in bow_corpus.items():
        lda_vec = lda_model[bow]
        
        article_lda[article_id] = lda_vec
        index += 1
        if(index % 1000 == 0):
            logger.debug("done with %d articles", index)


def waitForDumper(dumper, name):
    if(dumper != None):
        if(dumper.is_alive()):
            logger.info(
                "Waiting for" + name + " dumper to finish saving to disk")
            dumper.join()
        else:
            logger.info(
                name + " dumper has already finished saving to disk, not waiting")


# TODO remove?
def getWord_Corpus (articleDB, fold, persistentArticles = None):
    try:
        if not persistentArticles:
            word_corpus = {}
        else:
            tmpDirName = tempfile.mkdtemp()
            word_corpus_fileName = tmpDirName + "lda_word_corpus" + "_" + str(fold)
            word_corpus = SqliteDict(word_corpus_fileName, tablename='word_corpus', flag='n', autocommit=True)
        
        serialGetWordCorpus(articleDB, word_corpus)
        
        # TODO
    finally:
        if persistentArticles:
            word_corpus.close()

    return word_corpus

class ResetableValuesIter():
    def __init__(self, sqliteDict):
        self.sqliteDict = sqliteDict
        
    def __iter__ (self):
        return iter(self.sqliteDict.values())
    
    def __len__(self):
        return self.sqliteDict.__len__()

def create_model(process_pool, datasetName, articleDB, articleAcronymDB=None, useSavedTill=USESAVED.none, num_topics=100, numPasses=1, 
                                expansionTokens = False,
                                fold = "",
                                saveAndLoad = False,
                                persistentArticles = None,
                                executionTimeObserver = None):
    """
    This takes a long time to train (~1 week), 
    run on a compute node with ~250 GB RAM and fast processor
    for wikipedia corpus of 410k documents

    Above time and storage estimates are not correct yet.
    """
    
    generatedFilesFolder = getDatasetGeneratedFilesPath(datasetName)
    dicfileName = generatedFilesFolder + "lda_dict" + "_" + str(fold) + ".pickle"
    varToName = "_".join([str(s) for s in [datasetName,fold,num_topics,numPasses]])
    
    if expansionTokens:
        varToName += "_exp"
    
    ldaArticleFilePath = generatedFilesFolder + "/articleIDToLDADict" + '_' + varToName + ".sqlite"
    
    logger.critical("create_model LDA")
    if tracemalloc.is_tracing():
        gc.collect()
        snapshot3 = tracemalloc.take_snapshot()
        logger.critical("Current")
        display_top(snapshot3, limit = 20)
    logOSStatus()
    
    if saveAndLoad:
        ldaLocation = generatedFilesFolder + file_lda_model_all + "_" + varToName + ".pickle"
            
        if os.path.isfile(ldaLocation):
            if not persistentArticles:
                with open(ldaLocation, "rb") as f:
                    (ldaModel, articleIDToLDADict) = pickle.load(f)
            else:
                with open(ldaLocation, "rb") as f:
                    ldaModel = pickle.load(f)
                
                articleIDToLDADict = SqliteDict(filename=ldaArticleFilePath, flag='r', autocommit=True)
                
            dictionary = Dictionary.load(dicfileName)
            return LDAStructure(ldaModel, dictionary, articleIDToLDADict)

            
    
   
    # word_corpus, word_corpus_dumper = getWordCorpus(
    #    articleDB, process_pool, useSavedTill)
    bowfileName = generatedFilesFolder + "lda_bow" + "_" + str(fold)
    if expansionTokens:
        bowfileName += "_exp"
    bowfileName += ".pickle" if not persistentArticles else ".sqlite"
    
        
    if os.path.isfile(dicfileName) and os.path.isfile(bowfileName):
        dictionary = Dictionary.load(dicfileName)
        #with open(dicfileName) as f:
        #    fPickle = pickle.load(f)
        #    dictionary = fPickle.dictionary
             
        if not persistentArticles:
            with open(bowfileName, "rb") as f:
                bow_corpus = pickle.load(f)

    else:
        try:
            if not persistentArticles:
                word_corpus = {}
                bow_corpus = {}
            else:
                word_corpus = SqliteDict(flag='n', autocommit=True)
                bow_corpus = SqliteDict(bowfileName, tablename='bow', flag='n', autocommit=True)
            
            serialGetWordCorpus(articleDB, word_corpus, expansionTokens, articleAcronymDB)
            
            if executionTimeObserver:
                executionTimeObserver.start()    
                
            dictionary = getDictionary(word_corpus, useSavedTill)
            for articleId, words in word_corpus.items():
                bow_corpus[articleId] = dictionary.doc2bow(words)
                
            if executionTimeObserver:
                executionTimeObserver.stop()    
                
            if not persistentArticles:
                word_corpus.clear()
                _saveAll(bow_corpus, path=bowfileName)
                
            dictionary.save(dicfileName)
            #_saveAll(dictionary, path= dictionary)
            
        finally:
            if persistentArticles:
                word_corpus.clear()
                word_corpus.close()
                bow_corpus.close()
        
    #dictionary = getDictionary(word_corpus, useSavedTill)

    #bow_corpus, bow_corpus_dumper = getBoWCorpus(
    #    word_corpus, dictionary, process_pool, useSavedTill)

    logger.critical("After BOW/Dic")
    if tracemalloc.is_tracing():
        gc.collect()
        snapshot3 = tracemalloc.take_snapshot()
        logger.critical("Current")
        display_top(snapshot3, limit = 20)
    logOSStatus()

    if(process_pool):
        logger.info("terminating process pool")
        process_pool.close()
        process_pool.terminate()
    try:
        if persistentArticles:
            bow_corpus = SqliteDict(bowfileName, tablename='bow', flag='r', autocommit=True)
            
            logger.critical("after SQLITE read bow")
            if tracemalloc.is_tracing():
                gc.collect()
                snapshot3 = tracemalloc.take_snapshot()
                logger.critical("Current")
                display_top(snapshot3, limit = 20)
            logOSStatus()  
            
        if executionTimeObserver:
            executionTimeObserver.start()  
                
        ldaModel = getLdaModel(bow_corpus, dictionary, useSavedTill, num_topics=num_topics, numPasses=numPasses)
        
        logger.critical("After LDA model")
        if tracemalloc.is_tracing():
            gc.collect()
            snapshot3 = tracemalloc.take_snapshot()
            logger.critical("Current")
            display_top(snapshot3, limit = 20)
        logOSStatus()
    
        if not persistentArticles:
            articleIDToLDADict = {}
        else:
            if saveAndLoad:
                filename = ldaArticleFilePath
            else:
                filename = None
            
            articleIDToLDADict = SqliteDict(filename=filename, flag='n', autocommit=True)
            
        createArticleIdToLdaDict(bow_corpus, ldaModel, articleIDToLDADict)
    
        logger.critical("After createArticleIdToLdaDict")
        if tracemalloc.is_tracing():
            gc.collect()
            snapshot3 = tracemalloc.take_snapshot()
            logger.critical("Current")
            display_top(snapshot3, limit = 20)
        logOSStatus()
    
        if executionTimeObserver:
            executionTimeObserver.stop()

    finally:
        if persistentArticles:
            bow_corpus.close()

    if not persistentArticles and saveAndLoad:
        _saveAll((ldaModel, articleIDToLDADict), path=ldaLocation)
    elif saveAndLoad:
        _saveAll(ldaModel, path=ldaLocation)
        
    model_all = LDAStructure(ldaModel, dictionary, articleIDToLDADict)    
    return model_all


def _saveAll(model_all, path=file_lda_model_all):
    pickle.dump(model_all, open(path, "wb"), protocol=-1)


def update_model(articledb_path):
    """returns built lda_model, lda_dictionary"""
    pass  # todo: lda has update method, use it


def logConfig():
    logger.info("Logging config of script")
    logger.info("numProcesses = %d" % numProcesses)
    logger.info("goParallel = %s" % goParallel)
    logger.info("useSavedTill = %d" % useSavedTill)
    logger.info("chunkSize_getCleanedWords = %d" %
                       chunkSize_getCleanedWords)
    logger.info("chunkSize_doc2BoW = %d" % chunkSize_doc2BoW)
    logger.info("stem_words = %s" % stem_words)
    logger.info(
        "removeNumbers = %s" % removeNumbers)


# global config for making LDA model
numProcesses = 3
goParallel = False
useSavedTill = USESAVED.none
chunkSize_getCleanedWords = 1000
chunkSize_doc2BoW = 1000
stem_words = False
removeNumbers = True

if __name__ == "__main__":
    datasetName = MSH_SOA_DATASET
    expansionTokens = True
    
    articleDB = ArticleDB.getArticleDB(datasetName)
    articleAcronymDB = ArticleAcronymDB.getArticleAcronymDB(datasetName)
    
    logger.info("LDA Model script started")
    #logConfig()
    if(goParallel):
        process_pool = ProcessPoolExecutor(numProcesses)
        create_model(process_pool, datasetName=datasetName, articleDB=articleDB, articleAcronymDB = articleAcronymDB,
                     expansionTokens = expansionTokens, useSavedTill=useSavedTill, saveAndLoad=True)
    else:
        create_model(None, datasetName=datasetName, articleDB=articleDB, articleAcronymDB = articleAcronymDB,
                     expansionTokens = expansionTokens, useSavedTill=None, saveAndLoad=False)

