import math
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from DataCreators import ArticleDB
from string_constants import FILE_TFIDF_VECTORS, MSH_SOA_DATASET
from helper import getArticleDBPath, getDatasetGeneratedFilesPath
import joblib

logger = logging.getLogger(__name__)


stem_words = False
removeNumbers = True

def getArticleDB(datasetName):
    articleDBPath = getArticleDBPath(datasetName)    
    articleDB = ArticleDB.load(path=articleDBPath)
    return articleDB

def getTextCorpus(articleDB):
    text_corpus = {}
    for article_id, text in articleDB.items():
        text_corpus[article_id] = text
        if len(text_corpus) % 1000 == 0:
            logger.debug(
                "converted " + str(len(text_corpus)) + " articles to text")
    return text_corpus

def getWordsCount(text_corpus):
#    unique_words = set([[word for word in text.split()] for text in text_corpus.values()])
    unique_words = set()
    for text in text_corpus:
        [unique_words.add(word) for word in text.split()]
    return len(unique_words)

# num,max_features-max_df-min_df,ngram_range(min-max)
def getTFIDFModel(text_corpus, 
                  ngram_range=(1, 1), 
                  max_df=1.0, 
                  min_df=1,
                 max_features=None):
    if(max_features == 'log(nub_distinct_words)+1'):
        max_features = int(math.log(getWordsCount(text_corpus)))
    
    vectorizer = TfidfVectorizer(ngram_range = ngram_range,
                                 max_df=max_df,
                                 min_df=min_df, 
                                 max_features=max_features, # also the vocabulary, None=0, 1000000, 10000, 1000
                                 stop_words='english', 
                                 use_idf=True, 
                                 binary=False, 
                                 decode_error='ignore')
    vectorizer.fit(text_corpus)
    return vectorizer

def getTFIDFModelForArticles(articleDB, 
                  ngram_range=(1, 1), 
                  max_df=1.0, 
                  min_df=1,
                 max_features=None,
                 datasetName = None,
                 fold = "",
                 saveAndLoad = False,
                 executionTimeObserver = None):
    
    #Check if exists
    if saveAndLoad:
        generatedFilesFolder = getDatasetGeneratedFilesPath(datasetName)
        varToName = "_".join([str(s) for s in [datasetName,fold,ngram_range,max_df,min_df,max_features]])
        vectorizerLocation = generatedFilesFolder + "TFIDFModel_" + varToName + ".pickle"
            
        if os.path.isfile(vectorizerLocation):
            with open(vectorizerLocation, "rb") as f:
                return pickle.load(f)
       
    #text_corpus = getTextCorpus(articleDB)
    text_corpus = articleDB.values()
    if executionTimeObserver:
        executionTimeObserver.start()
    vectorizer = getTFIDFModel(text_corpus, 
                                 ngram_range = ngram_range,
                                 max_df=max_df,
                                 min_df=min_df, 
                                 max_features=max_features)
    if executionTimeObserver:
        executionTimeObserver.stop()
    
    if saveAndLoad:
        with open(vectorizerLocation, "wb") as f:
            pickle.dump(vectorizer, f, protocol=1)
    
    return vectorizer
    
def main():
    datasetName = MSH_SOA_DATASET
        
    articleDB = getArticleDB(datasetName)
    vectorizer = getTFIDFModelForArticles(articleDB, max_features=10000)
    
    generatedFilesFolder = getDatasetGeneratedFilesPath(datasetName)
    
    joblib.dump(vectorizer, generatedFilesFolder + FILE_TFIDF_VECTORS) 

if __name__ == "__main__":
    main()
