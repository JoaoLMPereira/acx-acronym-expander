
from string_constants import MSH_SOA_DATASET,\
    SCIENCE_WISE_DATASET, CS_WIKIPEDIA_DATASET, REUTERS_DATASET,\
    FULL_WIKIPEDIA_DATASET, USERS_WIKIPEDIA, SDU_AAAI_AD_DEDUPE_DATASET,\
    UAD_DATASET, MSH_ORGIN_DATASET, FOLDER_GENERATED_FILES, FOLDER_ROOT

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import spacy

import pandas as pd

from Logger import logging
from helper import get_acronym_db_path, getArticleDBPath,\
    getArticleAcronymDBPath, _processAcronymChoices, get_raw_article_db_path


from DataCreators import AcronymDB, ArticleDB, ArticleAcronymDB
import Logger

logger = logging.getLogger(__name__)

def getNumArticles(datasetName, storageType = None):
    logger.info(datasetName)
    articleDB = ArticleDB.load(path=getArticleDBPath(datasetName), storageType = storageType)
    numArticles = len(articleDB)
    logger.info("Number of articles: " + str(numArticles))
    
    
    articleAcronymDB = ArticleAcronymDB.load(path=getArticleAcronymDBPath(datasetName), storageType = storageType)
    avgNumOfExp = sum([len(value) for value in articleAcronymDB.values()]) / numArticles
    
    logger.info("AVG Number of distinct expansions per article: " + str(avgNumOfExp))


def processValues(values):
    nExpan = len(_processAcronymChoices([v[0] for v in values])[1])
        
    return nExpan


def getStatisticsAcronyms(datasetName, storageType=None, multiProcess=True):
    acronymDBPath= get_acronym_db_path(datasetName)
    acronymsDB = AcronymDB.load(path=acronymDBPath, storageType = storageType)
    numAcronyms = len(acronymsDB)
    logger.info("Number of acronyms: " + str(numAcronyms))
    
    acronymValues = list(acronymsDB.values())
    
    numExpansions = 0
    #numExpPerAmbiguousAcro = 0
    numOfAmbiguousAcro = 0
    sumExpPerAmbiguousAcro = 0
    if multiProcess:
        with ProcessPoolExecutor() as process_pool:
            with tqdm(total=numAcronyms) as pbar:
                #for i, r in tqdm(enumerate(process_pool.imap_unordered(processWikiFile, filePathsList, chunksize=1))):
                for i, nExpan in tqdm(enumerate(process_pool.map(processValues, acronymValues, chunksize=1))):
                    numExpansions += nExpan
                    if nExpan > 1:
                        numOfAmbiguousAcro += 1
                        sumExpPerAmbiguousAcro += nExpan
                    pbar.update()     
        
    else:
        for values in acronymValues:
            nExpan = processValues(values)
            #nExpan = len(_processAcronymChoices([v[0] for v in values])[1])
            numExpansions += nExpan
            if nExpan > 1:
                numOfAmbiguousAcro += 1
                sumExpPerAmbiguousAcro += nExpan
        
    #numExpPerAcro = [len(_processAcronymChoices([v[0] for v in value])[1]) for value in acronymValues]
    
    #numExpansions = sum(numExpPerAcro)
    logger.info("Number of expansions: " + str(numExpansions))
    avgExpPerAcro = numExpansions / numAcronyms
    logger.info("Average expansions per acronym: " + str(avgExpPerAcro))
    
    #numExpPerAmbiguousAcro = [l for l in numExpPerAcro if l > 1]
    #numOfAmbiguousAcro = len(numExpPerAmbiguousAcro)
    logger.info("Number of ambiguous acronyms: " + str(numOfAmbiguousAcro))
    
    avgExpPerAmbAcro = sumExpPerAmbiguousAcro / numOfAmbiguousAcro
    logger.info("Average expansions per ambiguous acronym: " + str(avgExpPerAmbAcro))

nlp = spacy.load("en_core_web_sm")
def num_of_sentences(text):
    sentences = nlp(text).sents
    return len(list(sentences)), len(text)

def get_num_sentences(datasetName, storageType):
    multiprocess = True
    articleDB = ArticleDB.load(path=get_raw_article_db_path(datasetName), storageType = storageType)
    numArticles = len(articleDB)
    logger.info("Number of articles: " + str(numArticles))
    
    sentence_number = 0
    char_num_total = 0
    if multiprocess: 
        with ProcessPoolExecutor() as process_pool:
            with tqdm(total=numArticles) as pbar:
                    #for i, r in tqdm(enumerate(process_pool.imap_unordered(processWikiFile, filePathsList, chunksize=1))):
                for i, (sent_numb, char_num) in tqdm(enumerate(process_pool.map(num_of_sentences, list(articleDB.values()), chunksize=1))):
                    
                    char_num_total +=char_num
                    sentence_number += sent_numb
                    pbar.update()
    else:
        with tqdm(total=numArticles) as pbar:
            for text in articleDB.values():
                char_num += len(text)
                sentences = nlp(text).sents
                sentence_number += len(list(sentences))
                pbar.update()
        
    logger.info("Number of sentences: " + str(sentence_number))
    logger.info("Number of chars: " + str(char_num_total) + "avg per article: " + str(char_num_total / numArticles))

    return sentence_number

def getStatistics(datasetName, storageType = None):
    getNumArticles(datasetName, storageType)
    getStatisticsAcronyms(datasetName, storageType)

def _transf_exp_article_tuple(exp_article_tuple):
    try:
        if len(exp_article_tuple) != 2:
            print(exp_article_tuple)
    except Exception:
        print(exp_article_tuple)
        
    return exp_article_tuple

def get_train_data_stats(dataset_name, storageType = None):
    path_acronym_db = FOLDER_GENERATED_FILES + dataset_name + "/db_TrainData_TrainAcronyms.sqlite"
    storageType="SQLite"
    acronym_db = AcronymDB.load(path_acronym_db, storageType)
    df = pd.DataFrame(acronym_db.items(), columns=["acronym", "expansions_list"])#[0:100]
    df["acronym_len"] = df.acronym.map(len)
    df_expansions = df.explode("expansions_list").dropna()
    df_expansions["expansion"] = df_expansions["expansions_list"].map(lambda x: x[0])
    df_acronym_count= df_expansions.groupby(["acronym"])["acronym"].count()
    df_exp_count = df_expansions.groupby(["acronym", "expansion"])["expansion"].count()

    merged_pd = pd.merge(df_exp_count, df_acronym_count,  how='inner', left_index=True, right_index=True)
    merged_pd["exp_freq"] = merged_pd["expansion"] / merged_pd["acronym"]
    merged_pd = merged_pd.rename(columns={'expansion':'expansion_count', 'acronym':'acronym_count'})
    
    merged_pd = pd.merge(df[["acronym", "acronym_len"]], merged_pd.reset_index(), on=["acronym"])
    
    #merged_pd.to_csv("/home/jpereira/MEGA/Acronym/decision_trees/pd_train_stats_"+dataset_name+".csv")
    merged_pd.to_csv(FOLDER_ROOT+"/decision_trees/pd_train_stats_"+dataset_name+".csv")
    
if __name__ == "__main__":

    #getStatistics(MSH_SOA_DATASET)
    
    #getStatistics(SCIENCE_WISE_DATASET)
    
    #getStatistics(CS_WIKIPEDIA_DATASET)
    
    #getStatistics(REUTERS_DATASET, "SQLite")
    #getStatistics(FULL_WIKIPEDIA_DATASET, "SQLite")
    #getStatistics(SDU_AAAI_AD_DEDUPE_DATASET)#, "SQLite")
    #get_num_sentences(UAD_DATASET, None)#, "SQLite")

    #getStatistics(USERS_WIKIPEDIA)
    
    #get_train_data_stats("Test=UsersWikipedia:TrainOut=FullWikipedia_MadDog:TrainIn=Ab3P-BioC")
    #get_train_data_stats("Test=UsersWikipedia:TrainOut=FullWikipedia:TrainIn=Ab3P-BioC")
    
    """
    get_train_data_stats(MSH_ORGIN_DATASET)
    
    get_train_data_stats(SCIENCE_WISE_DATASET)
    get_train_data_stats(SDU_AAAI_AD_DEDUPE_DATASET)
    """
    get_train_data_stats(CS_WIKIPEDIA_DATASET+"_res-dup")
    
    
