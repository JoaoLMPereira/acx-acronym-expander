'''
Created on Apr 21, 2022

@author: jpereira
'''
import os
import re
import json

import pandas as pd
from string_constants import SCIENCE_WISE_DATASET, FOLDER_LOGS, sep,\
    MSH_ORGIN_DATASET, SDU_AAAI_AD_DEDUPE_DATASET
import DataCreators.ArticleDB
from helper import get_raw_article_db_path

from Logger import logging

logger = logging.getLogger(__name__)

FOLDER_REPORTS = FOLDER_LOGS 

def load_resuls_confs_to_pd(dataset_name, expander):
    try:
        df_conf = pd.read_csv(FOLDER_REPORTS + sep + "confidences_"+dataset_name+"_confidences_"+expander+".csv")
        df_results = pd.read_csv(FOLDER_REPORTS + sep +"quality_results_"+dataset_name+"_confidences_"+expander+".csv")
        #df_results['actual_expansion'] = df_results['actual_expansion'].str.lower()
        #df2 = df_conf.join(df_results, lsuffix="conf", rsuffix="results")
        df_conf_results = pd.merge(df_conf, df_results,  how='left', left_on=['fold','doc_id','acronym'], right_on = ['fold','doc id','acronym']).drop(columns=["doc id", "confidence"])
        df_conf_results["success"] = df_conf_results["success"].astype(int)
    
        if df_conf_results.empty or df_conf_results.isnull().values.any():
            print("Couldn't create df for this expander: " + expander)
            return None
    
        return df_conf_results
    except Exception as e:
        print(e)
        return None

def get_expanders(datasetname):
    expanders = []
    for file in os.listdir(FOLDER_REPORTS):
            m = re.match('^confidences_'+datasetname+'_confidences_([.:=_\-\\w]+).csv$', file)
            if m:
                expander = m.group(1)
                expanders.append(expander)
    return expanders

def create_expander_results_dataframe(dataset_name):
    expanders_list = get_expanders(dataset_name)
    df_expanders = None
    for expander in expanders_list:
        print(expander)
        df = load_resuls_confs_to_pd(dataset_name, expander)
        if not isinstance(df, pd.DataFrame):
            continue
        
        if isinstance(df_expanders, pd.DataFrame):
            df_expanders_tmp = pd.merge(df_expanders, df,  how='outer', suffixes=[None, expander], on=['fold', 'doc_id', 'acronym', 'actual_expansion']).rename(columns={'confidences_json_dict': 'confidences_json_dict_'+expander, 'predicted_expansion': "predicted_expansion_"+expander, "success": "success_"+expander}, errors="raise")
            if df_expanders_tmp.isnull().values.any():
                print("Missmatch DFs: " + expander)
                continue
            df_expanders = df_expanders_tmp
        else:
            df_expanders = df.rename(columns={'confidences_json_dict': 'confidences_json_dict_'+expander, 'predicted_expansion': "predicted_expansion_"+expander, "success": "success_"+expander}, errors="raise")
           
    articles_db = DataCreators.ArticleDB.load(get_raw_article_db_path(dataset_name))
    df_articles = pd.DataFrame(articles_db.items(), columns=['doc_id', 'text'])
    
    df_success_only_col = df_expanders[df_expanders.columns[pd.Series(df_expanders.columns).str.startswith('success_')]]
    df_expanders["min(x,y,z) = 0 and max(x,y,z) = 1"] = (df_success_only_col.aggregate(min, "columns") == 0) & (df_success_only_col.aggregate(max, "columns") == 1)
    
    df_expanders = df_expanders.astype({"doc_id": str}, errors='raise')
    df_final = pd.merge(df_expanders, df_articles, how='left') 
    return df_final
    
def normalize_conf_values(values):

    min_value = min(values)
    if min_value < 0:  # negative values e.g., SVM decision func values
        non_neg_min = abs(min_value)
        values = [v + non_neg_min for v in values]

    values_sum = sum(values)
    for v in values:
        yield v / values_sum if values_sum != 0 else 0

def get_corr(dataset_name):
    expanders_list = get_expanders(dataset_name)
    values = []
    for expander in expanders_list:
        print(expander)
        df = load_resuls_confs_to_pd(dataset_name, expander)
        df['max_confidences'] = df['confidences_json_dict'].map(lambda x: max(normalize_conf_values(json.loads(x).values())))
        corr_value = df[['max_confidences', 'success']].corr()['max_confidences']['success']
        values.append(corr_value)
        
    return pd.Series(values, index=expanders_list)

if __name__ == '__main__':
    
    datasets = [SCIENCE_WISE_DATASET,
                MSH_ORGIN_DATASET, 
                SDU_AAAI_AD_DEDUPE_DATASET,
                "CSWikipedia_res-dup"
                ]
    
    for dataset_name in datasets:
        logger.critical("Creating report for dataset: " + dataset_name)
        df = create_expander_results_dataframe(dataset_name)
        df.to_csv(FOLDER_REPORTS + "confidences_report_"+dataset_name+".csv")
    
    
    corr_dict = {}
    for dataset_name in datasets:
        logger.critical("Correlations for dataset: " + dataset_name)
        corr_values = get_corr(dataset_name)
        corr_dict[dataset_name]  = corr_values
        
    pd_corr = pd.DataFrame(corr_dict)
    pd_corr.to_csv(FOLDER_REPORTS + "corr_conf_succ.csv")
    
    