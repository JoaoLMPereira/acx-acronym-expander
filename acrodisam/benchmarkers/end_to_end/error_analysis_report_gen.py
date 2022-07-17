'''
Created on Apr 21, 2022

@author: jpereira
'''
import os
import re

import pandas as pd
import sklearn.tree as tree
from sklearn.tree import DecisionTreeRegressor 
import matplotlib.pyplot as plt

from string_constants import SCIENCE_WISE_DATASET, FOLDER_LOGS, sep,\
    MSH_ORGIN_DATASET, SDU_AAAI_AD_DEDUPE_DATASET, CS_WIKIPEDIA_DATASET,\
    FOLDER_ROOT

from helper import AcronymExpansion

from Logger import logging

logger = logging.getLogger(__name__)

FOLDER_REPORTS = FOLDER_LOGS 

FOLDER_REPORTS_OUT_EXP = FOLDER_LOGS

DECISION_TREES_FOLDER = FOLDER_ROOT + "/decision_trees/"

OUT_EXP_DATASETS = [SCIENCE_WISE_DATASET,  MSH_ORGIN_DATASET, SDU_AAAI_AD_DEDUPE_DATASET, CS_WIKIPEDIA_DATASET+"_res-dup"]

def get_expanders(datasetname, folder_reports=FOLDER_REPORTS):
    expanders = []
    for file in os.listdir(folder_reports):
            m = re.match('^quality_results_'+datasetname+'([.:=_\-\\w]+).csv$', file)
            if m:
                expander = m.group(1)
                expanders.append(expander)
    return expanders


def compute_f1(tp, total_predicted, total_actual):
    if tp == 0.0 or total_predicted == 0.0 or total_actual  == 0.0:
        return 0.0
    
    p = tp / total_predicted
    r = tp / total_actual
    return (2 * p * r) / (p+r) 

def create_results_dataframe(dataset_name):
    expanders_list = get_expanders(dataset_name)
    df_train_stats = pd.read_csv(DECISION_TREES_FOLDER+"/pd_train_stats_"+dataset_name+".csv")
    
    for expander_args in expanders_list:
        print(expander_args)
        df = pd.read_csv(FOLDER_REPORTS + sep +"quality_results_"+dataset_name+expander_args+".csv")

        if not isinstance(df, pd.DataFrame):
            continue
        
        if df.empty:
            continue
        df = df[df['in_text'] == False] # Force to use out-expansions
        
        tp_df = df.groupby(['acronym','predicted_expansion'])["success"].sum().rename_axis(['acronym','expansion']).rename("tp")
        total_predicted = df.groupby(['acronym','predicted_expansion'])['predicted_expansion'].count().rename_axis(['acronym','expansion']).rename('total_predicted')
        total_actual = df.groupby(['acronym','actual_expansion'])['actual_expansion'].count().rename('total_actual').rename_axis(['acronym','actual_expansion'])
        
        merged_pd = pd.merge(df_train_stats, tp_df,  how='left', on=['acronym','expansion']) # we ignore expansions that are not in the train data
        merged_pd = pd.merge(merged_pd, total_predicted,  how='left', on=['acronym','expansion'])
        merged_pd["expansion"] = merged_pd["expansion"].map(lambda x: str(x).strip().lower())
            
        #total_actual["actual_expansion"] = total_actual["actual_expansion"].map(lambda x: x.strip().lower())

        merged_actual_pd = pd.merge(merged_pd, total_actual.reset_index(),  how='inner', on=['acronym'])
        merged_actual_pd2 = merged_actual_pd.apply(lambda x : AcronymExpansion.areExpansionsSimilar(x["actual_expansion"].strip().lower(),x["expansion"]),axis=1)
        total_actual_normalized_exp = merged_actual_pd[merged_actual_pd2].groupby(["acronym","expansion"])["total_actual"].sum().rename('total_actual').rename_axis(['acronym','expansion'])
        
        merged_pd = pd.merge(merged_pd, total_actual_normalized_exp,  how='left', on=['acronym','expansion'])
        merged_pd = merged_pd.dropna(how='all', subset=['tp','total_predicted','total_actual']) #drop if all those 3 columns have na
        merged_pd = merged_pd.fillna(0.0)#replace na per 0
        
        merged_pd["f1"] = merged_pd.apply(lambda x : compute_f1(x["tp"],x["total_predicted"], x["total_actual"]), axis=1)
        
        merged_pd.to_csv(DECISION_TREES_FOLDER+"/pd_merge_"+dataset_name+"_"+expander_args+".csv")
        
        train = merged_pd[["acronym_len", "acronym_count", "exp_freq", "expansion_count"]]
        test = merged_pd["f1"]
        

        dtr = DecisionTreeRegressor(max_depth=3, random_state=1234)
        model = dtr.fit(train, test)
        
        text_representation = tree.export_text(model,feature_names=train.columns.to_list(),show_weights=True)
        print(text_representation)
        with open(DECISION_TREES_FOLDER+"decision_tree_"+dataset_name+"_"+expander_args+".txt", 'w') as f:
            f.write(text_representation)
        
        plt.autoscale()
        fig = plt.figure(figsize=(12,12))
        _ = tree.plot_tree(dtr, 
                           feature_names=train.columns, class_names="f1", filled=True, fontsize=10)
        
        plt.autoscale()
        fig.savefig(DECISION_TREES_FOLDER+"decision_tree_"+dataset_name+"_"+expander_args+".png", dpi=100, bbox_inches = "tight")

        
        
def create_results_out_exp_dataframe(expander, representator):
    
    df_dataset_list = []
    for dataset_name in OUT_EXP_DATASETS:
        df_train_stats = pd.read_csv(DECISION_TREES_FOLDER+"pd_train_stats_"+dataset_name+".csv")
        expanders_list = get_expanders(dataset_name+"_confidences", FOLDER_REPORTS_OUT_EXP)
        #expanders_list = get_expanders(dataset_name, FOLDER_REPORTS_OUT_EXP)
        print(dataset_name)
        for expander_args in expanders_list:
            if not (expander in expander_args and representator in expander_args):# or "confidences" in expander_args:
                continue
            
            print(expander_args)
            df = pd.read_csv(FOLDER_REPORTS_OUT_EXP + sep +"quality_results_"+dataset_name+"_confidences"+expander_args+".csv")
            #df = pd.read_csv(FOLDER_REPORTS_OUT_EXP + sep +"quality_results_"+dataset_name+expander_args+".csv")
    
            if not isinstance(df, pd.DataFrame):
                continue
            
            if df.empty:
                continue
            #df = df[df['in_text'] == False] # Force to use out-expansions
            
            tp_df = df.groupby(['acronym','predicted_expansion'])["success"].sum().rename_axis(['acronym','expansion']).rename("tp")
            total_predicted = df.groupby(['acronym','predicted_expansion'])['predicted_expansion'].count().rename_axis(['acronym','expansion']).rename('total_predicted')
            total_actual = df.groupby(['acronym','actual_expansion'])['actual_expansion'].count().rename('total_actual').rename_axis(['acronym','actual_expansion'])
            
            merged_pd = pd.merge(df_train_stats, tp_df,  how='left', on=['acronym','expansion']) # we ignore expansions that are not in the train data
            merged_pd = pd.merge(merged_pd, total_predicted,  how='left', on=['acronym','expansion'])
            merged_pd["expansion"] = merged_pd["expansion"].map(lambda x: str(x).strip().lower())
                    
            merged_actual_pd = pd.merge(merged_pd, total_actual.reset_index(),  how='inner', on=['acronym'])
            merged_actual_pd2 = merged_actual_pd.apply(lambda x : AcronymExpansion.areExpansionsSimilar(x["actual_expansion"].strip().lower(),x["expansion"]),axis=1)
            total_actual_normalized_exp = merged_actual_pd[merged_actual_pd2].groupby(["acronym","expansion"])["total_actual"].sum().rename('total_actual').rename_axis(['acronym','expansion'])
            
            merged_pd = pd.merge(merged_pd, total_actual_normalized_exp,  how='left', on=['acronym','expansion'])
            merged_pd = merged_pd.dropna(how='all', subset=['tp','total_predicted','total_actual']) #drop if all those 3 columns have na
            merged_pd["dataset"] = dataset_name
            df_dataset_list.append(merged_pd)
        
    if len(df_dataset_list) != len(OUT_EXP_DATASETS):
        print("-------------------------------------------------Failed " + expander + " " + representator)
        return None
    
    merged_pd= pd.concat(df_dataset_list, ignore_index=True)

    merged_pd = merged_pd.fillna(0.0)#replace na per 0
    #TODO alternative is to remove
    
    merged_pd["f1"] = merged_pd.apply(lambda x : compute_f1(x["tp"],x["total_predicted"], x["total_actual"]), axis=1)
    
    merged_pd.to_csv(DECISION_TREES_FOLDER+"pd_merge_out_expander_"+expander+"_"+representator+".csv")
    
    train = merged_pd[["acronym_len", "acronym_count", "exp_freq", "expansion_count","dataset"]]
    train = pd.get_dummies(train)
    test = merged_pd["f1"]
    

    dtr = DecisionTreeRegressor(max_depth=3, random_state=1234)
    model = dtr.fit(train, test)
    
    text_representation = tree.export_text(model,feature_names=train.columns.to_list(),show_weights=True)
    print(text_representation)
    with open(DECISION_TREES_FOLDER+"decision_tree_out_expander_"+expander+"_"+representator+".txt", 'w') as f:
        f.write(text_representation)
    
    plt.autoscale()
    fig = plt.figure(figsize=(12,12))
    _ = tree.plot_tree(dtr,
                       feature_names=train.columns, class_names="f1", filled=True, fontsize=10)
    
    plt.autoscale()
    fig.savefig(DECISION_TREES_FOLDER+"decision_tree_out_expander_"+expander+"_"+representator+".png", dpi=100, bbox_inches = "tight")

        
def get_dfs(dataset_name):
    expanders = []
    for file in os.listdir(DECISION_TREES_FOLDER):
            m = re.match('^pd_merge_'+dataset_name+'_'+'([.:=_\-\\w]+)_0.csv$', file)
            #quality_results_'+datasetname+'([.:=_\-\\w]+).csv$', file)
            if m:
                expander = m.group(1)
                expanders.append(expander)
    return expanders
        
def create_methods_dt(dataset_name):
        df_list = []
        for expander_args in get_dfs(dataset_name):
            df_out_expander = pd.read_csv(DECISION_TREES_FOLDER+"pd_merge_"+dataset_name+"_"+expander_args+"_0.csv")
            if "svm" in expander_args:
                df_out_expander["svm"] = 1
            elif "cossim" in expander_args:
                df_out_expander["cossim"] = 1
            elif "maddog_True" in expander_args:
                #df_out_expander["maddog"] = 1
                continue
                
            if "doc2vec" in expander_args:
                df_out_expander["doc2vec"] = 1
            elif "sbert" in expander_args:
                df_out_expander["sbert"] = 1
                
            df_list.append(df_out_expander)
        
        merged_pd= pd.concat(df_list, ignore_index=True)
        merged_pd = merged_pd.fillna(0)
        train = merged_pd[["acronym_len", "acronym_count", "exp_freq", "expansion_count", 'cossim', 'sbert', 'svm',
        'doc2vec']]
        test = merged_pd["f1"]
        

        dtr = DecisionTreeRegressor(max_depth=4, random_state=1234)
        model = dtr.fit(train, test)
        
        text_representation = tree.export_text(model,feature_names=train.columns.to_list(),show_weights=True)
        print(text_representation)
        with open(DECISION_TREES_FOLDER+"decision_tree_all_out_expanders_"+dataset_name+".txt", 'w') as f:
            f.write(text_representation)
        
        #plt.autoscale()
        fig = plt.figure(figsize=(24,12))
        _ = tree.plot_tree(dtr, 
                           feature_names=train.columns, class_names="f1", filled=True, fontsize=10)
        
        plt.autoscale()
        fig.savefig(DECISION_TREES_FOLDER+"decision_tree_all_out_expanders_"+dataset_name+".png", dpi=100, bbox_inches = "tight")
        
if __name__ == '__main__':
    """
    dataset_name = "Test=UsersWikipedia:TrainOut=FullWikipedia_MadDog:TrainIn=Ab3P-BioC"
    
    create_results_dataframe(dataset_name)
    
    create_results_dataframe("Test=UsersWikipedia:TrainOut=FullWikipedia:TrainIn=Ab3P-BioC")
    """
    
    """
    create_results_out_exp_dataframe("cossim","sbert")
    create_results_out_exp_dataframe("cossim","classic_context_vector")
    create_results_out_exp_dataframe("svm","sbert")
    create_results_out_exp_dataframe("cossim","doc2vec")
    create_results_out_exp_dataframe("svm","doc2vec")
    create_results_out_exp_dataframe("cossim","tfidf")
    """
    #create_results_out_exp_dataframe("ensembler","soft")
    #create_results_out_exp_dataframe("ensembler","hard")
    #create_results_out_exp_dataframe("sci_dr","base")
    
    dataset_name = "Test=UsersWikipedia:TrainOut=FullWikipedia_MadDog:TrainIn=Ab3P-BioC"
    #dataset_name = "Test=UsersWikipedia:TrainOut=FullWikipedia:TrainIn=Ab3P-BioC"
    create_methods_dt(dataset_name)


        
      
    