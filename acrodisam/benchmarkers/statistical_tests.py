'''
Created on Aug 7, 2020

@author: jpereira
'''
import os
import csv
import re
import random
import math
import matplotlib.pyplot as plt

import numpy as np
from string_constants import folder_logs

from Logger import logging

logger = logging.getLogger(__name__)

def save_to_file(datasetname, technique1, technique2, score_diff, p_value):
    file_path = folder_logs + "statistical_tests.csv"
    row_to_write={"Dataset":datasetname,
                  "technique1": technique1,
                  "technique2": technique2,
                  "measure1": m1,
                  "measure2": m2,
                  "score_diference": score_diff,
                  "p-value": p_value}
    
    with open(file_path, 'a') as file:
        w = csv.DictWriter(file, row_to_write.keys())

        if file.tell() == 0:
            w.writeheader()

        w.writerow(row_to_write)

def swamp_scores(scores_1, scores_2):
    scores_1 = scores_1.copy()
    scores_2 = scores_2.copy()
    
    for i in range(len(scores_1)):
        if random.random() >= 0.5:
            #swamp values
            aux = scores_1[i]
            scores_1[i] = scores_2[i]
            scores_2[i] = aux
    
    return scores_1, scores_2

def non_parametric_test(scores_1, scores_2, 
                              func,
                               method='approximate',
                               num_rounds=1000,
                               seed=0):
    
    random.seed(seed)
    score_diff=func(scores_1, scores_2)
    successes = 0
    for i in range(num_rounds):
        swamped_s1, swamped_s2 = swamp_scores(scores_1, scores_2)
        swamped_diff = func(swamped_s1, swamped_s2)
        
        if swamped_diff >= score_diff:
            successes +=1
    
    p_value = successes / num_rounds
    return p_value

def process_out_expansion_results_file(datasetname, out_expander):
    filename = folder_logs + "quality_results_"+ datasetname + '_' + out_expander + ".csv" 
    
    fold = None
    docIds = []
    total_expansions = 0
    scores_per_article_list = []
    
    total_sucesses = 0
    
    with open(filename, 'r' ) as f:
        reader = csv.DictReader(f)
        
        current_doc_id = None
        current_doc_sucesses = 0
        for row in reader:

            if not fold:
                fold = row.get('fold', None)
                if not fold:
                    fold = row['\ufefffold']
            elif fold != row.get('fold', None) and fold != row.get('ufefffold', None):
                foldnew = row.get('fold', None)
                if not foldnew:
                    foldnew = row['\ufefffold']
                logger.warning("Different folds found in same results file: %s and %s",fold, foldnew)
                
            if current_doc_id != row['doc id']:
                if current_doc_id:
                    docIds.append(current_doc_id)
                    scores_per_article_list.append(current_doc_sucesses)
                
                current_doc_sucesses = 0
                current_doc_id = row['doc id']
                
            if row['success'] == "True":
                current_doc_sucesses +=1
                total_sucesses+=1
            elif row['success'] != "False":
                logger.critical("Unkown success value: %s", row['success'])

            total_expansions += 1
        
        docIds.append(current_doc_id)
        scores_per_article_list.append(current_doc_sucesses)
        
    return fold, docIds, total_expansions, np.array(scores_per_article_list)
            
def compute_accuracy(scores_per_article, total_expansions):
    return np.sum(scores_per_article) / total_expansions

def compute_paired_test(datasetname, out_expander1, out_expander2):
    try:
        fold1, docIds1, total_expansions1, scores_per_article1 = process_out_expansion_results_file(datasetname, out_expander1)
        
        fold2, docIds2, total_expansions2, scores_per_article2 = process_out_expansion_results_file(datasetname, out_expander2)
    except Exception:
        logger.exception("A test failed for " + out_expander1 + " vs " + out_expander2)
        return None, None
        
    if (fold1, docIds1, total_expansions1) != (fold2, docIds2, total_expansions2):
        logger.error("Incompatible results files: %s %s", out_expander1, out_expander2)
        return None, None

    total_expansions = total_expansions1

    func = lambda x, y: compute_accuracy(x, total_expansions) - compute_accuracy(y, total_expansions)
    m1 = compute_accuracy(scores_per_article1, total_expansions)
    m2 = compute_accuracy(scores_per_article2, total_expansions)
    logger.critical("%s vs %s", out_expander1, out_expander2)
    logger.critical("Accuracy1: %f", m1)
    logger.critical("Accuracy2: %f", m2)
     
    score_diff=func(scores_per_article1, scores_per_article2)
    logger.critical("Accuracy difference: %f", score_diff)
    
    if score_diff < 0:
        aux = scores_per_article1
        scores_per_article1 = scores_per_article2
        scores_per_article2 = aux
    
    p_value = non_parametric_test(scores_per_article1, scores_per_article2,
                               func=func,
                               method='approximate',
                               num_rounds=1000,
                               seed=0)
    logger.critical("P-Value: %f", p_value)
    
    save_to_file(datasetname, out_expander1, out_expander2, score_diff, p_value, m1=m1, m2=m2 )
    
    return score_diff, p_value


def compute_tests_for_technique(datasetname, technique, only_exec = False, end_to_end = False):
    if not only_exec:
        logger.critical("Quality tests for " + datasetname)
        for file in os.listdir(folder_logs):
            m = re.match('^quality_results_'+datasetname+'[_:]([.:=_\-\\w]+).csv$', file)
            if m:
                technique_2 = m.group(1)
                if '_confidences' in technique_2:
                    continue
                if not end_to_end:
                    compute_paired_test(datasetname, technique, technique_2)
                else:
                    compute_paired_test_end_to_end(datasetname + ":", technique, technique_2, end_to_end)
        for file in os.listdir(folder_logs):
            m = re.match('^quality_results_'+datasetname+'_confidences_([.:=_\-\\w]+).csv$', file)
            if m:
                technique_2 = m.group(1)
                if not end_to_end:
                    compute_paired_test(datasetname+'_confidences', technique.replace('confidences_', ''), technique_2)
                else:
                    compute_paired_test_end_to_end(datasetname+'_confidences', technique.replace('confidences_', ''), technique_2, end_to_end)
    else:               
        for file in os.listdir(folder_logs):
            m_exec = re.match('^exec_time_results_'+datasetname+'[_:]([.:=_\-\\w]+).csv$', file)
            if m_exec:
                technique_2 = m_exec.group(1)
                if not end_to_end:
                    datasetname_suffix = "_"
                else:
                    datasetname_suffix = ":"
                compute_paired_test_exec_times(datasetname + datasetname_suffix, technique, technique_2)
    

def process_end_to_end_expansion_results_file(datasetname, out_expander):
    #filename = folder_logs + "results_extraction_"+ datasetname+ "_" + out_expander + ".csv" 
    filename = folder_logs + "quality_results_"+ datasetname + out_expander + ".csv" 
    
    #fold = None
    docIds = []
    total_expansions = 0
    scores_per_article_list = []
    
    
    with open(filename, 'r' ) as f:
        reader = csv.DictReader(f)
        
        current_doc_id = None
        
        extracted = 0
        correct = 0
        gold = 0
        for row in reader:

            #if not fold:
            #    fold = row['fold']
            #elif fold != row['fold']:
            #    logger.warning("Different folds found in same results file: %s and %s",fold, row['fold'])
                
            if current_doc_id != row['doc id']:
                if current_doc_id:
                    docIds.append(current_doc_id)
                    scores_per_article_list.append((correct, extracted, gold))
                
                extracted = 0
                correct = 0
                gold = 0
                current_doc_id = row['doc id']

            if row['actual_expansion'] != "":
                gold += 1
            if row['predicted_expansion'] != "":
                extracted += 1
            
            if row['success'] == "True":
                correct += 1
        
        docIds.append(current_doc_id)
        scores_per_article_list.append((correct, extracted, gold))     
        
        scores_per_article_list = [x for _,x in sorted(zip(docIds,scores_per_article_list))]
    return sorted(docIds), total_expansions, np.array(scores_per_article_list)

def compute_precision(scores_per_article):
    correct, extracted, gold = np.sum(scores_per_article, axis=0)
    return correct / extracted

def compute_recall(scores_per_article):
    correct, extracted, gold = np.sum(scores_per_article, axis=0)
    return correct / gold

def compute_f1(scores_per_article):
    correct, extracted, gold = np.sum(scores_per_article, axis=0)
    p = correct / extracted
    r = correct / gold
    return (2 * p * r) / (p+r) 

def compute_end_test_aux(datasetname, out_expander1, out_expander2, 
                         scores_per_article1, scores_per_article2, metric):
    
    if metric == "precision":
        func = lambda x, y: compute_precision(x) - compute_precision(y)
        m1 = compute_precision(scores_per_article1)
        m2 = compute_precision(scores_per_article2)
        logger.critical("Precision: %f", m1)
        logger.critical("Precision: %f", m2)
        
    elif metric == "recall":
        func = lambda x, y: compute_recall(x) - compute_recall(y)
        m1 = compute_recall(scores_per_article1)
        m2 = compute_recall(scores_per_article2)
        logger.critical("Recall: %f", m1)
        logger.critical("Recall: %f", m2)
    else:
        func = lambda x, y: compute_f1(x) - compute_f1(y)

        m1 = compute_f1(scores_per_article1)
        m2 = compute_f1(scores_per_article2)
        logger.critical("F1: %f", m1)
        logger.critical("F1: %f", m2)
     
    score_diff=func(scores_per_article1, scores_per_article2)
    logger.critical(metric + " difference: %f", score_diff)
    
    if score_diff < 0:
        aux = scores_per_article1
        scores_per_article1 = scores_per_article2
        scores_per_article2 = aux
    try:
        p_value = non_parametric_test(scores_per_article1, scores_per_article2,
                                   func=func,
                                   method='approximate',
                                   num_rounds=1000,
                                   seed=0)
        logger.critical("P-Value: %f", p_value)
        save_to_file(datasetname, out_expander1, out_expander2, score_diff, p_value, m1=m1, m2=m2, metric=metric)
    except Exception:
        logger.exception("Failed to compute P-value for results files: %s %s", out_expander1, out_expander2)
        return None, None

def compute_paired_test_end_to_end(datasetname, out_expander1, out_expander2, metric):
    docIds1, total_expansions1, scores_per_article1 = process_end_to_end_expansion_results_file(datasetname, out_expander1)
    
    docIds2, total_expansions2, scores_per_article2 = process_end_to_end_expansion_results_file(datasetname, out_expander2)
    
    if set(docIds1) != set(docIds2):
        logger.error("Incompatible results files: %s %s", out_expander1, out_expander2)
        #return None, None

    compute_end_test_aux(datasetname, out_expander1, out_expander2, 
                         scores_per_article1, scores_per_article2, metric)



def process_execution_times_results_file(datasetname, out_expander):
    filename = folder_logs + "exec_time_results_"+ datasetname+ out_expander + ".csv" 
    
    fold = None
    docIds = []
    exec_times_per_article_list = []
        
    with open(filename, 'r' ) as f:
        reader = csv.DictReader(f)
        try:
            for row in reader:
                docIds.append(row['doc id'])
                exec_times_per_article_list.append(float(row['Execution Times']))
                
                if not fold:
                    fold = row.get('fold', None)
                    if not fold:
                        fold = row['\ufefffold']
                elif fold != row.get('fold', None) and fold != row.get('ufefffold', None):
                    foldnew = row.get('fold', None)
                    if not foldnew:
                        foldnew = row['\ufefffold']
                    logger.warning("Different folds found in same results file: %s and %s",fold, foldnew)
                
        except Exception:
            logger.exception("Error reading file: " + filename)
        exec_times_per_article_list = [x for _,x in sorted(zip(docIds,exec_times_per_article_list))]
    return sorted(docIds), len(docIds), np.array(exec_times_per_article_list)

def compute_exec_time_avg(exec_times_per_article_list):
    avg = np.mean(exec_times_per_article_list)
    return avg

def compute_paired_test_exec_times(datasetname, out_expander1, out_expander2):
    docIds1, total_expansions1, scores_per_article1 = process_execution_times_results_file(datasetname, out_expander1)
    
    docIds2, total_expansions2, scores_per_article2 = process_execution_times_results_file(datasetname, out_expander2)
    
    if (docIds1, total_expansions1) != (docIds2, total_expansions2):
        logger.error("Incompatible results files: %s %s", out_expander1, out_expander2)
        return None, None

    func = lambda x, y: compute_exec_time_avg(x) - compute_exec_time_avg(y)
    m1 = compute_exec_time_avg(scores_per_article1)
    m2 = compute_exec_time_avg(scores_per_article2)
    logger.critical("Execution Times: %f", m1)
    logger.critical("Execution Times: %f", m2)
     
    score_diff=func(scores_per_article1, scores_per_article2)
    logger.critical("Execution Times difference: %f", score_diff)
    
    if score_diff < 0:
        aux = scores_per_article1
        scores_per_article1 = scores_per_article2
        scores_per_article2 = aux
    
    p_value = non_parametric_test(scores_per_article1, scores_per_article2,
                               func=func,
                               method='approximate',
                               num_rounds=1000,
                               seed=0)
    logger.critical("P-Value: %f", p_value)
    if p_value > 0.005:
        logger.critical("----------------------------------------------------not significant for+ " + out_expander2)
    
    save_to_file("exec_time_" + datasetname, out_expander1, out_expander2, score_diff, p_value, m1=m1, m2=m2)
    
    return score_diff, p_value


"""
    Confidence intervals code
"""
def bootstrap(x):
        samp_x = []
        for i in range(len(x)):
                samp_x.append(random.choice(x))
        return samp_x
    
def compute_confidence_intervals(datasetname, out_expander):
    conf_interval = 0.9
    num_resamples = 10000   # number of times we will resample from our original samples

    fold, docIds, total_expansions, scores_per_article = process_out_expansion_results_file(datasetname, out_expander)
    accuracy_list = []
    n_articles = len(scores_per_article)
    accuracy = compute_accuracy(scores_per_article, total_expansions)
    print("Accuracy: " + str(accuracy))
    for i in range(num_resamples):
        #random take n_articles with repetition from scores_per_article1
        new_scores = bootstrap(scores_per_article)
        acc = compute_accuracy(new_scores, total_expansions)
        accuracy_list.append(acc)
        
    accuracy_list.sort()

    # standard confidence interval computations
    tails = (1 - conf_interval) / 2
    
    # in case our lower and upper bounds are not integers,
    # we decrease the range (the values we include in our interval),
    # so that we can keep the same level of confidence
    lower_bound = int(math.ceil(num_resamples * tails))
    upper_bound = int(math.floor(num_resamples * (1 - tails)))
    
    print("Lower bound: " + str(accuracy_list[lower_bound]))
    print("Upper bound: " + str(accuracy_list[upper_bound]))
    return  accuracy, (accuracy_list[upper_bound] - accuracy, accuracy - accuracy_list[lower_bound])
        
def plot_histogram_confidences_interval(expanders_names, accuracies, intervals):
    x_pos = np.arange(len(expanders_names))
    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, accuracies, yerr= np.asarray(intervals).T, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Out-Expansion Accuracy (%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(expanders_names)
    ax.set_title('Out-Expansion Accuracy Coffidence Intervals for 90% cofidence.')
    ax.yaxis.grid(True)
    
    # Save the figure and show
    plt.tight_layout()
    plt.savefig('bar_plot_with_error_bars.png')
    plt.show()
    
if __name__ == '__main__':
    #compute_paired_test("ScienceWISE", "SVM_Doc2Vec_6_l2_0.1", "DualEncoder_1_6_2")
    
    """
    compute_tests_for_technique("ScienceWISE", "SVM_Concat1_6_l2_0.1")
    compute_tests_for_technique("MSHCorpusSOA", "SVM_Concat1_5_l2_0.01")
    compute_tests_for_technique("CSWikipedia_res-dup","SVM_Concat1_5_l2_0.01")
    """
    
    """
    compute_tests_for_technique("ScienceWISE", "ContextVector_1")
    compute_tests_for_technique("MSHCorpusSOA", "ContextVector_1")
    compute_tests_for_technique("CSWikipedia_res-dup","ContextVector_1")
    """
    
    #compute_paired_test("ScienceWISE","SVM_Concat1_6_l2_0.1", "ContextVector_1")
    #compute_paired_test_f1("FullWikipedia_UsersWikipedia","Orig_SH_SVM_Concat1_5_l2_0.01","with_links_Orig_SH_SVM_Concat1_5_l2_0.01")
    #compute_paired_test_exec_times("FullWikipedia_UsersWikipedia","Orig_SH_SVM_Concat1_5_l2_0.01","with_links_Orig_SH_SVM_Concat1_5_l2_0.01")
    """
    compute_paired_test_f1("FullWikipedia_UsersWikipedia","Orig_SH_ContextVector_1","Orig_SH_SVM_Concat1_5_l2_0.01")
    compute_paired_test_f1("FullWikipedia_UsersWikipedia","Orig_SH_ContextVector_1","Orig_SH_SVM_Doc2Vec_5_l2_0.01")
    compute_paired_test_f1("FullWikipedia_UsersWikipedia","Orig_SH_SVM_Concat1_5_l2_0.01","Orig_SH_SVM_Doc2Vec_5_l2_0.01")
    """
    
    """
    compute_tests_for_technique("ScienceWISE", "SVM_Concat2_6_l2_0.1")
    compute_tests_for_technique("MSHCorpusSOA", "SVM_Concat2_6_l2_0.1")
    compute_tests_for_technique("CSWikipedia_res-dup","SVM_Concat2_5_l2_0.01")
    """
    """
    compute_tests_for_technique("ScienceWISE", "SVM_Doc2Vec_6_l2_0.1")
    compute_tests_for_technique("MSHCorpusSOA", "SVM_Doc2Vec_6_l2_0.1")
    compute_tests_for_technique("CSWikipedia_res-dup","SVM_Doc2Vec_5_l2_0.01")
    """
    
    #accuracy, interval = compute_confidence_intervals("ScienceWISE", "cossim_document_context_vector")
    #accuracy2, interval2 = compute_confidence_intervals("ScienceWISE", "uad_vote_None")
    
    #plot_histogram_confidences_interval(["cossim document context vector", "UAD"], [accuracy, accuracy2], [interval, interval2])
    
    #vldb
    #compute_tests_for_technique("ScienceWISE", "cossim_classic_context_vector")
    #compute_tests_for_technique("MSHCorpus", "svm_l2_0.1_0_concat_classic_context_vector_1_doc2vec_25_CBOW_200_2")
    #compute_tests_for_technique("CSWikipedia_res-dup","sci_dr_base_32")
    #compute_tests_for_technique("SDU-AAAI-AD-dedupe","sci_dr_base_32")
    
    #compute_tests_for_technique("ScienceWISE", "cossim_classic_context_vector")
    """
    compute_tests_for_technique("MSHCorpus", "cossim_classic_context_vector", True)
    compute_tests_for_technique("CSWikipedia_res-dup","cossim_classic_context_vector", True)
    compute_tests_for_technique("SDU-AAAI-AD-dedupe","cossim_classic_context_vector", True)
    """
    """
    compute_tests_for_technique("Test=UsersWikipedia:TrainOut=FullWikipedia", "_MadDog:TrainIn=Ab3P-BioC_mad_dog_None_svm_l2_0.01_0_doc2vec_100_CBOW_100_5_1", end_to_end = True)
    #compute_tests_for_technique("Test=UsersWikipedia:TrainOut=FullWikipedia", ":TrainIn=Ab3P-BioC_schwartz_hearst_None_svm_l2_0.01_0_doc2vec_100_CBOW_100_5_1", end_to_end = True)
    #compute_tests_for_technique("Test=UsersWikipedia:TrainOut=FullWikipedia", ":TrainIn=Ab3P-BioC_schwartz_hearst_None_svm_l2_0.01_0_doc2vec_100_CBOW_100_5_0", only_exec = True)
    """
    
    #vldb major
    """
    compute_tests_for_technique("ScienceWISE", "confidences_cossim_sbert_all-mpnet-base-v2_simple")
    compute_tests_for_technique("MSHCorpus", "svm_l2_0.1_0_concat_classic_context_vector_1_doc2vec_25_CBOW_200_2")
    compute_tests_for_technique("CSWikipedia_res-dup","confidences_sci_dr_base_32_cpu")
    compute_tests_for_technique("SDU-AAAI-AD-dedupe","confidences_sci_dr_base_32") #both?
    compute_tests_for_technique("SDU-AAAI-AD-dedupe","confidences_sci_dr_both_32")
    
    compute_tests_for_technique("ScienceWISE", "confidences_ensembler_hard")
    compute_tests_for_technique("MSHCorpus", "confidences_ensembler_hard")
    compute_tests_for_technique("CSWikipedia_res-dup","confidences_ensembler_soft")
    compute_tests_for_technique("SDU-AAAI-AD-dedupe","confidences_ensembler_hard") #both?
    """

    #compute_tests_for_technique("Test=UsersWikipedia", "TrainOut=FullWikipedia_MadDog:TrainIn=Ab3P-BioC_maddog_None_svm_l2_1_0_sbert_all-mpnet-base-v2_simple_0", end_to_end = "precision")
    #compute_tests_for_technique("Test=UsersWikipedia", "TrainOut=FullWikipedia:TrainIn=Ab3P-BioC_schwartz_hearst_None_svm_l2_1_0_sbert_all-mpnet-base-v2_simple_0", end_to_end = "recall")
    #compute_tests_for_technique("Test=UsersWikipedia", "TrainOut=FullWikipedia_MadDog:TrainIn=Ab3P-BioC_maddog_None_svm_l2_1_0_sbert_all-mpnet-base-v2_simple_0", end_to_end = "f1")

    compute_tests_for_technique("Test=UsersWikipedia", "TrainOut=FullWikipedia:TrainIn=Ab3P-BioC_schwartz_hearst_None_cossim_sbert_all-mpnet-base-v2_simple_0", end_to_end=True, only_exec=True )

    #process_out_expansion_results_file("ScienceWISE_confidences", "ensembler_soft")
    