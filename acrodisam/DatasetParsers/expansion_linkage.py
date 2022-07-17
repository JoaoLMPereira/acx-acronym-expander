'''
Created on Sep 20, 2020

@author: jpereira
'''

import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from sqlitedict import SqliteDict
import pandas as pd
import networkx as nx

from helper import get_acronym_db_path, getArticleDBPath,  getArticleAcronymDBPath,\
    extend_dict_of_lists, AcronymExpansion, get_raw_article_db_path,\
    get_preprocessed_article_db_path
from string_constants import FULL_WIKIPEDIA_DATASET, CS_WIKIPEDIA_DATASET,\
    MSH_SOA_DATASET, SCIENCE_WISE_DATASET
from Logger import logging
from DataCreators.ArticleAcronymDB import create_article_acronym_db_from_acronym_db
from text_preparation import get_expansion_without_spaces

logger = logging.getLogger(__name__)

def aggregate_criteria_expansion(expansions):
    value_counts = expansions.value_counts()
    max_value_counts = value_counts[value_counts == value_counts.max()]
    if len(max_value_counts) < 2:
        return max_value_counts.index[0]
    
    index_largest_value = max_value_counts.index.map(len).argmax()
    
    return max_value_counts.index[index_largest_value]

def _get_approximate_duplicates(data_frame):
    expansions = data_frame['expansion']
    app_dup = []
    for i in range(0,len(expansions) -1):
        for j in range(i+1,len(expansions)):
            if AcronymExpansion.areExpansionsSimilar(expansions[i], expansions[j]):
                app_dup.append((i, j))
    
    return app_dup

def _resolve_exp_for_acronym(item):
    acronym = item[0]
    exp_article_pairs = item[1]
    exp_changes_articles = {}
    #exp_article_pairs = acronymDB[acronym]
    if len(exp_article_pairs) > 1:
        df = pd.DataFrame(exp_article_pairs, columns=["expansion","article"])
    
        matches_indx = _get_approximate_duplicates(df)
    
    
        df_article_duplicates = df[df.duplicated("article", keep=False)]
        if not df_article_duplicates.empty :
            df_idx_pairs = df_article_duplicates.groupby('article').apply(lambda x: tuple(x.index)).tolist()
            for dup_tup in df_idx_pairs:
                if not dup_tup in matches_indx:
                    logger.warning("Found two non-approximate expansions: %s and %s for acronym %s in article %s. We will consider them approximate duplicates.",
                                   df["expansion"][dup_tup[0]],df["expansion"][dup_tup[1]],acronym, df["article"][dup_tup[0]])
                    matches_indx.append(dup_tup)
    
        # Cluster, applies transitive closure to match pairs
        G = nx.Graph()
        G.add_edges_from(matches_indx)
        cluster_tuples = []
        cluster_id = 0
        for connected_component in nx.connected_components(G):
            tuples = [(index, cluster_id) for index in connected_component]
            cluster_tuples += tuples
            cluster_id += 1
        
                    
        df_clusters = pd.DataFrame(cluster_tuples, columns=["indx","cluster_id"])
        df_result = pd.merge(df_clusters, df, left_on='indx', right_index=True)
        
        # Merge clusters
        df_consolidated = df_result.groupby("cluster_id").agg({"expansion": aggregate_criteria_expansion, "article": list})
        
        # Replace expansions by its normalized expansion from merging
        consolidated_expansions_articles = df_consolidated.explode("article").drop_duplicates()
        
        merged_consolidated_all = pd.merge(df, 
                    consolidated_expansions_articles, 
                    left_on = "article", 
                    right_on = "article",
                    how='left', 
                    suffixes=("_old","_new"))
        
        nan_idx = pd.isna(merged_consolidated_all['expansion_new'])
        new_exp_indx = ~nan_idx
        
        # Save into db
        new_exp_article_pairs = pd.concat([df[nan_idx], consolidated_expansions_articles]).drop_duplicates().to_records(False).tolist()
        
        # keep track changes for articles text
        changes_to_apply = merged_consolidated_all[new_exp_indx].drop_duplicates()
                                                
        #add to dict
        for row in changes_to_apply.itertuples():
            if row.expansion_old != row.expansion_new:
                article_changes = exp_changes_articles.setdefault(row.article,set())
                article_changes.add((row.expansion_old, row.expansion_new))
        
        return acronym, new_exp_article_pairs, exp_changes_articles
    #acronym_db_new[acronym] = exp_article_pairs
    return acronym, exp_article_pairs, None

def _generator_singular_acronyms(acronym_db):
    already_processed_acros = set()
    for acronym, expansions in acronym_db.items():
        if acronym in already_processed_acros:
            continue
        
        if acronym[-1] == "s":
            singular_acro = acronym[:-1]
            other_acro_version = singular_acro
        else:
            singular_acro = acronym
            other_acro_version = acronym + "s"
        
        other_version_expansions = acronym_db.get(other_acro_version, None)
        if other_version_expansions:
            already_processed_acros.add(other_acro_version)
            yield singular_acro, expansions + other_version_expansions
        else:
            yield singular_acro, expansions

def _resolve_exp_acronym_db(acronymDB, acronym_db_new):

    exp_changes_articles = {}
    
    tasksNum = len(acronymDB)
    with ProcessPoolExecutor() as process_pool:
        with tqdm(total=tasksNum) as pbar:
            for _, r in tqdm(enumerate(process_pool.map(_resolve_exp_for_acronym, _generator_singular_acronyms(acronymDB), chunksize=1))):
                acronym = r[0]
                exp_article_pairs = r[1]
                exp_changes_articles_new = r[2]
                acronym_db_new[acronym] =  exp_article_pairs
                
                if exp_changes_articles_new is not None:
                    extend_dict_of_lists(exp_changes_articles, exp_changes_articles_new)
                pbar.update()
                
    return exp_changes_articles


def replace_exp_in_article(article_id, text, exp_changes_articles):
    if exp_changes_articles is None:
        return text
    
    exp_changes = exp_changes_articles.get(article_id)
    new_text = text
    
    if exp_changes:
        replaced_exp = []
        for change in exp_changes:
            old_exp_token = get_expansion_without_spaces(change[0])
            new_exp_token = get_expansion_without_spaces(change[1])
            (new_string, number_of_subs_made) = re.subn("\\b"+old_exp_token+"\\b",new_exp_token, new_text)
            new_text = new_string
            if number_of_subs_made < 1:
                if old_exp_token in replaced_exp:
                    logger.warning("Expansion %s already replaced in article %s", old_exp_token, article_id)
                else:
                    logger.error("When replacing expansions, it was unable to find expansion: "
                                  + change[0] + " in article: " + article_id)
                
            replaced_exp.append(old_exp_token)
    return new_text

def _replace_exp_articles(old_articles_db, new_articles_db, exp_changes_articles):
    for article_id, text in old_articles_db.items():
        new_articles_db[article_id] = replace_exp_in_article(article_id, text, exp_changes_articles)

    #if len(exp_changes_articles) > 0:
    #    logger.error("Unable to apply %d expansion replacements to articles.", len(exp_changes_articles))
        
def resolve_approximate_duplicate_expansions(dataset_name, sqlite = False):
    new_dataset_name = dataset_name + "_res-dup"
    if sqlite:
        with SqliteDict(get_acronym_db_path(dataset_name),
                         flag='r',
                         autocommit=True) as acronymDB, SqliteDict(get_acronym_db_path(new_dataset_name),
                         flag='n',
                         autocommit=True) as acronym_db_new:
            exp_changes_articles = _resolve_exp_acronym_db(acronymDB, acronym_db_new)
            
        raise NotImplementedError 
    else:
        old_acronym_db = pickle.load(open(get_acronym_db_path(dataset_name), "rb"))
        new_acronym_db = {}
        
        exp_changes_articles = _resolve_exp_acronym_db(old_acronym_db, new_acronym_db)

        pickle.dump(new_acronym_db, open(get_acronym_db_path(new_dataset_name), "wb"), protocol=2)


        articleIDToAcronymExpansions = create_article_acronym_db_from_acronym_db(
            new_acronym_db)
        pickle.dump(articleIDToAcronymExpansions, open(
            getArticleAcronymDBPath(new_dataset_name), "wb"), protocol=2)

        old_raw_articles_db = pickle.load(open(get_raw_article_db_path(dataset_name), "rb"))
        new_raw_articles_db = {}
        _replace_exp_articles(old_raw_articles_db, new_raw_articles_db, exp_changes_articles)
        
        pickle.dump(new_raw_articles_db, open(get_raw_article_db_path(new_dataset_name), "wb"), protocol=2)
        
        
        old_preprocessed_articles_db = pickle.load(open(get_preprocessed_article_db_path(dataset_name), "rb"))
        new_preprocessed_articles_db = {}
        _replace_exp_articles(old_preprocessed_articles_db, new_preprocessed_articles_db, exp_changes_articles)
        
        pickle.dump(new_preprocessed_articles_db, open(get_preprocessed_article_db_path(new_dataset_name), "wb"), protocol=2)
        
        logger.info("End")

if __name__ == '__main__':
    dataset_name = CS_WIKIPEDIA_DATASET
    resolve_approximate_duplicate_expansions(dataset_name)
