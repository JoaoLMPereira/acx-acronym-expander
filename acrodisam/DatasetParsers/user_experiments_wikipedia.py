"""Creates necessary databases for Users Wikipedia

Created April 2021

@author: JRCasanova
"""
import csv
import functools
import pickle
import random
from concurrent.futures import ProcessPoolExecutor

from helper import (
    get_acronym_db_path,
    get_raw_article_db_path,
    getArticleAcronymDBPath,
    getArticleDBPath,
    getDatasetGeneratedFilesPath,
    getDatasetPath,
)
from Logger import logging
from string_constants import (
    DB_WITH_LINKS_SUFFIX,
    FILE_USER_WIKIPIDIA_ANNOTATIONS,
    FULL_WIKIPEDIA_DATASET,
    USERS_WIKIPEDIA,
)
from tqdm import tqdm

from DatasetParsers.pre_user_experiments_wikipedia_preparation import (
    get_wiki_file_path,
    getDocText,
    process_wiki_file,
    processAnnotations,
)

logger = logging.getLogger(__name__)


def create_articles_db_with_links(user_annotations_file, wikipath):
    """Creates a pickle dump of a dictionary with wiki articles.

    Creates a database which is a dictionary where the keys are the wikipedia IDs
    of the articles annotated and the values are the text of the wiki articles with links (not plain text).

    Args:
        user_annotations_file (str):
         csv file with the user annotations for the wikipedia articles.
        wikipath (str):
         path to the data folder for Full Wikipedia that should contain all of English wikipedia
        with links in the text.
    """

    article_db = {}

    path = getDatasetPath(USERS_WIKIPEDIA)

    articles_to_get = set()

    with open(path + user_annotations_file) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(csvreader)
        for row in csvreader:

            wiki_id = row[2]

            articles_to_get.add(int(wiki_id))

    file_paths_list = get_wiki_file_path(wikipath)

    tasks_num = len(file_paths_list)
    partial_func = functools.partial(process_wiki_file, articles_to_get=articles_to_get)
    with ProcessPoolExecutor() as process_pool:
        with tqdm(total=tasks_num) as pbar:
            for _, results in tqdm(
                enumerate(process_pool.map(partial_func, file_paths_list, chunksize=1))
            ):
                for r in results:
                    doc_id = r[0]
                    text = r[1]
                    article_db[doc_id] = text
                pbar.update()

    # Save Database

    pickle.dump(
        article_db,
        open(getArticleDBPath(USERS_WIKIPEDIA) + DB_WITH_LINKS_SUFFIX, "wb"),
        protocol=2,
    )


def create_dbs(user_annotations_file):
    """Creates 3 pickle dumps for three databases: one with wiki text, one with acronyms/expansions, one with wiki IDs.

    Creates three databases: articleDB, articleIDtoAcronymExpansions and testArticles. articleDB is a dictionary where the keys are wiki IDs
    and the values are the plain text of the wikipedia articles (no links). articleIDtoAcronymExpansions is a dictionary where the keys are wiki IDs
    and the values are another dictionary in which the keys are acronyms and the values are tuples like: (expansion, bool indicating presence in text).
    testArticles is a simple list with the wiki IDs of all the articles.

    Args:
        user_annotations_file (str):
         csv file with the user annotations for the wikipedia articles.

    """

    article_ids = []
    article_id_to_acronym_expansions = {}
    article_db = {}

    path = getDatasetPath(USERS_WIKIPEDIA)

    with open(path + user_annotations_file) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(csvreader)
        for row in csvreader:

            doc_title = row[1]
            text = getDocText(doc_title)
            wiki_id = row[2]

            article_db[wiki_id] = text
            article_ids.append(wiki_id)

            in_exp_annotations = processAnnotations(
                row[3]
            )  # Annotations for acronyms that have the expansion in text
            out_exp_annotations = processAnnotations(
                row[4]
            )  # Annotations for acronym that DO NOT have the expansion in text

            if len(in_exp_annotations) == 0 and len(out_exp_annotations) == 0:
                article_id_to_acronym_expansions[wiki_id] = {}
            else:
                if wiki_id not in article_id_to_acronym_expansions:
                    article_id_to_acronym_expansions[wiki_id] = {}

                if len(in_exp_annotations) != 0:
                    for (acro, exp) in in_exp_annotations:
                        article_id_to_acronym_expansions[wiki_id][acro] = (exp, True)

                if len(out_exp_annotations) != 0:
                    for (acro, exp) in out_exp_annotations:
                        article_id_to_acronym_expansions[wiki_id][acro] = (exp, False)

    random.seed(5)
    random.shuffle(article_ids)
    split_article_id = int(len(article_ids) * 0.7)
    train_articles = article_ids[:split_article_id]
    test_articles = article_ids[split_article_id:]

    # Save Databases
    pickle.dump(
        article_db, open(get_raw_article_db_path(USERS_WIKIPEDIA), "wb"), protocol=2
    )

    pickle.dump(
        article_id_to_acronym_expansions,
        open(getArticleAcronymDBPath(USERS_WIKIPEDIA), "wb"),
        protocol=2,
    )

    generatedFilesFolder = getDatasetGeneratedFilesPath(USERS_WIKIPEDIA)

    pickle.dump(
        train_articles,
        open(generatedFilesFolder + "train_articles.pickle", "wb"),
        protocol=2,
    )

    pickle.dump(
        test_articles, open(generatedFilesFolder + "articles.pickle", "wb"), protocol=2
    )


def create_acronym_db():
    """Creates a pickle dump of a dictionary with the possible expansion for an acronym.

    Creates database that is a dictionary where the keys are acronyms and the values are tuples
    like: (possible expansion, wiki ID, boolean indicating presence in text)

    """

    article_id_to_acronym_expansions = pickle.load(
        open(getArticleAcronymDBPath(USERS_WIKIPEDIA), "rb")
    )

    acronym_db = {}

    for article_id, acro_exp in article_id_to_acronym_expansions.items():
        for acro, exp in acro_exp.items():
            exp_list = acronym_db.get(acro, [])
            exp_list.append((exp[0], article_id, exp[1]))
            acronym_db[acro] = exp_list

    pickle.dump(
        acronym_db, open(get_acronym_db_path(USERS_WIKIPEDIA), "wb"), protocol=2
    )


if __name__ == "__main__":
    create_articles_db_with_links(
        FILE_USER_WIKIPIDIA_ANNOTATIONS,
        getDatasetPath(FULL_WIKIPEDIA_DATASET),
    )

    create_dbs(FILE_USER_WIKIPIDIA_ANNOTATIONS)

    create_acronym_db()
