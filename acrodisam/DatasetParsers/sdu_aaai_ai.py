import json
import logging
import pickle
import re

from helper import (
    get_acronym_db_path,
    get_raw_article_db_path,
    getArticleAcronymDBPath,
    getCrossValidationFolds,
    getDatasetGeneratedFilesPath,
    getDatasetPath,
)
from string_constants import SDU_AAAI_AI_DATASET

from DatasetParsers.process_tokens_and_bio_tags import (
    create_diction,
    tokens_to_raw_text,
)

from nltk.tokenize.treebank import TreebankWordDetokenizer

logger = logging.getLogger(__name__)


def _create_article_and_acronym_db_and_article_acronym_db(dataset_name):
    """Creates the necessary databases

    Args:
        dataset_name (str): the path to the SDU AAAI AI dataset

    Returns:
        dict:
         a dictionary where each key is an acronym and each value is a list of lists with an expansion and article id
        dict:
         a dictionary where each key is an article id and each value is a dict where each key is an acronym and each value is an expansion
        dict:
         a dictionary where each key is an article id and each value is the raw text of the article
        list:
         a list with the article ids for training
        list:
         a list with the article ids for testing
    """

    dataset_path = getDatasetPath(dataset_name)

    raw_article_db = {}
    acronym_db = {}
    article_acronym_db = {}

    train_articles = []
    dev_articles = []

    train_data = json.load(open(dataset_path + "/train.json"))
    dev_data = json.load(open(dataset_path + "/dev.json"))

    for train_sample in train_data:
        pmid = train_sample["id"]
        tokens = train_sample["tokens"]
        labels = train_sample["labels"]

        # transforming the tokens back into raw text
        sentence = tokens_to_raw_text(tokens)

        raw_article_db[pmid] = sentence
        train_articles.append(pmid)

        acro_exp_dict = create_diction(tokens, labels)

        if pmid not in article_acronym_db:
            article_acronym_db[pmid] = {}

        for acro, exp in acro_exp_dict.items():
            if acro not in acronym_db:
                acronym_db[acro] = []
            acronym_db[acro].append([exp, pmid])
            article_acronym_db[pmid][acro] = exp

    for dev_sample in dev_data:
        pmid = dev_sample["id"]
        tokens = dev_sample["tokens"]
        labels = dev_sample["labels"]

        # transforming the tokens back into raw text
        sentence = tokens_to_raw_text(tokens)

        raw_article_db[pmid] = sentence
        dev_articles.append(pmid)

        acro_exp_dict = create_diction(tokens, labels)

        if pmid not in article_acronym_db:
            article_acronym_db[pmid] = {}

        for acro, exp in acro_exp_dict.items():
            if acro not in acronym_db:
                acronym_db[acro] = []
            acronym_db[acro].append([exp, pmid])
            article_acronym_db[pmid][acro] = exp

    return acronym_db, article_acronym_db, raw_article_db, train_articles, dev_articles


def make_dbs(dataset_name):
    """Pickle dumps the necessary databases.

    Creates and does a pickle dumb of five databases: a dict with the acronyms and possible expansions, a dict with the acronyms and expansions
    per article, a dict with the raw text per article, a list with the article ids to train and a list with article ids to test,

    Args:
        dataset_name (str): the path to the SDU AAAI AI dataset
    """

    folds_num = 5
    (
        acronym_db,
        article_acronym_db,
        article_db,
        train_articles,
        test_articles,
    ) = _create_article_and_acronym_db_and_article_acronym_db(dataset_name)

    pickle.dump(
        article_db, open(get_raw_article_db_path(dataset_name), "wb"), protocol=2
    )
    pickle.dump(acronym_db, open(get_acronym_db_path(dataset_name), "wb"), protocol=2)
    pickle.dump(
        article_acronym_db,
        open(getArticleAcronymDBPath(dataset_name), "wb"),
        protocol=2,
    )

    generated_files_folder = getDatasetGeneratedFilesPath(dataset_name)

    pickle.dump(
        train_articles,
        open(generated_files_folder + "train_articles.pickle", "wb"),
        protocol=2,
    )
    pickle.dump(
        test_articles,
        open(generated_files_folder + "test_articles.pickle", "wb"),
        protocol=2,
    )

    folds_file_path = (
        generated_files_folder + str(folds_num) + "-cross-validation.pickle"
    )

    folds = getCrossValidationFolds(train_articles, folds_num)
    pickle.dump(folds, open(folds_file_path, "wb"), protocol=2)


if __name__ == "__main__":
    make_dbs(SDU_AAAI_AI_DATASET)
