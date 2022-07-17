"""Evaluates user performance for annotations of Wikipedia

Created on May 2021

@author: JRCasanova
"""


import csv
import pickle

from helper import (
    AcronymExpansion,
    get_acronym_db_path,
    get_raw_article_db_path,
    getArticleAcronymDBPath,
    getDatasetGeneratedFilesPath,
    getDatasetPath,
)
from Logger import logging
from nltk import word_tokenize
from nltk.metrics import agreement
from nltk.metrics.distance import jaccard_distance, masi_distance, edit_distance
from string_constants import (
    DB_WITH_LINKS_SUFFIX,
    FILE_USER_WIKIPEDIA_ANNOTATIONS_RAW,
    FOLDER_DATA,
    USERS_WIKIPEDIA,
)

from DatasetParsers.pre_user_experiments_wikipedia_preparation import (
    get_wiki_file_path,
    getDocText,
    getFinalPages,
    getStringAnnotations,
    loadPageWikiIds,
    process_wiki_file,
    processAnnotations,
)

from DatasetParsers.expansion_linkage import _resolve_exp_acronym_db

from nltk.tokenize.regexp import RegexpTokenizer

tokenizer = RegexpTokenizer(r"\w+")

logger = logging.getLogger(__name__)


def compare_annotations(doc_id, wiki_id, actual_expansions, predicted_expansions):
    """Returns the number of correct, extracted and gold annotations on a wikipedia article.

    Args:
        doc_id (str): the document id for the wikipedia article
        wiki_id (str): the wikipedia id for the wikipedia article
        actual_expansions (dict): the final expansions selected by the reviewer in form of a dict. For example:

         {123456789: {'CIA': ('Central Intelligence Agency', True)} ...}

        predicted_expansions (dict): the annotations made by the annotator for this particular article. For example:

         {'CIA': 'Central Intelligence Agency',
          'NSA': 'National Security Agency'}
    Returns:
        dict:
         a dictionary with the number of correct, extracted and gold annotations for the acronyms, expansions and
         acronym-expansion pairs
    """

    results = {
        "Acronyms": {"correct": 0, "extracted": 0, "gold": 0},
        "Expansions": {"correct": 0, "extracted": 0, "gold": 0},
        "Pairs": {"correct": 0, "extracted": 0, "gold": 0},
    }

    for acronym, actual_expansion in actual_expansions.items():
        results["Acronyms"]["gold"] += 1
        results["Expansions"]["gold"] += 1
        results["Pairs"]["gold"] += 1

        predicted_expansion = predicted_expansions.pop(acronym, None)

        # try removing s or add s
        if predicted_expansion is None and len(acronym) > 1 and acronym[-1] == "s":
            predicted_expansion = predicted_expansions.pop(acronym[:-1], None)
        elif predicted_expansion is None and acronym[-1] != "s":
            predicted_expansion = predicted_expansions.pop(acronym + "s", None)

        if predicted_expansion:

            results["Acronyms"]["extracted"] += 1
            results["Expansions"]["extracted"] += 1
            results["Pairs"]["extracted"] += 1

            results["Acronyms"]["correct"] += 1

            if AcronymExpansion.areExpansionsSimilar(
                actual_expansion.strip().lower(), predicted_expansion.strip().lower()
            ):
                results["Expansions"]["correct"] += 1
                results["Pairs"]["correct"] += 1
                logger.debug(
                    "Expansion matching succeeded in doc_id %s wiki_id %s, (%s): %s, %s"
                    % (
                        doc_id,
                        wiki_id,
                        acronym,
                        actual_expansion,
                        predicted_expansion,
                    )
                )
            else:
                logger.debug(
                    "Expansion matching failed in doc_id %s wiki_id %s (%s): %s, %s"
                    % (
                        doc_id,
                        wiki_id,
                        acronym,
                        actual_expansion,
                        predicted_expansion,
                    )
                )
    for acronym in predicted_expansions:
        results["Acronyms"]["extracted"] += 1
        results["Pairs"]["extracted"] += 1
        logger.debug(
            "Expansion matching failed in doc_id %s wiki_id %s (%s): %s, %s"
            % (doc_id, wiki_id, acronym, None, predicted_expansions[acronym])
        )

    return results


def evaluate_user_responses(user_annotations_raw_file):
    """Logs performance metrics for the user annotations of Wikipedia articles

    Logs precision, recall and f1-score for acronyms, expansions and acronym-expansion pairs.
    These metrics are logged and separated for acronyms with and without expansion in text.

    Args:
        user_annotations_raw_file (str):
         csv file with the two user annotations for each wikipedia article. This should
         be present in the data folder for users wikipedia
    """
    article_id_to_acronym_expansions = pickle.load(
        open(getArticleAcronymDBPath(USERS_WIKIPEDIA), "rb")
    )

    path = getDatasetPath(USERS_WIKIPEDIA)

    test_articles = pickle.load(
        open(getDatasetGeneratedFilesPath(USERS_WIKIPEDIA) + "articles.pickle", "rb")
    )

    results_all_in = {
        "Acronyms": {"correct": 0, "extracted": 0, "gold": 0},
        "Expansions": {"correct": 0, "extracted": 0, "gold": 0},
        "Pairs": {"correct": 0, "extracted": 0, "gold": 0},
    }

    results_all_out = {
        "Acronyms": {"correct": 0, "extracted": 0, "gold": 0},
        "Expansions": {"correct": 0, "extracted": 0, "gold": 0},
        "Pairs": {"correct": 0, "extracted": 0, "gold": 0},
    }

    with open(path + user_annotations_raw_file) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(csvreader)
        for row in csvreader:

            doc_id = str(row[0])
            wiki_id = str(row[2])

            if wiki_id not in test_articles:
                continue

            annotator_1_in = {
                k.lower(): v for k, v in dict(processAnnotations(row[3])).items()
            }

            annotator_1_out = {
                k.lower(): v for k, v in dict(processAnnotations(row[4])).items()
            }

            annotator_2_in = {
                k.lower(): v for k, v in dict(processAnnotations(row[5])).items()
            }

            annotator_2_out = {
                k.lower(): v for k, v in dict(processAnnotations(row[6])).items()
            }

            reviewer_in = {
                k.lower(): v[0]
                for k, v in article_id_to_acronym_expansions.get(wiki_id, {}).items()
                if v[1]
            }

            reviewer_out = {
                k.lower(): v[0]
                for k, v in article_id_to_acronym_expansions.get(wiki_id, {}).items()
                if not v[1]
            }

            results = compare_annotations(doc_id, wiki_id, reviewer_in, annotator_1_in)
            results_all_in["Acronyms"]["correct"] += results["Acronyms"]["correct"]
            results_all_in["Acronyms"]["extracted"] += results["Acronyms"]["extracted"]
            results_all_in["Acronyms"]["gold"] += results["Acronyms"]["gold"]
            results_all_in["Expansions"]["correct"] += results["Expansions"]["correct"]
            results_all_in["Expansions"]["extracted"] += results["Expansions"][
                "extracted"
            ]
            results_all_in["Expansions"]["gold"] += results["Expansions"]["gold"]
            results_all_in["Pairs"]["correct"] += results["Pairs"]["correct"]
            results_all_in["Pairs"]["extracted"] += results["Pairs"]["extracted"]
            results_all_in["Pairs"]["gold"] += results["Pairs"]["gold"]
            results = compare_annotations(doc_id, wiki_id, reviewer_in, annotator_2_in)
            results_all_in["Acronyms"]["correct"] += results["Acronyms"]["correct"]
            results_all_in["Acronyms"]["extracted"] += results["Acronyms"]["extracted"]
            results_all_in["Acronyms"]["gold"] += results["Acronyms"]["gold"]
            results_all_in["Expansions"]["correct"] += results["Expansions"]["correct"]
            results_all_in["Expansions"]["extracted"] += results["Expansions"][
                "extracted"
            ]
            results_all_in["Expansions"]["gold"] += results["Expansions"]["gold"]
            results_all_in["Pairs"]["correct"] += results["Pairs"]["correct"]
            results_all_in["Pairs"]["extracted"] += results["Pairs"]["extracted"]
            results_all_in["Pairs"]["gold"] += results["Pairs"]["gold"]

            results = compare_annotations(
                doc_id, wiki_id, reviewer_out, annotator_1_out
            )
            results_all_out["Acronyms"]["correct"] += results["Acronyms"]["correct"]
            results_all_out["Acronyms"]["extracted"] += results["Acronyms"]["extracted"]
            results_all_out["Acronyms"]["gold"] += results["Acronyms"]["gold"]
            results_all_out["Expansions"]["correct"] += results["Expansions"]["correct"]
            results_all_out["Expansions"]["extracted"] += results["Expansions"][
                "extracted"
            ]
            results_all_out["Expansions"]["gold"] += results["Expansions"]["gold"]
            results_all_out["Pairs"]["correct"] += results["Pairs"]["correct"]
            results_all_out["Pairs"]["extracted"] += results["Pairs"]["extracted"]
            results_all_out["Pairs"]["gold"] += results["Pairs"]["gold"]
            results = compare_annotations(
                doc_id, wiki_id, reviewer_out, annotator_2_out
            )
            results_all_out["Acronyms"]["correct"] += results["Acronyms"]["correct"]
            results_all_out["Acronyms"]["extracted"] += results["Acronyms"]["extracted"]
            results_all_out["Acronyms"]["gold"] += results["Acronyms"]["gold"]
            results_all_out["Expansions"]["correct"] += results["Expansions"]["correct"]
            results_all_out["Expansions"]["extracted"] += results["Expansions"][
                "extracted"
            ]
            results_all_out["Expansions"]["gold"] += results["Expansions"]["gold"]
            results_all_out["Pairs"]["correct"] += results["Pairs"]["correct"]
            results_all_out["Pairs"]["extracted"] += results["Pairs"]["extracted"]
            results_all_out["Pairs"]["gold"] += results["Pairs"]["gold"]

    logger.info("Metrics Total")

    precision_acronyms_total = (
        results_all_in["Acronyms"]["correct"] + results_all_out["Acronyms"]["correct"]
    ) / (
        results_all_in["Acronyms"]["extracted"]
        + results_all_out["Acronyms"]["extracted"]
    )

    recall_acronyms_total = (
        results_all_in["Acronyms"]["correct"] + results_all_out["Acronyms"]["correct"]
    ) / (results_all_in["Acronyms"]["gold"] + results_all_out["Acronyms"]["gold"])

    f1_score_acronyms_total = 2 * (
        (precision_acronyms_total * recall_acronyms_total)
        / (precision_acronyms_total + recall_acronyms_total)
    )

    precision_expansions_total = (
        results_all_in["Expansions"]["correct"]
        + results_all_out["Expansions"]["correct"]
    ) / (
        results_all_in["Expansions"]["extracted"]
        + results_all_out["Expansions"]["extracted"]
    )
    recall_expansions_total = (
        results_all_in["Expansions"]["correct"]
        + results_all_out["Expansions"]["correct"]
    ) / (results_all_in["Expansions"]["gold"] + results_all_out["Expansions"]["gold"])

    f1_score_expansions_total = 2 * (
        (precision_expansions_total * recall_expansions_total)
        / (precision_expansions_total + recall_expansions_total)
    )

    precision_pairs_total = (
        results_all_in["Pairs"]["correct"] + results_all_out["Pairs"]["correct"]
    ) / (results_all_in["Pairs"]["extracted"] + results_all_out["Pairs"]["extracted"])

    recall_pairs_total = (
        results_all_in["Pairs"]["correct"] + results_all_out["Pairs"]["correct"]
    ) / (results_all_in["Pairs"]["gold"] + results_all_out["Pairs"]["gold"])

    f1_score_pairs_total = 2 * (
        (precision_pairs_total * recall_pairs_total)
        / (precision_pairs_total + recall_pairs_total)
    )

    logger.info("Precision for Acronyms: %f", precision_acronyms_total)
    logger.info("Recall for Acronyms: %f", recall_acronyms_total)
    logger.info("F1-Score for Acronyms: %f", f1_score_acronyms_total)

    logger.info("Precision for Expansions: %f", precision_expansions_total)
    logger.info("Recall for Expansions: %f", recall_expansions_total)
    logger.info("F1-Score for Expansions: %f", f1_score_expansions_total)

    logger.info("Precision for Pairs: %f", precision_pairs_total)
    logger.info("Recall for Pairs: %f", recall_pairs_total)
    logger.info("F1-Score for Pairs: %f", f1_score_pairs_total)

    logger.info("Metrics for Acronym Idenfication")

    precision_acronyms_in = (
        results_all_in["Acronyms"]["correct"] / results_all_in["Acronyms"]["extracted"]
    )
    recall_acronyms_in = (
        results_all_in["Acronyms"]["correct"] / results_all_in["Acronyms"]["gold"]
    )
    f1_score_acronyms_in = 2 * (
        (precision_acronyms_in * recall_acronyms_in)
        / (precision_acronyms_in + recall_acronyms_in)
    )

    precision_expansions_in = (
        results_all_in["Expansions"]["correct"]
        / results_all_in["Expansions"]["extracted"]
    )
    recall_expansions_in = (
        results_all_in["Expansions"]["correct"] / results_all_in["Expansions"]["gold"]
    )
    f1_score_expansions_in = 2 * (
        (precision_expansions_in * recall_expansions_in)
        / (precision_expansions_in + recall_expansions_in)
    )

    precision_pairs_in = (
        results_all_in["Pairs"]["correct"] / results_all_in["Pairs"]["extracted"]
    )
    recall_pairs_in = (
        results_all_in["Pairs"]["correct"] / results_all_in["Pairs"]["gold"]
    )
    f1_score_pairs_in = 2 * (
        (precision_pairs_in * recall_pairs_in) / (precision_pairs_in + recall_pairs_in)
    )

    logger.info("Precision for Acronyms: %f", precision_acronyms_in)
    logger.info("Recall for Acronyms: %f", recall_acronyms_in)
    logger.info("F1-Score for Acronyms: %f", f1_score_acronyms_in)

    logger.info("Precision for Expansions: %f", precision_expansions_in)
    logger.info("Recall for Expansions: %f", recall_expansions_in)
    logger.info("F1-Score for Expansions: %f", f1_score_expansions_in)

    logger.info("Precision for Pairs: %f", precision_pairs_in)
    logger.info("Recall for Pairs: %f", recall_pairs_in)
    logger.info("F1-Score for Pairs: %f", f1_score_pairs_in)

    logger.info("Metrics for Acronym Disambiguation")

    precision_acronyms_out = (
        results_all_out["Acronyms"]["correct"]
        / results_all_out["Acronyms"]["extracted"]
    )
    recall_acronyms_out = (
        results_all_out["Acronyms"]["correct"] / results_all_out["Acronyms"]["gold"]
    )
    f1_score_acronyms_out = 2 * (
        (precision_acronyms_out * recall_acronyms_out)
        / (precision_acronyms_out + recall_acronyms_out)
    )

    precision_expansions_out = (
        results_all_out["Expansions"]["correct"]
        / results_all_out["Expansions"]["extracted"]
    )
    recall_expansions_out = (
        results_all_out["Expansions"]["correct"] / results_all_out["Expansions"]["gold"]
    )
    f1_score_expansions_out = 2 * (
        (precision_expansions_out * recall_expansions_out)
        / (precision_expansions_out + recall_expansions_out)
    )

    precision_pairs_out = (
        results_all_out["Pairs"]["correct"] / results_all_out["Pairs"]["extracted"]
    )
    recall_pairs_out = (
        results_all_out["Pairs"]["correct"] / results_all_out["Pairs"]["gold"]
    )
    f1_score_pairs_out = 2 * (
        (precision_pairs_out * recall_pairs_out)
        / (precision_pairs_out + recall_pairs_out)
    )

    logger.info("Precision for Acronyms: %f", precision_acronyms_out)
    logger.info("Recall for Acronyms: %f", recall_acronyms_out)
    logger.info("F1-Score for Acronyms: %f", f1_score_acronyms_out)

    logger.info("Precision for Expansions: %f", precision_expansions_out)
    logger.info("Recall for Expansions: %f", recall_expansions_out)
    logger.info("F1-Score for Expansions: %f", f1_score_expansions_out)

    logger.info("Precision for Pairs: %f", precision_pairs_out)
    logger.info("Recall for Pairs: %f", recall_pairs_out)
    logger.info("F1-Score for Pairs: %f", f1_score_pairs_out)


def calculate_iaa(user_annotations_raw_file):
    """Logs the IAA using Krippendorff’s alpha and Cohen's kappa with Jaccard and Masi distance for users annotations.

    Logs the the Krippendorff’s alpha and Cohen's kappa for all annotations as well as for
    Acronym Identification and Acronym Disambiguation with Jaccard/Masi distance
    for users annotations considering the acronyms, expansions and acronym/expansion pairs.

    Args:
        user_annotations_raw_file (str):
        csv file with the two user annotations for each wikipedia article. The annotations
        should be separated in acronyms with/without expansion in text. This should
        be present in the data folder for user wikipedia.
    """
    annotations_data_acro_in = []
    annotations_data_exp_in = []
    annotations_data_acro_exp_in = []

    annotations_data_acro_out = []
    annotations_data_exp_out = []
    annotations_data_acro_exp_out = []

    annotations_data_acro_total = []
    annotations_data_exp_total = []
    annotations_data_acro_exp_total = []

    path = getDatasetPath(USERS_WIKIPEDIA)

    with open(path + user_annotations_raw_file) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(csvreader)
        for row in csvreader:

            annotations_u1_in = {
                k.lower(): v for k, v in dict(processAnnotations(row[3])).items()
            }

            annotations_u1_out = {
                k.lower(): v for k, v in dict(processAnnotations(row[4])).items()
            }

            annotations_u1_total = {**annotations_u1_in, **annotations_u1_out}

            annotations_u2_in = {
                k.lower(): v for k, v in dict(processAnnotations(row[5])).items()
            }

            annotations_u2_out = {
                k.lower(): v for k, v in dict(processAnnotations(row[6])).items()
            }

            annotations_u2_total = {**annotations_u2_in, **annotations_u2_out}

            if len(annotations_u1_in) != 0 and len(annotations_u2_in) != 0:

                u1_acro_in = set()
                u2_acro_in = set()
                u1_exp_in = set()
                u2_exp_in = set()
                u1_acro_exp_in = set()
                u2_acro_exp_in = set()

                for acro, exp_u1 in annotations_u1_in.items():

                    exp_u2 = annotations_u2_in.pop(acro, None)

                    if exp_u2 is None and len(acro) > 1 and acro[-1] == "s":
                        exp_u2 = annotations_u2_in.pop(acro[:-1], None)
                    elif exp_u2 is None and acro[-1] != "s":
                        exp_u2 = annotations_u2_in.pop(acro + "s", None)

                    if exp_u2 != None:
                        if AcronymExpansion.areExpansionsSimilar(
                            exp_u1.strip().lower(), exp_u2.strip().lower()
                        ):
                            u1_acro_in.add(acro)
                            u2_acro_in.add(acro)
                            u1_exp_in.add(exp_u1)
                            u2_exp_in.add(exp_u1)
                            u1_acro_exp_in.add((acro, exp_u1))
                            u2_acro_exp_in.add((acro, exp_u1))
                        else:
                            u1_acro_in.add(acro)
                            u2_acro_in.add(acro)
                            u1_exp_in.add(exp_u1)
                            u2_exp_in.add(exp_u2)
                            u1_acro_exp_in.add((acro, exp_u1))
                            u2_acro_exp_in.add((acro, exp_u2))
                    else:
                        u1_acro_in.add(acro)
                        u1_exp_in.add(exp_u1)
                        u1_acro_exp_in.add((acro, exp_u1))
                for acro, exp_u2 in annotations_u2_in.items():
                    u2_acro_in.add(acro)
                    u2_exp_in.add(exp_u2)
                    u2_acro_exp_in.add((acro, exp_u2))

                annotations_data_acro_in.append(
                    ("u1", int(row[0]), frozenset(u1_acro_in))
                )
                annotations_data_acro_in.append(
                    ("u2", int(row[0]), frozenset(u2_acro_in))
                )

                annotations_data_exp_in.append(
                    ("u1", int(row[0]), frozenset(u1_exp_in))
                )
                annotations_data_exp_in.append(
                    ("u2", int(row[0]), frozenset(u2_exp_in))
                )

                annotations_data_acro_exp_in.append(
                    ("u1", int(row[0]), frozenset(u1_acro_exp_in))
                )
                annotations_data_acro_exp_in.append(
                    ("u2", int(row[0]), frozenset(u2_acro_exp_in))
                )

            if len(annotations_u1_out) != 0 and len(annotations_u2_out) != 0:

                u1_acro_out = set()
                u2_acro_out = set()
                u1_exp_out = set()
                u2_exp_out = set()
                u1_acro_exp_out = set()
                u2_acro_exp_out = set()

                for acro, exp_u1 in annotations_u1_out.items():

                    exp_u2 = annotations_u2_out.pop(acro, None)

                    if exp_u2 is None and len(acro) > 1 and acro[-1] == "s":
                        exp_u2 = annotations_u2_out.pop(acro[:-1], None)
                    elif exp_u2 is None and acro[-1] != "s":
                        exp_u2 = annotations_u2_out.pop(acro + "s", None)

                    if exp_u2 != None:
                        if AcronymExpansion.areExpansionsSimilar(
                            exp_u1.strip().lower(), exp_u2.strip().lower()
                        ):
                            u1_acro_out.add(acro)
                            u2_acro_out.add(acro)
                            u1_exp_out.add(exp_u1)
                            u2_exp_out.add(exp_u1)
                            u1_acro_exp_out.add((acro, exp_u1))
                            u2_acro_exp_out.add((acro, exp_u1))
                        else:
                            u1_acro_out.add(acro)
                            u2_acro_out.add(acro)
                            u1_exp_out.add(exp_u1)
                            u2_exp_out.add(exp_u2)
                            u1_acro_exp_out.add((acro, exp_u1))
                            u2_acro_exp_out.add((acro, exp_u2))
                    else:
                        u1_acro_out.add(acro)
                        u1_exp_out.add(exp_u1)
                        u1_acro_exp_out.add((acro, exp_u1))
                for acro, exp_u2 in annotations_u2_out.items():
                    u2_acro_out.add(acro)
                    u2_exp_out.add(exp_u2)
                    u2_acro_exp_out.add((acro, exp_u2))

                annotations_data_acro_out.append(
                    ("u1", int(row[0]), frozenset(u1_acro_out))
                )
                annotations_data_acro_out.append(
                    ("u2", int(row[0]), frozenset(u2_acro_out))
                )

                annotations_data_exp_out.append(
                    ("u1", int(row[0]), frozenset(u1_exp_out))
                )
                annotations_data_exp_out.append(
                    ("u2", int(row[0]), frozenset(u2_exp_out))
                )

                annotations_data_acro_exp_out.append(
                    ("u1", int(row[0]), frozenset(u1_acro_exp_out))
                )
                annotations_data_acro_exp_out.append(
                    ("u2", int(row[0]), frozenset(u2_acro_exp_out))
                )

            if len(annotations_u1_total) != 0 and len(annotations_u2_total) != 0:
                u1_acro_total = set()
                u2_acro_total = set()
                u1_exp_total = set()
                u2_exp_total = set()
                u1_acro_exp_total = set()
                u2_acro_exp_total = set()

                for acro, exp_u1 in annotations_u1_total.items():

                    exp_u2 = annotations_u2_total.pop(acro, None)

                    if exp_u2 is None and len(acro) > 1 and acro[-1] == "s":
                        exp_u2 = annotations_u2_total.pop(acro[:-1], None)
                    elif exp_u2 is None and acro[-1] != "s":
                        exp_u2 = annotations_u2_total.pop(acro + "s", None)

                    if exp_u2 != None:
                        if AcronymExpansion.areExpansionsSimilar(
                            exp_u1.strip().lower(), exp_u2.strip().lower()
                        ):
                            u1_acro_total.add(acro)
                            u2_acro_total.add(acro)
                            u1_exp_total.add(exp_u1)
                            u2_exp_total.add(exp_u1)
                            u1_acro_exp_total.add((acro, exp_u1))
                            u2_acro_exp_total.add((acro, exp_u1))
                        else:
                            u1_acro_total.add(acro)
                            u2_acro_total.add(acro)
                            u1_exp_total.add(exp_u1)
                            u2_exp_total.add(exp_u2)
                            u1_acro_exp_total.add((acro, exp_u1))
                            u2_acro_exp_total.add((acro, exp_u2))
                    else:
                        u1_acro_total.add(acro)
                        u1_exp_total.add(exp_u1)
                        u1_acro_exp_total.add((acro, exp_u1))
                for acro, exp_u2 in annotations_u2_total.items():
                    u2_acro_total.add(acro)
                    u2_exp_total.add(exp_u2)
                    u2_acro_exp_total.add((acro, exp_u2))

                annotations_data_acro_total.append(
                    ("u1", int(row[0]), frozenset(u1_acro_total))
                )
                annotations_data_acro_total.append(
                    ("u2", int(row[0]), frozenset(u2_acro_total))
                )

                annotations_data_exp_total.append(
                    ("u1", int(row[0]), frozenset(u1_exp_total))
                )
                annotations_data_exp_total.append(
                    ("u2", int(row[0]), frozenset(u2_exp_total))
                )

                annotations_data_acro_exp_total.append(
                    ("u1", int(row[0]), frozenset(u1_acro_exp_total))
                )
                annotations_data_acro_exp_total.append(
                    ("u2", int(row[0]), frozenset(u2_acro_exp_total))
                )

    logger.info(
        "Inter-annotator agreement for all annotations (Krippendorff’s alpha and Cohen's Kappa)"
    )

    task_acro_total = agreement.AnnotationTask(
        data=annotations_data_acro_total, distance=jaccard_distance
    )

    task_exp_total = agreement.AnnotationTask(
        data=annotations_data_exp_total, distance=jaccard_distance
    )

    task_acro_exp_total = agreement.AnnotationTask(
        data=annotations_data_acro_exp_total, distance=jaccard_distance
    )

    logger.info(
        "Krippendorff’s alpha for Acronyms with Jaccard: %f", task_acro_total.alpha()
    )
    logger.info("Cohen's Kappa for Acronyms with Jaccard: %f", task_acro_total.kappa())
    logger.info(
        "Krippendorff’s alpha for Expansions with Jaccard: %f", task_exp_total.alpha()
    )
    logger.info("Cohen's Kappa for Expansions with Jaccard: %f", task_exp_total.kappa())
    logger.info(
        "Krippendorff’s alpha for Pairs with Jaccard: %f", task_acro_exp_total.alpha()
    )
    logger.info("Cohen's Kappa for Pairs with Jaccard: %f", task_acro_exp_total.kappa())

    task_acro_total = agreement.AnnotationTask(
        data=annotations_data_acro_total, distance=masi_distance
    )

    task_exp_total = agreement.AnnotationTask(
        data=annotations_data_exp_total, distance=masi_distance
    )

    task_acro_exp_total = agreement.AnnotationTask(
        data=annotations_data_acro_exp_total, distance=masi_distance
    )

    logger.info(
        "Krippendorff’s alpha for Acronyms with MISA: %f", task_acro_total.alpha()
    )
    logger.info("Cohen's Kappa for Acronyms with MISA: %f", task_acro_total.kappa())
    logger.info(
        "Krippendorff’s alpha for Expansions with MISA: %f", task_exp_total.alpha()
    )
    logger.info("Cohen's Kappa for Expansions with MISA: %f", task_exp_total.kappa())
    logger.info(
        "Krippendorff’s alpha for Pairs with MISA: %f", task_acro_exp_total.alpha()
    )
    logger.info("Cohen's Kappa for Pairs with MISA: %f", task_acro_exp_total.kappa())

    logger.info(
        "Inter-annotator agreement for Acronym Identification (Krippendorff’s alpha and Cohen's Kappa)"
    )

    task_acro_in = agreement.AnnotationTask(
        data=annotations_data_acro_in, distance=jaccard_distance
    )

    task_exp_in = agreement.AnnotationTask(
        data=annotations_data_exp_in, distance=jaccard_distance
    )

    task_acro_exp_in = agreement.AnnotationTask(
        data=annotations_data_acro_exp_in, distance=jaccard_distance
    )

    logger.info(
        "Krippendorff’s alpha for Acronyms with Jaccard: %f", task_acro_in.alpha()
    )
    logger.info("Cohen's Kappa for Acronyms with Jaccard: %f", task_acro_in.kappa())
    logger.info(
        "Krippendorff’s alpha for Expansions with Jaccard: %f", task_exp_in.alpha()
    )
    logger.info("Cohen's Kappa for Expansions with Jaccard: %f", task_exp_in.kappa())
    logger.info(
        "Krippendorff’s alpha for Pairs with Jaccard: %f", task_acro_exp_in.alpha()
    )
    logger.info("Cohen's Kappa for Pairs with Jaccard: %f", task_acro_exp_in.kappa())

    task_acro_in = agreement.AnnotationTask(
        data=annotations_data_acro_in, distance=masi_distance
    )

    task_exp_in = agreement.AnnotationTask(
        data=annotations_data_exp_in, distance=masi_distance
    )

    task_acro_exp_in = agreement.AnnotationTask(
        data=annotations_data_acro_exp_in, distance=masi_distance
    )

    logger.info("Krippendorff’s alpha for Acronyms with MISA: %f", task_acro_in.alpha())
    logger.info("Cohen's Kappa for Acronyms with MISA: %f", task_acro_in.kappa())
    logger.info(
        "Krippendorff’s alpha for Expansions with MISA: %f", task_exp_in.alpha()
    )
    logger.info("Cohen's Kappa for Expansions with MISA: %f", task_exp_in.kappa())
    logger.info(
        "Krippendorff’s alpha for Pairs with MISA: %f", task_acro_exp_in.alpha()
    )
    logger.info("Cohen's Kappa for Pairs with MISA: %f", task_acro_exp_in.kappa())

    logger.info(
        "Inter-annotator agreement for Acronym Disambiguation (Krippendorff’s alpha and Cohen's Kappa)"
    )

    task_acro_out = agreement.AnnotationTask(
        data=annotations_data_acro_out, distance=jaccard_distance
    )

    task_exp_out = agreement.AnnotationTask(
        data=annotations_data_exp_out, distance=jaccard_distance
    )

    task_acro_exp_out = agreement.AnnotationTask(
        data=annotations_data_acro_exp_out, distance=jaccard_distance
    )

    logger.info(
        "Krippendorff’s alpha for Acronyms with Jaccard: %f", task_acro_out.alpha()
    )
    logger.info("Cohen's Kappa for Acronyms with Jaccard: %f", task_acro_out.kappa())
    logger.info(
        "Krippendorff’s alpha for Expansions with Jaccard: %f", task_exp_out.alpha()
    )
    logger.info("Cohen's Kappa for Expansions with Jaccard: %f", task_exp_out.kappa())
    logger.info(
        "Krippendorff’s alpha for Pairs with Jaccard: %f", task_acro_exp_out.alpha()
    )
    logger.info("Cohen's Kappa for Pairs with Jaccard: %f", task_acro_exp_out.kappa())

    task_acro_out = agreement.AnnotationTask(
        data=annotations_data_acro_out, distance=masi_distance
    )

    task_exp_out = agreement.AnnotationTask(
        data=annotations_data_exp_out, distance=masi_distance
    )

    task_acro_exp_out = agreement.AnnotationTask(
        data=annotations_data_acro_exp_out, distance=masi_distance
    )

    logger.info(
        "Krippendorff’s alpha for Acronyms with MISA: %f", task_acro_out.alpha()
    )
    logger.info("Cohen's Kappa for Acronyms with MISA: %f", task_acro_out.kappa())
    logger.info(
        "Krippendorff’s alpha for Expansions with MISA: %f", task_exp_out.alpha()
    )
    logger.info("Cohen's Kappa for Expansions with MISA: %f", task_exp_out.kappa())
    logger.info(
        "Krippendorff’s alpha for Pairs with MISA: %f", task_acro_exp_out.alpha()
    )
    logger.info("Cohen's Kappa for Pairs with MISA: %f", task_acro_exp_out.kappa())


def calculate_alpha_kappa(user_annotations_raw_file):
    """Logs the Krippendorff’s alpha and Cohen's kappa with Jaccard and Masi distance for users annotations against GOLD.

    Logs the the Krippendorff’s alpha and Cohen's kappa for all annotations as well as for
    Acronym Identification and Acronym Disambiguation with Jaccard/Masi distance
    for users annotations against GOLD considering the acronyms, expansions and acronym/expansion pairs.

    Args:
        user_annotations_raw_file (str):
        csv file with the two user annotations for each wikipedia article. The annotations
        should be separated in acronyms with/without expansion in text. This should
        be present in the data folder for user wikipedia.
    """

    article_id_to_acronym_expansions = pickle.load(
        open(getArticleAcronymDBPath(USERS_WIKIPEDIA), "rb")
    )

    test_articles = pickle.load(
        open(getDatasetGeneratedFilesPath(USERS_WIKIPEDIA) + "articles.pickle", "rb")
    )

    annotations_data_acro_in = []
    annotations_data_exp_in = []
    annotations_data_acro_exp_in = []

    annotations_data_acro_out = []
    annotations_data_exp_out = []
    annotations_data_acro_exp_out = []

    annotations_data_acro_total = []
    annotations_data_exp_total = []
    annotations_data_acro_exp_total = []

    path = getDatasetPath(USERS_WIKIPEDIA)

    with open(path + user_annotations_raw_file) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(csvreader)
        for row in csvreader:

            wiki_id = str(row[2])

            if wiki_id not in test_articles:
                continue

            # we are using user's 2 annotations as they are better overall

            annotations_u2_in = {
                k.lower(): v for k, v in dict(processAnnotations(row[5])).items()
            }

            annotations_u2_out = {
                k.lower(): v for k, v in dict(processAnnotations(row[6])).items()
            }

            annotations_u2_total = {**annotations_u2_in, **annotations_u2_out}

            gold_in = {
                k.lower(): v[0]
                for k, v in article_id_to_acronym_expansions[row[2]].items()
                if v[1]
            }

            gold_out = {
                k.lower(): v[0]
                for k, v in article_id_to_acronym_expansions[row[2]].items()
                if not v[1]
            }

            gold_total = {
                k.lower(): v[0]
                for k, v in article_id_to_acronym_expansions[row[2]].items()
            }

            if len(annotations_u2_in) != 0 and len(gold_in) != 0:
                annotations_data_acro_in.append(
                    (("u1"), int(row[0]), frozenset(annotations_u2_in.keys()))
                )
                annotations_data_acro_in.append(
                    (("u2"), int(row[0]), frozenset(gold_in.keys()))
                )
                annotations_data_exp_in.append(
                    (("u1"), int(row[0]), frozenset(annotations_u2_in.values()))
                )
                annotations_data_exp_in.append(
                    (("u2"), int(row[0]), frozenset(gold_in.values()))
                )
                annotations_data_acro_exp_in.append(
                    (("u1"), int(row[0]), frozenset(annotations_u2_in.items()))
                )
                annotations_data_acro_exp_in.append(
                    (("u2"), int(row[0]), frozenset(gold_in.items()))
                )

            if len(annotations_u2_out) != 0 and len(gold_out) != 0:
                annotations_data_acro_out.append(
                    (("u1"), int(row[0]), frozenset(annotations_u2_out.keys()))
                )
                annotations_data_acro_out.append(
                    (("u2"), int(row[0]), frozenset(gold_out.keys()))
                )
                annotations_data_exp_out.append(
                    (("u1"), int(row[0]), frozenset(annotations_u2_out.values()))
                )
                annotations_data_exp_out.append(
                    (("u2"), int(row[0]), frozenset(gold_out.values()))
                )
                annotations_data_acro_exp_out.append(
                    (("u1"), int(row[0]), frozenset(annotations_u2_out.items()))
                )
                annotations_data_acro_exp_out.append(
                    (("u2"), int(row[0]), frozenset(gold_out.items()))
                )

            if len(annotations_u2_total) != 0 and len(gold_total) != 0:
                annotations_data_acro_total.append(
                    (("u1"), int(row[0]), frozenset(annotations_u2_total.keys()))
                )
                annotations_data_acro_total.append(
                    (("u2"), int(row[0]), frozenset(gold_total.keys()))
                )
                annotations_data_exp_total.append(
                    (("u1"), int(row[0]), frozenset(annotations_u2_total.values()))
                )
                annotations_data_exp_total.append(
                    (("u2"), int(row[0]), frozenset(gold_total.values()))
                )
                annotations_data_acro_exp_total.append(
                    (("u1"), int(row[0]), frozenset(annotations_u2_total.items()))
                )
                annotations_data_acro_exp_total.append(
                    (("u2"), int(row[0]), frozenset(gold_total.items()))
                )

    logger.info("Krippendorff’s alpha and Cohen's Kappa: Human vs Gold")

    task_acro_total = agreement.AnnotationTask(
        data=annotations_data_acro_total, distance=jaccard_distance
    )

    task_exp_total = agreement.AnnotationTask(
        data=annotations_data_exp_total, distance=jaccard_distance
    )

    task_acro_exp_total = agreement.AnnotationTask(
        data=annotations_data_acro_exp_total, distance=jaccard_distance
    )

    logger.info(
        "Krippendorff’s alpha for Acronyms with Jaccard: %f", task_acro_total.alpha()
    )
    logger.info("Cohen's Kappa for Acronyms with Jaccard: %f", task_acro_total.kappa())
    logger.info(
        "Krippendorff’s alpha for Expansions with Jaccard: %f", task_exp_total.alpha()
    )
    logger.info("Cohen's Kappa for Expansions with Jaccard: %f", task_exp_total.kappa())
    logger.info(
        "Krippendorff’s alpha for Pairs with Jaccard: %f", task_acro_exp_total.alpha()
    )
    logger.info("Cohen's Kappa for Pairs with Jaccard: %f", task_acro_exp_total.kappa())

    task_acro_total = agreement.AnnotationTask(
        data=annotations_data_acro_total, distance=masi_distance
    )

    task_exp_total = agreement.AnnotationTask(
        data=annotations_data_exp_total, distance=masi_distance
    )

    task_acro_exp_total = agreement.AnnotationTask(
        data=annotations_data_acro_exp_total, distance=masi_distance
    )

    logger.info(
        "Krippendorff’s alpha for Acronyms with MISA: %f", task_acro_total.alpha()
    )
    logger.info("Cohen's Kappa for Acronyms with MISA: %f", task_acro_total.kappa())
    logger.info(
        "Krippendorff’s alpha for Expansions with MISA: %f", task_exp_total.alpha()
    )
    logger.info("Cohen's Kappa for Expansions with MISA: %f", task_exp_total.kappa())
    logger.info(
        "Krippendorff’s alpha for Pairs with MISA: %f", task_acro_exp_total.alpha()
    )
    logger.info("Cohen's Kappa for Pairs with MISA: %f", task_acro_exp_total.kappa())

    logger.info(
        "Krippendorff’s alpha and Cohen's Kappa for Acronym Identification: Human vs Gold"
    )

    task_acro_in = agreement.AnnotationTask(
        data=annotations_data_acro_in, distance=jaccard_distance
    )

    task_exp_in = agreement.AnnotationTask(
        data=annotations_data_exp_in, distance=jaccard_distance
    )

    task_acro_exp_in = agreement.AnnotationTask(
        data=annotations_data_acro_exp_in, distance=jaccard_distance
    )

    logger.info(
        "Krippendorff’s alpha for Acronyms with Jaccard: %f", task_acro_in.alpha()
    )
    logger.info("Cohen's Kappa for Acronyms with Jaccard: %f", task_acro_in.kappa())
    logger.info(
        "Krippendorff’s alpha for Expansions with Jaccard: %f", task_exp_in.alpha()
    )
    logger.info("Cohen's Kappa for Expansions with Jaccard: %f", task_exp_in.kappa())
    logger.info(
        "Krippendorff’s alpha for Pairs with Jaccard: %f", task_acro_exp_in.alpha()
    )
    logger.info("Cohen's Kappa for Pairs with Jaccard: %f", task_acro_exp_in.kappa())

    task_acro_in = agreement.AnnotationTask(
        data=annotations_data_acro_in, distance=masi_distance
    )

    task_exp_in = agreement.AnnotationTask(
        data=annotations_data_exp_in, distance=masi_distance
    )

    task_acro_exp_in = agreement.AnnotationTask(
        data=annotations_data_acro_exp_in, distance=masi_distance
    )

    logger.info("Krippendorff’s alpha for Acronyms with MISA: %f", task_acro_in.alpha())
    logger.info("Cohen's Kappa for Acronyms with MISA: %f", task_acro_in.kappa())
    logger.info(
        "Krippendorff’s alpha for Expansions with MISA: %f", task_exp_in.alpha()
    )
    logger.info("Cohen's Kappa for Expansions with MISA: %f", task_exp_in.kappa())
    logger.info(
        "Krippendorff’s alpha for Pairs with MISA: %f", task_acro_exp_in.alpha()
    )
    logger.info("Cohen's Kappa for Pairs with MISA: %f", task_acro_exp_in.kappa())

    logger.info(
        "Krippendorff’s alpha and Cohen's Kappa for Acronym Disambiguation: Human vs Gold"
    )

    task_acro_out = agreement.AnnotationTask(
        data=annotations_data_acro_out, distance=jaccard_distance
    )

    task_exp_out = agreement.AnnotationTask(
        data=annotations_data_exp_out, distance=jaccard_distance
    )

    task_acro_exp_out = agreement.AnnotationTask(
        data=annotations_data_acro_exp_out, distance=jaccard_distance
    )

    logger.info(
        "Krippendorff’s alpha for Acronyms with Jaccard: %f", task_acro_out.alpha()
    )
    logger.info("Cohen's Kappa for Acronyms with Jaccard: %f", task_acro_out.kappa())
    logger.info(
        "Krippendorff’s alpha for Expansions with Jaccard: %f", task_exp_out.alpha()
    )
    logger.info("Cohen's Kappa for Expansions with Jaccard: %f", task_exp_out.kappa())
    logger.info(
        "Krippendorff’s alpha for Pairs with Jaccard: %f", task_acro_exp_out.alpha()
    )
    logger.info("Cohen's Kappa for Pairs with Jaccard: %f", task_acro_exp_out.kappa())

    task_acro_out = agreement.AnnotationTask(
        data=annotations_data_acro_out, distance=masi_distance
    )

    task_exp_out = agreement.AnnotationTask(
        data=annotations_data_exp_out, distance=masi_distance
    )

    task_acro_exp_out = agreement.AnnotationTask(
        data=annotations_data_acro_exp_out, distance=masi_distance
    )

    logger.info(
        "Krippendorff’s alpha for Acronyms with MISA: %f", task_acro_out.alpha()
    )
    logger.info("Cohen's Kappa for Acronyms with MISA: %f", task_acro_out.kappa())
    logger.info(
        "Krippendorff’s alpha for Expansions with MISA: %f", task_exp_out.alpha()
    )
    logger.info("Cohen's Kappa for Expansions with MISA: %f", task_exp_out.kappa())
    logger.info(
        "Krippendorff’s alpha for Pairs with MISA: %f", task_acro_exp_out.alpha()
    )
    logger.info("Cohen's Kappa for Pairs with MISA: %f", task_acro_exp_out.kappa())


def calculate_user_dict_level_performance(user_annotations_raw_file):
    """Calculate performance metrics for the user annotations of Wiki articles on a dictionary level

    Logs precision, recall and f1-score for acronyms and acronym/expansion pairs for acronynms that have
    expansion in text. Only unique acronyms are considered.

    Args:
        user_annotations_raw_file (str):
         csv file with the two user annotations for each wikipedia article. This should
         be present in the data folder for users wikipedia
    """
    path = getDatasetPath(USERS_WIKIPEDIA)

    acronym_db = pickle.load(open(get_acronym_db_path(USERS_WIKIPEDIA), "rb"))

    test_articles = pickle.load(
        open(getDatasetGeneratedFilesPath(USERS_WIKIPEDIA) + "articles.pickle", "rb")
    )

    acronym_db_in = {}

    # extracting expansions that are in text
    for acro, expansions in acronym_db.items():
        expansions_in = []
        for expansion in expansions:
            if expansion[1] not in test_articles:
                continue
            if expansion[2] == True:
                expansions_in.append((expansion[0], expansion[1]))

        if len(expansions_in) > 0:
            acronym_db_in[acro] = expansions_in

    acronym_db_in_clean = {}

    # substituting similar expansions with the same expansion
    _resolve_exp_acronym_db(acronym_db_in, acronym_db_in_clean)

    acronym_db_in_clean = {k.lower(): v for k, v in acronym_db_in_clean.items()}

    # removing duplicate expansions
    for acro, expansions in acronym_db_in_clean.items():
        expansions_in = []
        for expansion in expansions:
            expansions_in.append(expansion[0])

        expansions_in = list(dict.fromkeys(expansions_in))
        acronym_db_in_clean[acro] = expansions_in

    user_annotations_in = {}

    with open(path + user_annotations_raw_file) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(csvreader)
        for row in csvreader:

            article_id = row[2]

            if article_id not in test_articles:
                continue

            annotator_1_in = dict(processAnnotations(row[3]))

            annotator_2_in = dict(processAnnotations(row[5]))

            for acro, user_expansion in annotator_1_in.items():
                if acro in user_annotations_in.keys():
                    user_annotations_in[acro].append((user_expansion, article_id))
                else:
                    user_annotations_in[acro] = [(user_expansion, article_id)]

            for acro, user_expansion in annotator_2_in.items():
                if acro in user_annotations_in.keys():
                    user_annotations_in[acro].append((user_expansion, article_id))
                else:
                    user_annotations_in[acro] = [(user_expansion, article_id)]

    user_annotations_in_clean = {}

    _resolve_exp_acronym_db(user_annotations_in, user_annotations_in_clean)

    user_annotations_in_clean = {
        k.lower(): v for k, v in user_annotations_in_clean.items()
    }

    for acro, expansions in user_annotations_in_clean.items():
        expansions_in = []
        for expansion in expansions:
            expansions_in.append(expansion[0])

        expansions_in = list(dict.fromkeys(expansions_in))
        user_annotations_in_clean[acro] = expansions_in

    results = {
        "Acronyms": {"correct": 0, "extracted": 0, "gold": 0},
        "Pairs": {"correct": 0, "extracted": 0, "gold": 0},
    }

    for db_acro, db_exps in acronym_db_in_clean.items():
        results["Acronyms"]["gold"] += len(db_exps)
        results["Pairs"]["gold"] += len(db_exps)

        user_expansions = user_annotations_in_clean.pop(db_acro, None)
        if user_expansions is None and len(db_acro) > 1 and db_acro[-1] == "s":
            user_expansions = user_annotations_in_clean.pop(db_acro[:-1], None)
        elif user_expansions is None and db_acro[-1] != "s":
            user_expansions = user_annotations_in_clean.pop(db_acro + "s", None)

        if user_expansions != None:
            results["Acronyms"]["extracted"] += len(user_expansions)
            results["Pairs"]["extracted"] += len(user_expansions)
            if len(db_exps) - len(user_expansions) > 0:
                results["Acronyms"]["correct"] += len(user_expansions)
            else:
                results["Acronyms"]["correct"] += len(db_exps)

            for usr_exp in user_expansions:
                for exp in db_exps:
                    if AcronymExpansion.areExpansionsSimilar(usr_exp, exp):
                        results["Pairs"]["correct"] += 1
                        break

    for acro, exp in user_annotations_in_clean.items():
        results["Acronyms"]["extracted"] += len(exp)
        results["Pairs"]["extracted"] += len(exp)

    logger.info("Metrics for Acronym Identification on Dictionary Level")

    logger.info("Considering only unique Pairs of Acronym/Expansion")

    precision_acronym = (
        results["Acronyms"]["correct"] / results["Acronyms"]["extracted"]
    )
    recall_acronym = results["Acronyms"]["correct"] / results["Acronyms"]["gold"]
    f1_score_acronym = 2 * (
        (precision_acronym * recall_acronym) / (precision_acronym + recall_acronym)
    )

    logger.info("Precision for Acronyms: %f", precision_acronym)
    logger.info("Recall for Acronyms: %f", recall_acronym)
    logger.info("F1-Score for Acronyms: %f", f1_score_acronym)

    precision_pair = results["Pairs"]["correct"] / results["Pairs"]["extracted"]
    recall_pair = results["Pairs"]["correct"] / results["Pairs"]["gold"]
    f1_score_pair = 2 * (
        (precision_pair * recall_pair) / (precision_pair + recall_pair)
    )

    logger.info("Precision for Pairs: %f", precision_pair)
    logger.info("Recall for Pairs: %f", recall_pair)
    logger.info("F1-Score for Pairs: %f", f1_score_pair)


if __name__ == "__main__":

    evaluate_user_responses(FILE_USER_WIKIPEDIA_ANNOTATIONS_RAW)
    # calculate_iaa(FILE_USER_WIKIPEDIA_ANNOTATIONS_RAW)
    # calculate_alpha_kappa(FILE_USER_WIKIPEDIA_ANNOTATIONS_RAW)
    # calculate_user_dict_level_performance(FILE_USER_WIKIPEDIA_ANNOTATIONS_RAW)
