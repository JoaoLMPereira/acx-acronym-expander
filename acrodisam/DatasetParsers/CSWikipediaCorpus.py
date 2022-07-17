"""
Parser for dataset used in:
Paper Acronym Disambiguation: A Domain Independent Approach
"""
import json
import pickle

from DataCreators.ArticleAcronymDB import create_article_acronym_db_from_acronym_db
from DataCreators.ArticleDB import get_preprocessed_article_db
from Logger import logging
from helper import (
    getDatasetGeneratedFilesPath,
    get_acronym_db_path,
    getArticleAcronymDBPath,
    getCrossValidationFolds,
    getTrainTestData,
    get_raw_article_db_path,
    get_preprocessed_article_db_path,
)
from string_constants import (
    folder_cs_wikipedia_generated,
    folder_cs_wikipedia_corpus,
    CS_WIKIPEDIA_DATASET,
)
from text_preparation import (
    full_text_preprocessing,
    transform_text_with_exp_tokens,
    stop_words,
)


logger = logging.getLogger(__name__)


def _create_article_and_acronym_db():
    articleDB = {}
    acronymDB = {}
    trainArticles = []
    testArticles = []
    filename = folder_cs_wikipedia_generated + "cs_wikipedia.json"
    f = open(filename, "r")
    data = json.load(f)

    abbvFilename = folder_cs_wikipedia_corpus + "list_of_abbr.txt"
    abbvFile = open(abbvFilename, "r")
    abbvSet = set(r[:-1].upper() for r in abbvFile.readlines())
    abbvFile.close()

    n = 0
    for acronym, v in data.items():

        if acronym.upper() not in abbvSet:
            logger.info("No acronym: " + acronym + " found in abbv list")
            continue

        n += 1
        logger.info(str(n) + "th short form processed")
        logger.info("Processing acronym: " + str(acronym))

        full_form = v["full_form"]
        pmid = str(v["link"])

        if full_form.lower() in stop_words:
            logger.warning(
                "Skipped stop word expansion %s for acronym %s found in article %s",
                full_form,
                acronym,
                pmid,
            )
        else:
            if pmid in articleDB:
                rawText = articleDB[pmid]
            else:
                rawText = str(v["content"])

            text, success, _ = transform_text_with_exp_tokens(
                acronym, full_form, rawText
            )

            if not success:
                logger.warn(
                    "No acronym: "
                    + acronym
                    + " nor expansion: "
                    + full_form
                    + " found in: "
                    + pmid
                )
            else:
                tmpAcronymDB = []
                tmpAcronymDB.append([full_form, pmid])
                # articleDB[pmid] = text
                tmpArticleDB = {}
                tmpArticleDB[pmid] = text

        for poss in v["possibilities"]:
            pmid = str(poss["link"])

            # if str(pmid).lower().endswith("LTE_(telecommunication)".lower()):
            #    print("here")

            if pmid in tmpArticleDB:
                logger.warn(
                    "Duplicate article found! url: " + pmid + " Acronym: " + acronym
                )
                continue

            full_form = poss["full_form"]

            if full_form.lower() in stop_words:
                logger.warning(
                    "Skipped stop word expansion %s for acronym %s found in article %s",
                    full_form,
                    acronym,
                    pmid,
                )

            if pmid in articleDB:
                rawText = articleDB[pmid]
            else:
                rawText = poss["content"]

            text, success, _ = transform_text_with_exp_tokens(
                acronym, full_form, rawText
            )

            if not success:
                logger.warn(
                    "No acronym: "
                    + acronym
                    + " nor expansion: "
                    + full_form
                    + " found in: "
                    + pmid
                )
                continue

            tmpArticleDB[pmid] = text
            tmpAcronymDB.append([full_form, pmid])

        # IF we find no alternative full_form then we have to discard this
        if len(tmpAcronymDB) > 1:
            acronymDB[acronym] = tmpAcronymDB
            trainArticles.append(tmpAcronymDB[0][1])
            testArticles += [acro[1] for acro in tmpAcronymDB[1:]]

            # merges dicts
            articleDB = {**articleDB, **tmpArticleDB}

        else:
            logger.debug("Discarded acronym: " + str(acronym))

    return acronymDB, articleDB, trainArticles, testArticles


def make_dbs(createFolds=True):
    foldsNum = 5

    acronymDB, articleDB, trainArticles, testArticles = _create_article_and_acronym_db()

    pickle.dump(
        articleDB, open(get_raw_article_db_path(CS_WIKIPEDIA_DATASET), "wb"), protocol=2
    )
    # removed acronymDB = applyManualCorrections(acronymDB)

    articleIDToAcronymExpansions = create_article_acronym_db_from_acronym_db(acronymDB)

    pickle.dump(
        acronymDB, open(get_acronym_db_path(CS_WIKIPEDIA_DATASET), "wb"), protocol=2
    )
    pickle.dump(
        articleIDToAcronymExpansions,
        open(getArticleAcronymDBPath(CS_WIKIPEDIA_DATASET), "wb"),
        protocol=2,
    )

    generatedFilesFolder = getDatasetGeneratedFilesPath(CS_WIKIPEDIA_DATASET)

    if createFolds:
        pickle.dump(
            trainArticles,
            open(generatedFilesFolder + "train_articles.pickle", "wb"),
            protocol=2,
        )
        pickle.dump(
            testArticles,
            open(generatedFilesFolder + "test_articles.pickle", "wb"),
            protocol=2,
        )

        newFolds = getCrossValidationFolds(trainArticles, foldsNum)

        foldsFilePath = (
            generatedFilesFolder + str(foldsNum) + "-cross-validation.pickle"
        )
        pickle.dump(newFolds, open(foldsFilePath, "wb"), protocol=2)

        # New train, test and folds
        train_ids, test_ids = getTrainTestData(articleDB.keys(), 0.70)
        pickle.dump(
            train_ids,
            open(generatedFilesFolder + "train_articles_new.pickle", "wb"),
            protocol=2,
        )
        pickle.dump(
            test_ids,
            open(generatedFilesFolder + "test_articles_new.pickle", "wb"),
            protocol=2,
        )

        newFolds = getCrossValidationFolds(train_ids, foldsNum)

        foldsFilePath = (
            generatedFilesFolder + str(foldsNum) + "-cross-validation_new.pickle"
        )
        pickle.dump(newFolds, open(foldsFilePath, "wb"), protocol=2)
    else:
        train_ids = pickle.load(
            open(generatedFilesFolder + "train_articles_new.pickle", "rb")
        )
        test_ids = pickle.load(
            open(generatedFilesFolder + "test_articles_new.pickle", "rb")
        )

    (
        preprocessed_article_db,
        train_exec_time,
        test_avg_exec_time,
    ) = get_preprocessed_article_db(
        articleDB,
        articleIDToAcronymExpansions,
        train_ids,
        test_ids,
        full_text_preprocessing,
    )

    logger.critical(
        "Total train preprocessing execution time per test document: "
        + str(train_exec_time)
    )
    logger.critical(
        "Average preprocessing execution time per test document: "
        + str(test_avg_exec_time)
    )

    pickle.dump(
        preprocessed_article_db,
        open(get_preprocessed_article_db_path(CS_WIKIPEDIA_DATASET), "wb"),
        protocol=2,
    )


def _classToIndex(cls):
    return int(cls[1:]) - 1


def _fileNameToAcronym(fileName):
    return fileName.split("_")[0]


if __name__ == "__main__":
    make_dbs(False)
