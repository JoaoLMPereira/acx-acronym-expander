"""
Created on Jun 26, 2019

@author: jpereira
"""
import os
import pickle
import random

import bioc
from DataCreators.ArticleAcronymDB import create_article_acronym_db_from_acronym_db
from helper import (
    get_acronym_db_path,
    get_raw_article_db_path,
    getArticleAcronymDBPath,
    getDatasetGeneratedFilesPath,
    getDatasetPath,
)
from Logger import logging

logger = logging.getLogger(__name__)


def _addExpToDB(acronymDB, docId, acronym, expansion):
    expList = acronymDB.setdefault(acronym, [])
    expList.append((expansion, docId))


def processPassage(passage, docId, acronymDB):
    acronyms = {}
    expansions = {}
    for annotation in passage.annotations:
        annotationType = annotation.infons["type"]
        if annotationType != "ABBR":
            logger.warn("Unknown annotation type: " + annotationType)
            break

        annotationABBR = annotation.infons["ABBR"]
        if annotationABBR == "LongForm":
            expansions[annotation.id] = annotation.text
        elif annotationABBR == "ShortForm":
            acronyms[annotation.id] = annotation.text
        else:
            logger.warn("Unknown annotation ABBR value: " + annotationABBR)

    for relation in passage.relations:
        relationType = relation.infons["type"]
        if relationType != "ABBR":
            logger.warn("Unknown relation type: " + relationType)
            break

        nodes = relation.nodes
        if len(nodes) != 2:
            logger.error("Invalid number of nodes: " + len(nodes) + " , expected 2.")
            break

        for node in nodes:
            nodeRef = node.refid
            if node.role == "LongForm":
                expansionId = nodeRef
            elif node.role == "ShortForm":
                acronymId = nodeRef
            else:
                logger.error("Unknown node role: " + node.role)
                break

        expansion = expansions[expansionId]
        acronym = acronyms[acronymId]

        _addExpToDB(acronymDB, docId, acronym, expansion)


def _getGoldFile(datasetPath):
    for filename in os.listdir(datasetPath):
        if filename.endswith("_gold.xml"):
            return datasetPath + filename


def _create_article_and_acronym_db(datasetName):
    datasetPath = getDatasetPath(datasetName)
    filename = _getGoldFile(datasetPath)
    acronymDB = {}
    articleDB = {}
    article_ids = []

    reader = bioc.BioCXMLDocumentReader(filename)
    collection_info = reader.get_collection_info()
    for document in reader:
        docId = document.id
        docText = None
        for passage in document.passages:
            passageType = passage.infons["type"]
            if passageType == "title" or passageType == "abstract":
                processPassage(passage, docId, acronymDB)
                if docText:
                    docText += "\n" + passage.text
                else:
                    docText = passage.text
            elif passageType != "affiliation":
                logger.warn("Unknown passage type: " + passageType)
        articleDB[docId] = docText
        article_ids.append(docId)

    random.seed(5)
    random.shuffle(article_ids)
    split_article_id = int(len(article_ids) * 0.7)
    train_articles = article_ids[:split_article_id]
    test_articles = article_ids[split_article_id:]

    return acronymDB, articleDB, train_articles, test_articles


def make_dbs(datasetName):
    (
        acronymDB,
        articleDB,
        train_articles,
        test_articles,
    ) = _create_article_and_acronym_db(datasetName)

    articleIDToAcronymExpansions = create_article_acronym_db_from_acronym_db(acronymDB)

    pickle.dump(articleDB, open(get_raw_article_db_path(datasetName), "wb"), protocol=2)
    pickle.dump(acronymDB, open(get_acronym_db_path(datasetName), "wb"), protocol=2)
    pickle.dump(
        articleIDToAcronymExpansions,
        open(getArticleAcronymDBPath(datasetName), "wb"),
        protocol=2,
    )
    pickle.dump(
        train_articles,
        open(getDatasetGeneratedFilesPath(datasetName) + "train_articles.pickle", "wb"),
        protocol=2,
    )
    pickle.dump(
        test_articles,
        open(getDatasetGeneratedFilesPath(datasetName) + "test_articles.pickle", "wb"),
        protocol=2,
    )


def main():
    make_dbs("SH-BioC")
    make_dbs("Ab3P-BioC")
    make_dbs("BioADI-BioC")
    make_dbs("MEDSTRACT")


if __name__ == "__main__":
    main()
