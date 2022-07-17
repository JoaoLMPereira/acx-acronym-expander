"""Creates HTML docs, processes user annotations and differs user responses

Created on Apr 22, 2019

@author: jpereira
"""

import csv
import functools
import json
import os
import pickle
import random
import shutil
from concurrent.futures import ProcessPoolExecutor

import requests
from bs4 import BeautifulSoup
from helper import (
    AcronymExpansion,
    get_acronym_db_path,
    get_raw_article_db_path,
    getArticleAcronymDBPath,
    getDatasetPath,
)
from Logger import logging
from nltk.corpus import stopwords
from nltk.metrics import agreement
from nltk.metrics.distance import jaccard_distance, masi_distance
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from string_constants import (
    DB_WITH_LINKS_SUFFIX,
    FILE_USER_WIKIPEDIA_ANNOTATIONS_RAW,
    FOLDER_DATA,
    USERS_WIKIPEDIA,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

CATEGORY_URL = "https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle={}&cmlimit=500&format=json"

ROOT_CATEGORY = "Computing"

website = "http://web.tecnico.ulisboa.pt/ist164790/acronym-user-exp/documents/"


stop_words = set(stopwords.words("english"))
p_stemmer = PorterStemmer()
tokeniser = RegexpTokenizer(r"\w+")

wikiExtractorOutputPath = FOLDER_DATA + "/FullWikipedia"
storageType = None
jsonDecoder = json.JSONDecoder()

"""
Section to create HTML plain text Wikipedia docs
"""


def getCategoryInfo(categoryName):
    pages = []
    categories = []

    url = CATEGORY_URL.format(categoryName.replace(" ", "_").replace("/", "\\"))
    response = requests.get(url)

    jsonObjc = jsonDecoder.decode(response.text)
    members = jsonObjc["query"]["categorymembers"]
    # {'pageid': 51002505, 'ns': 0, 'title': 'IEEE Rebooting Computing'}
    # {'pageid': 30519278, 'ns': 14, 'title': 'Category:Computer standards'}
    for member in members:
        title = member["title"]
        pageid = member["pageid"]
        if member["ns"] == 0:
            pages.append(title)
        elif member["ns"] == 14:
            # categories
            categories.append(title)
    return pages, categories


def getPagesFromCategory(categoryName):
    categoryName = "Category:" + categoryName
    pages, categories = getCategoryInfo(categoryName)
    i = 0
    while categories and i < 1000:
        rand = random.randint(0, len(categories) - 1)
        categoryName = categories.pop(rand)
        newPages, newCategories = getCategoryInfo(categoryName)
        pages.extend(newPages)
        categories.extend(newCategories)

        i = i + 1
        print(i)

    return pages


def storePagesFromCategory(pages, filename):
    path = getDatasetPath(USERS_WIKIPEDIA)
    with open(path + filename, "w") as f:
        for page in pages:
            f.write(page + "\n")


def getUserPageTitles():
    pages = getPagesFromCategory(ROOT_CATEGORY)

    storePagesFromCategory(pages, "pages.txt")


def processWikiFile(filePath):

    logger.debug("Processing file: " + filePath)
    with open(filePath) as file:
        soupOut = BeautifulSoup(markup=file, features="lxml")
        for doc in soupOut.findAll(name="doc"):
            attributes = doc.attrs
            docTitle = attributes["title"]

            text = doc.text

            yield docTitle, text


def processWikiFolder(startdir):
    filePathsList = []

    directories = os.listdir(startdir)
    for wikiDir in directories:
        fullPathWikiDir = os.path.join(startdir, wikiDir)
        if os.path.isdir(fullPathWikiDir):
            for file in os.listdir(fullPathWikiDir):
                filePath = os.path.join(fullPathWikiDir, file)
                filePathsList.append(filePath)

    for filePath in filePathsList:
        for docTitle, text in processWikiFile(filePath):
            yield docTitle, text


def getPagesText(pages):
    pages = set(p for p in pages)
    fullWikiPath = wikiExtractorOutputPath
    pagesText = {}

    for docTitle, text in processWikiFolder(fullWikiPath):

        try:
            pages.remove(docTitle)
        except KeyError:
            continue

        # Test tokens
        if len(tokeniser.tokenize(text)) >= 100:
            pagesText[docTitle] = text

    return pagesText


def storeWikipediaPageTexts(pages, textDict):
    pathDocuments = getDatasetPath(USERS_WIKIPEDIA) + "documents/"
    if os.path.exists(pathDocuments):
        shutil.rmtree(pathDocuments, ignore_errors=True)
    os.makedirs(pathDocuments)

    for p in pages:
        text = textDict[p]

        with open(
            pathDocuments + p.replace(" ", "_").replace("/", "\\") + ".txt", "w"
        ) as text_file:
            text_file.write(text)


def getDocuments():
    path = getDatasetPath(USERS_WIKIPEDIA)
    with open(path + "pages.txt", "r") as f:
        lines = f.read().split("\n")

    pages = {p for p in lines if p.lower().find("list") < 0}

    # pageSample = random.sample(pages)
    pageSample = random.choices(list(pages), k=1000)
    # pageSample.extend(["Andorra","Anarchism","Alkali metal"])
    pagesText = getPagesText(pageSample)

    # finalPages = random.choices(list(pagesText.keys()), k=200)
    finalPages = random.sample(list(pagesText.keys()), 200)

    storePagesFromCategory(finalPages, "finalPages.txt")

    storeWikipediaPageTexts(finalPages, pagesText)

    # Write HTML String to file.html
    # with open(path + "documents.html", "w") as file:
    #    file.write(html)


# HTML String
htmlPart1 = """
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta charset="utf-8"/>
<title>Wikipedia Documents List</title>
</head>
<body>
<table border=1>
     <tr>
       <th>Document Identifier</th>
       <th>Page Name</th>
     </tr>
     <indent>
"""

htmlPart2 = """
     </indent>
</table>
</body>
</html>
"""


def getHTMLPage():
    path = getDatasetPath(USERS_WIKIPEDIA)
    with open(path + "finalPages.txt", "r") as f:
        pages = f.read().split("\n")

    i = 1
    htmlPages = htmlPart1
    for p in pages:
        if p != "":
            pageLink = website + p.replace(" ", "_").replace("/", "%5c")

            htmlPages += """
            <tr>
             <td>{}</td>
             <td><a href="{}">{}</a></td>
           </tr>
            """.format(
                i, pageLink, p
            )
            i += 1
    htmlPages += htmlPart2

    # Write HTML String to file.html
    with open(path + "index.html", "w") as file:
        file.write(htmlPages)


"""
Methods for processing annotations
"""


def processAnnotations(text):
    annotations = set()
    for s in text.split("\n"):
        pair = s.split(";")
        if pair and len(pair) != 2:
            logger.warning("Pair discarded :" + s)
            continue
        acronym, expansion = pair
        annotations.add((acronym.strip(), expansion.strip().casefold()))
    return annotations


def getStringAnnotations(annotationsSet):
    annotationsList = []
    for (acro, exp) in annotationsSet:
        annotationsList.append(acro + ";" + exp)
    return "\n".join(annotationsList)


def processWikiFileGetIds(filePath):

    logger.debug("Processing file: " + filePath)
    with open(filePath) as file:
        soupOut = BeautifulSoup(markup=file, features="lxml")
        for doc in soupOut.findAll(name="doc"):
            attributes = doc.attrs
            docId = attributes["id"]
            # docUrl = attributes["url"]
            docTitle = attributes["title"]

            # text = doc.text

            yield docId, docTitle


def processWikiFolderGetIds(startdir):
    filePathsList = []

    directories = os.listdir(startdir)
    for wikiDir in directories:
        fullPathWikiDir = os.path.join(startdir, wikiDir)
        if os.path.isdir(fullPathWikiDir):
            for file in os.listdir(fullPathWikiDir):
                filePath = os.path.join(fullPathWikiDir, file)
                filePathsList.append(filePath)

    for filePath in filePathsList:
        for docId, docTitle in processWikiFileGetIds(filePath):
            yield docId, docTitle


def loadPageWikiIds():
    path = getDatasetPath(USERS_WIKIPEDIA)
    with open(path + "mapWikiIds.txt", "r") as f:
        lines = f.read().split("\n")

    listAux = [l.split(",") for l in lines]
    mapIdToWikiId = {int(l[0]): int(l[1]) for l in listAux if len(l) == 2}

    return mapIdToWikiId


def getFinalPages():
    path = getDatasetPath(USERS_WIKIPEDIA)

    with open(path + "finalPages.txt", "r") as f:
        pages = f.read().split("\n")

    return pages


def getPageWikiIds():
    path = getDatasetPath(USERS_WIKIPEDIA)

    pages = getFinalPages()

    pagesSet = set(p for p in pages)
    fullWikiPath = wikiExtractorOutputPath

    wikiIds = {}

    for docId, docTitle in processWikiFolderGetIds(fullWikiPath):

        try:
            pagesSet.remove(docTitle)
            indx = pages.index(docTitle)
            wikiIds[indx + 1] = docId
            print("Found " + docTitle)
        except KeyError:
            continue

    with open(path + "mapWikiIds.txt", "w") as f:
        sortKeys = list(wikiIds.keys())
        sortKeys.sort()
        for key in sortKeys:
            f.write(str(key) + "," + str(wikiIds[key]) + "\n")


def getDocText(docTitle):
    pathDocuments = getDatasetPath(USERS_WIKIPEDIA) + "documents/"
    with open(
        pathDocuments + docTitle.replace(" ", "_").replace("/", "\\") + ".txt", "r"
    ) as text_file:
        # remove utf-8 code
        return text_file.read()[1:]

    raise Exception


def get_wiki_file_path(startdir):
    filePathsList = []
    directories = os.listdir(startdir)
    filePathsList = []
    for wikiDir in directories:
        fullPathWikiDir = os.path.join(startdir, wikiDir)
        if os.path.isdir(fullPathWikiDir):
            for file in os.listdir(fullPathWikiDir):
                filePath = os.path.join(fullPathWikiDir, file)
                filePathsList.append(filePath)
    return filePathsList


def process_wiki_file(file_path, articles_to_get):
    results = []
    with open(file_path) as file:
        try:
            soupOut = BeautifulSoup(markup=file, features="lxml")
            for doc in soupOut.findAll(name="doc"):
                attributes = doc.attrs
                docId = attributes["id"]
                if int(docId) in articles_to_get:
                    results.append((docId, str(doc)))

        except:
            logger.exception("Error processing file: %s", file_path)

        return results


"""
Methods for comparing user annotations for Wikipedia articles
"""


def diff_user_responses_old():
    """NOTE: This function was used to process the first dataset version (first 67 wikipedia articles)"""
    path = getDatasetPath(USERS_WIKIPEDIA)
    userResponses = {}
    emails = set()

    with open(path + "Research Study Acronym.csv", "r") as theFile:
        reader = csv.DictReader(theFile)
        fieldNames = reader.fieldnames
        fieldEmail = fieldNames[1]
        fieldID = fieldNames[2]
        field1stDoc = fieldNames[3]
        field2ndDoc = fieldNames[4]

        for line in reader:
            userEmail = line.get(fieldEmail)
            emails.add(userEmail.strip().lower())
            userIdStr = line.get(fieldID)
            try:
                userId = int(userIdStr)
            except ValueError:
                logger.warning("Invalid user ID " + userIdStr)
                continue

            if userId > 100:
                logger.warning("Invalid user ID " + str(userId))
                continue

            firstDoc = processAnnotations(line.get(field1stDoc))
            secondDoc = processAnnotations(line.get(field2ndDoc))
            userResponses[userId] = (firstDoc, secondDoc)

    docAnnotations = dict()
    for userId, response in userResponses.items():

        firstDocId = userId
        firstDocOutput = response[0]
        firstDocAnnotations = docAnnotations.get(firstDocId, [None, None])
        firstDocAnnotations[0] = firstDocOutput
        docAnnotations[firstDocId] = firstDocAnnotations

        secondDocId = userId + 1
        secondDocOutput = response[1]
        secondDocAnnotations = docAnnotations.get(secondDocId, [None, None])
        secondDocAnnotations[1] = secondDocOutput
        docAnnotations[secondDocId] = secondDocAnnotations

    mapIdToWikiId = loadPageWikiIds()
    pages = getFinalPages()
    toDiscard = []
    with open(path + "userAnnotationsToReview.csv", "w") as f:
        csvWriter = csv.writer(f, delimiter=",", quotechar='"')
        for docId in range(1, max(docAnnotations.keys())):
            # for docId, annotations in docAnnotations.items():
            docTitle = pages[docId - 1]
            annotations = docAnnotations[docId]
            wikiId = mapIdToWikiId[docId]

            if annotations[0] != annotations[1]:
                print(docId)

                if annotations[0] == None:
                    print("None")
                    print(annotations[1])
                    annotations[0] = set()

                if annotations[1] == None:
                    print(annotations[0])
                    print("None")
                    annotations[1] = set()

            toDiscard.append(docId)

            interception = annotations[0] & annotations[1]
            left = annotations[0] - annotations[1]
            right = annotations[1] - annotations[0]

            diffAll = left | right

            csvWriter.writerow(
                [
                    str(docId),
                    docTitle,
                    str(wikiId),
                    getStringAnnotations(interception),
                    getStringAnnotations(diffAll),
                    getStringAnnotations(left),
                    getStringAnnotations(right),
                ]
            )

    toDiscard.sort()

    print(toDiscard)
    for id in toDiscard:
        docAnnotations.pop(id)

    print(docAnnotations.keys())

    print(", ".join(emails))

    for docId in range(1, max(docAnnotations.keys())):
        print(docId)


def diff_user_responses_new(annot_raw, annot_to_review):
    """Creates a csv file with the user annotations for each wiki article including interception, left and right.

    Args:
        annot_raw (str):
         csv file that has the RAW user annotations for the wikipedia articles. This file should be straight from the
         forms output and is not the raw file present in the data folder for users wikipedia.
        annot_to_review (str):
         the name of the csv file that will have the interception, left and right of the annotators for each wikipedia article.
    """
    path = getDatasetPath(USERS_WIKIPEDIA)
    user_responses = {}

    with open(path + annot_raw, "r") as file:
        annot_reader = csv.reader(file, delimiter=",", quotechar='"')
        next(annot_reader)
        for row in annot_reader:

            try:
                task_id = int(row[1])
                first_doc_id = int(row[2])
                second_doc_id = int(row[5])
            except ValueError:
                logger.warning(
                    "Invalid taskID: "
                    + task_id
                    + ", firstDocID: "
                    + first_doc_id
                    + ", secondDocID: "
                    + second_doc_id
                )

            first_doc_in = processAnnotations(row[3])
            first_doc_out = processAnnotations(row[4])
            second_doc_in = processAnnotations(row[6])
            second_doc_out = processAnnotations(row[7])

            user_responses[task_id] = [
                [first_doc_id, first_doc_in, first_doc_out],
                [second_doc_id, second_doc_in, second_doc_out],
            ]

    doc_annotations = {}
    for task_id, response in user_responses.items():

        # holds annotations for acronyms in a doc:
        # - First sublist: acronyms with expansion in text
        # - Second sublist: acronyms with expansion not in text
        annotations = doc_annotations.get(response[0][0], [[], []])
        annotations[0].append(response[0][1])
        annotations[1].append(response[0][2])
        doc_annotations[response[0][0]] = annotations

        annotations = doc_annotations.get(response[1][0], [[], []])
        annotations[0].append(response[1][1])
        annotations[1].append(response[1][2])
        doc_annotations[response[1][0]] = annotations

    map_id_to_wiki_id = loadPageWikiIds()
    pages = getFinalPages()
    with open(path + annot_to_review, "w") as f:
        csv_writer = csv.writer(f, delimiter=",", quotechar='"')
        csv_writer.writerow(
            [
                "DocID",
                "Title",
                "WikiID",
                "InterceptionIn",
                "LeftIn",
                "RightIn",
                "Reviewed Annotations In",
                "InterceptionOut",
                "LeftOut",
                "RightOut",
                "Reviewed Annotations Out",
            ]
        )

        for doc_id in range(
            min(doc_annotations.keys()), max(doc_annotations.keys()) + 1
        ):
            doc_title = pages[doc_id - 1]
            annotations = doc_annotations[doc_id]
            wiki_id = map_id_to_wiki_id[doc_id]

            interception_in = annotations[0][0] & annotations[0][1]
            interception_out = annotations[1][0] & annotations[1][1]
            left_in = annotations[0][0] - annotations[0][1]
            left_out = annotations[1][0] - annotations[1][1]
            right_in = annotations[0][1] - annotations[0][0]
            right_out = annotations[1][1] - annotations[1][0]

            csv_writer.writerow(
                [
                    str(doc_id),
                    doc_title,
                    str(wiki_id),
                    getStringAnnotations(interception_in),
                    getStringAnnotations(left_in),
                    getStringAnnotations(right_in),
                    "",
                    getStringAnnotations(interception_out),
                    getStringAnnotations(left_out),
                    getStringAnnotations(right_out),
                    "",
                ]
            )


def old_dataset_to_new_format_review(
    old_user_annotations_sep_user,
    old_user_annotations_sep_in_out,
    user_annotations_to_review,
):
    """Transforms old user dataset into the new format for review.

    Transforms the old user dataset that only has columns for all the annotations of annotator 1 and 2
    into a format where each user has a columns for acronyms that have expansion in text and not in text.
    The csv file created is only for review purposes.

    Args:
        old_user_annotations_sep_user (string): csv file that holds dataset with one column for user 1 and one for user 2 with all annotations
        old_user_annotations_sep_in_out (string):
         csv file that for each wiki article as a column for the acronyms with expansion in text and another for the acronyms with expansion not in text
        user_annotations_to_review (string): the output csv file for revision
    """
    path = getDatasetPath(USERS_WIKIPEDIA)

    csvfile_sep_user = open(path + old_user_annotations_sep_user)
    csvreader_sep_user = csv.reader(csvfile_sep_user, delimiter=",", quotechar='"')
    next(csvreader_sep_user)

    csvfile_sep_in_out = open(path + old_user_annotations_sep_in_out)
    csvreader_sep_in_out = csv.reader(csvfile_sep_in_out, delimiter=",", quotechar='"')
    next(csvreader_sep_in_out)

    csvfile_to_review = open(path + user_annotations_to_review, "w")
    csv_writer = csv.writer(csvfile_to_review, delimiter=",", quotechar='"')
    csv_writer.writerow(
        [
            "DocID",
            "Title",
            "WikiID",
            "U1 not known",
            "U1 In",
            "U1 Out",
            "U2 not known",
            "U2 In",
            "U2 Out",
        ]
    )

    for row_sep_user, row_sep_in_out in zip(csvreader_sep_user, csvreader_sep_in_out):
        doc_id = str(row_sep_user[0])
        title = str(row_sep_user[1])
        wiki_id = str(row_sep_user[2])

        annotator_1_full = dict(processAnnotations(row_sep_user[3]))
        annotator_2_full = dict(processAnnotations(row_sep_user[4]))

        annotations_in = dict(processAnnotations(row_sep_in_out[3]))
        annotations_out = dict(processAnnotations(row_sep_in_out[4]))

        annotator_1_in = {}
        annotator_1_out = {}
        annotator_1_not_known = {}

        for acro in annotator_1_full.keys():
            if acro in annotations_in.keys():
                annotator_1_in[acro] = annotator_1_full[acro]
            elif acro in annotations_out.keys():
                annotator_1_out[acro] = annotator_1_full[acro]
            else:
                annotator_1_not_known[acro] = annotator_1_full[acro]

        annotator_2_in = {}
        annotator_2_out = {}
        annotator_2_not_known = {}

        for acro in annotator_2_full.keys():
            if acro in annotations_in.keys():
                annotator_2_in[acro] = annotator_2_full[acro]
            elif acro in annotations_out.keys():
                annotator_2_out[acro] = annotator_2_full[acro]
            else:
                annotator_2_not_known[acro] = annotator_2_full[acro]

        csv_writer.writerow(
            [
                doc_id,
                title,
                wiki_id,
                getStringAnnotations(set(annotator_1_not_known.items())),
                getStringAnnotations(set(annotator_1_in.items())),
                getStringAnnotations(set(annotator_1_out.items())),
                getStringAnnotations(set(annotator_2_not_known.items())),
                getStringAnnotations(set(annotator_2_in.items())),
                getStringAnnotations(set(annotator_2_out.items())),
            ]
        )


if __name__ == "__main__":

    # getUserPageTitles()
    # getDocuments()
    # execute this for all documents txt sed -i '1s/^\(\xef\xbb\xbf\)\?/\xef\xbb\xbf/' *
    # getHTMLPage()
    # getPageWikiIds()

    # diff_user_responses_new("annotations_raw.csv", "annotationsToReview.csv")

    """old_dataset_to_new_format_review(
        "complete_user_annotations_raw.csv",
        "old_user_annotations.csv",
        "old_annotations_to_review.csv",
    )"""
