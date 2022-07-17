import csv
import os
import time
import re
import urllib
import datetime
from bs4 import BeautifulSoup
import requests

from Logger import logging
from string_constants import file_ScienceWise_index_train,\
    folder_scienceWise_pdfs, folder_scienceWise_abstracts,\
    file_ScienceWise_index_train_noabbr, file_ScienceWise_index_test

logger = logging.getLogger(__name__)


def downloadPdfs():
    with open(file_ScienceWise_index_train, "r") as file:
        reader = csv.DictReader(file, delimiter=",")
        for line in reader:
            pdfID = line["ARXIV_ID"]
            filename = arxivIDToFilename(pdfID) + ".pdf"
            try:
                if(os.path.exists(folder_scienceWise_pdfs + filename)):
                    logger.debug("present already " + pdfID)
                    continue
                _downloadPdf(pdfID)
                logger.debug("successfully downloaded " + pdfID)
                time.sleep(5 * 60)
            except:
                logger.exception("Error in file " + pdfID)


def arxivIDToFilename(arxivID):
    filename = arxivID.replace("/", "_").replace("\\", "_")
    return filename


def _downloadPdf(pdfID):
    url = "http://arxiv.org/pdf/" + pdfID + ".pdf"
    response = urllib.request.urlopen(url)

    filename = arxivIDToFilename(pdfID) + ".pdf"
    local_file = open(folder_scienceWise_pdfs + filename, "wb")
    local_file.write(response.read())

    response.close()
    local_file.close()


def visualize():
    import matplotlib.pyplot as plt

    data = {}
    with open(file_ScienceWise_index_train, "r") as file:
        reader = csv.reader(file, delimiter=",")
        for line in reader:
            acronym = line[1]
            expansion = line[-1]
            if(not acronym in data):
                data[acronym] = []
            if(not expansion in data[acronym]):
                data[acronym].append(expansion)

    logger.info("number of acronyms", len(data.keys()))

    numAmbs = []
    for key in data.keys():
        num = len(data[key]) - 1
        if(num > 0):
            numAmbs.append(num)

    logger.info(len(numAmbs))
    logger.info(max(numAmbs))

    plt.subplot(121)
    plt.title("Histogram of number of ambiguities")
    plt.grid()
    plt.yticks(range(1, 66))
    plt.hist(numAmbs)

    plt.subplot(122)
    plt.title("Plot of number of ambiguities")
    plt.plot(numAmbs)

    plt.show()


def get_doc(url):
    text = list()
    if url == "":
        logger.info(url)
        return ""
    response = requests.get(url)
    soup = BeautifulSoup(markup=response.text, features="lxml")
    if soup is None:
        logger.info(url)
        return ""
    content = soup.find(name="div", attrs={"class": "abstract mathjax"})
    if content is None:
        logger.info(url)
        return ""
    list_p = content.findAll(name="p")
    if list_p is None:
        logger.info(url)
        return ""
    for p in list_p:
        text.append(str(p.text))
    return " ".join(text)


def _storeAbstract(articleID, content):
    filename = arxivIDToFilename(articleID) + ".txt"
    with open(folder_scienceWise_abstracts + filename, "w") as local_file:
        local_file.write(content)

#1203.5519
#<id>http://arxiv.org/abs/1002.0991v1</id>
#<updated>2010-02-04T13:06:46Z</updated>
#s<published>2010-02-04T13:06:46Z</published>
referenceDateTime = datetime.datetime.strptime("2012-09-30T23:59:59Z", '%Y-%m-%dT%H:%M:%SZ')
def _downloadAbstract(articleID):
    url = "http://export.arxiv.org/api/query?id_list=" + articleID
    #response = urllib.request.urlopen(url)
    logger.debug("Fetching: "+url)
    with requests.get(url) as response:
        soup = BeautifulSoup(markup=response.text, features="lxml")
        
    if soup is None:
        logger.error(url)
        return
    
    entryList = soup.find_all(name="entry")
    if(len(entryList) != 1):
        logger.warn("Expected one article, found "+ len(len(entryList))+" for article " + articleID)
        return
    
    #soup.entry.updated soup.entry.id.contents[0][-1]
    version = int(soup.entry.id.contents[0].rpartition("v")[-1])
    updatedString = soup.entry.updated.contents[0]
    # 2010-02-04T13:06:46Z
    updatedDateTime = datetime.datetime.strptime(updatedString, '%Y-%m-%dT%H:%M:%SZ')
    if updatedDateTime < referenceDateTime:
        content = re.sub('\\n',' ',soup.entry.summary.contents[0]).strip()
        _storeAbstract(articleID, content)
        return
    else:
        for v in range(version, 0, -1):
            urlWithVersion = url + 'v' +str(v)
            logger.info("Fetching: " + urlWithVersion)
            with requests.get(urlWithVersion) as response:
                soup = BeautifulSoup(markup=response.text, features="lxml")
            if soup is None:
                logger.error(url)
                return
            updatedString = soup.entry.updated.contents[0]
            updatedDateTime = datetime.datetime.strptime(updatedString, '%Y-%m-%dT%H:%M:%SZ')
            if updatedDateTime < referenceDateTime:
                content = re.sub('\\n',' ',soup.entry.summary.contents[0]).strip()
                _storeAbstract(articleID, content)
                return

def downloadAbstracts(file_ScienceWise):
    logger.info("Downloading abstract of articles from " + file_ScienceWise)
    with open(file_ScienceWise, "r") as file:
        reader = csv.DictReader(file, delimiter=",", fieldnames=["ARXIV_ID", "ACRONYM", "ID", "DEFINITION"])
        for line in reader:
            articleID = line["ARXIV_ID"]
            filename = arxivIDToFilename(articleID) + ".txt"

            try:
                if(os.path.exists(folder_scienceWise_abstracts + filename)):
                    logger.debug("present already " + articleID)
                    continue
                _downloadAbstract(articleID)
                logger.debug("successfully downloaded " + articleID)
                time.sleep(3)
            except:
                logger.exception("Error in file " + articleID)


if __name__ == "__main__":
    #visualize()
    downloadAbstracts(file_ScienceWise_index_train)
    downloadAbstracts(file_ScienceWise_index_train_noabbr)
    downloadAbstracts(file_ScienceWise_index_test)
