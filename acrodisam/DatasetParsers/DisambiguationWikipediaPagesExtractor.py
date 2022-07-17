import os
import traceback
import wikipedia
from bs4 import BeautifulSoup
import requests
import json
import re
from AcronymExpanders.Expander_fromText_CSWikipedia import Expander_fromText_CSWikipedia
from string_constants import folder_cs_wikipedia_corpus, folder_cs_wikipedia_generated
DATA_FILE_PATH = folder_cs_wikipedia_corpus + "cs_wikipedia_acronyms.json"
SEARCH_URL = "https://en.wikipedia.org/w/index.php?title=Special:Search&go=Go&search="
DISAMBIGUATION_URL = "https://en.wikipedia.org/wiki/%s_(disambiguation)"


expander = Expander_fromText_CSWikipedia()
#Part of this code from Paper Acronym Disambiguation: A Domain Independent Approach


"""
def get_doc(url):
    text_all = list()
    if url == "":
        # print(url)
        return ""
    response = requests.get(url)
    soup = BeautifulSoup(markup=response.text, features="lxml")
    if soup is None:
        # print(url)
        return ""
    content = soup.find(name="div", attrs={"class": "mw-parser-output"})
    if content is None:
        # print(url)
        return ""
    list_p = content.findAll(name="p")
    if list_p is None:
        # print(url)
        return ""

    for p in list_p:
        text_all.append(str(p.text))
  
    return " ".join(text_all)
"""

def get_doc(url):
    text = list()
    if url == "":
        print(url)
        return ""
    response = requests.get(url)
    soup = BeautifulSoup(markup=response.text, features="lxml")
    if soup is None:
        print(url)
        return ""
    content = soup.find(name="div", attrs={"class": "mw-parser-output"})
    if content is None:
        print(url)
        return ""
    list_p = content.findAll(name="p")
    if list_p is None:
        print(url)
        return ""
    for p in list_p:
        text.append(str(p.text))
    return " ".join(text)


def get_docFromPage(responseText):
    text = list()

    soup = BeautifulSoup(markup=responseText, features="lxml")
    if soup is None:
       # print(url)
        return ""
    content = soup.find(name="div", attrs={"class": "mw-parser-output"})
    if content is None:
       # print(url)
        return ""
    list_p = content.findAll(name="p")
    if list_p is None:
        #print(url)
        return ""
    for p in list_p:
        text.append(str(p.text))
    return " ".join(text)

def get_pages(query):
    pages = list()
    if len(query.strip()) <= 0:
        raise ValueError

    response = requests.get(SEARCH_URL + str(query))
    soup = BeautifulSoup(markup=response.text, features="lxml")

    if soup is None:
        raise Exception

    if "search" in str(soup.title).lower():
        result_ul = soup.find(name="ul", attrs={"class": "mw-search-results"})
        results_list = result_ul.find_all("li")

        for li in results_list:
            li_div = li.find(name="div", attrs={"class": "mw-search-result-heading"})
            a = li_div.find("a")
            link = "https://en.wikipedia.org" + a["href"]
            heading = str(a.text)
            pages.append((link, heading))

        return pages
    else:
        return wikipedia.summary(query)


def get_surrounding_substring(startIndex, stopIndex, text, windowSize = 300):
    size = windowSize // 2
    textLenght = len(text)
    minIndex = 0
    maxIndex = textLenght
    if startIndex - size > 0:
        minIndex = startIndex - size
    
    if stopIndex + size < textLenght:
        maxIndex = stopIndex + size
    
    return text[minIndex:startIndex] + " " + text[stopIndex:maxIndex]

def get_expansion_text(text, acronym):
    pattern =re.compile(r'\b'+acronym.strip()+r'[s]?\b', re.MULTILINE | re.IGNORECASE)
    for match in re.finditer(pattern, text):
        try:
            subString = get_surrounding_substring(match.start(), match.end(), text)
        except Exception as e:
            print("type error: " + str(e))
            print(traceback.format_exc())
        expansion = expander._expandInText(subString, acronym)
        if expansion:
            return expansion

def get_acronyms(query, csArticle):
    possibilities = list()
    extractedLinks = [csArticle["link"].strip().lower()]
    extractedDocs = [csArticle["content"].strip().lower()]

    response = requests.get(DISAMBIGUATION_URL % str(query))
    print(DISAMBIGUATION_URL % str(query))

    query = query.lower()
    if response.status_code != 404:
        print("Disambiguation Page Exists :D")
        soup = BeautifulSoup(markup=response.text, features="lxml")
        if soup is None:
            return None
        div = soup.find("div", attrs={"class": "mw-parser-output"})
        all_uls = div.findAll("ul")

        for ul in all_uls:
            all_lis = ul.findAll("li")
            for li in all_lis:
                a = li.find("a")
                if a is None or a["href"][0] == '#':
                    continue
                url = "https://en.wikipedia.org" + a["href"]
                #print(li)
                # TODO verificar repeticoes artigos etc...
                if not url.strip().lower() in extractedLinks:
                    content = str(get_doc(url))#.lower()
                    if not content.strip().lower() in extractedDocs:
                        exp = expander._expandInText(li.text.split(',')[0], query)
                        #exp = expander._expandInText(a.text, query)
                        if exp == None:
                            exp = expander._expandInText(li.text, query)
                            if exp == None:
                                print("No expansion found for "+query+" in: " + li.text)
                                continue
                       
                        extractedLinks.append(url.strip().lower())
                        extractedDocs.append(content.strip().lower())
                        print("Found: " + exp + " in: "+ li.text)
                        possibilities.append({"full_form": exp,
                                              "content" : content,
                                              "link" : url})
        
                    else:
                        print("Removed duplicate doc for: " + str(query))
                else:
                    print("Removed duplicate link for: " + str(url))


    results = wikipedia.search(query=query, results=10)
    print(results)
    if len(results) <= 0:
        return possibilities

    for each_result in results:
        try:
            url = wikipedia.page(each_result).url#.lower()
            if not url.strip().lower() in extractedLinks:
                content = str(get_doc(url))#.lower()
                if not content.strip().lower() in extractedDocs:
                    
                    #TODO procurar expansion no each_result, se não procurar os acronimos no texto e a expansão mais valida até um limite? Escolher a expansao mais proxima de um acronimo
                    # Procurar acronym separado por qualquer simbolo que não seja alfanumerico?
                    exp = expander._expandInText(each_result, query)
                    if exp == None:
                        
                        exp = get_expansion_text(content, query)
                        if exp == None:
                            print("No expansion found for "+query+" in: " + each_result)
                            continue
                            
                    extractedLinks.append(url.strip().lower())
                    extractedDocs.append(content.strip().lower())
                    possibilities.append({"full_form": exp,
                                            "content" : content,
                                            "link" : url})
                    print("Found: " + exp + " in: "+ each_result)# + " or in: " + content)

                else:
                    print("search Removed duplicate doc for: " + str(query))
            else:
                print("search Removed duplicate link for: " + str(url))
        except Exception:
            continue

    return possibilities


# print(get_acronyms("FDS"))

new_acronyms = dict()
acronyms = json.load(open(DATA_FILE_PATH, mode="r"))
print("Existing File Loaded...")
print("Total Acronyms: %s" % len(acronyms))
i = 0
numAcronym = len(acronyms)
for item in acronyms: 
    i += 1
    print("Acronyms for %s, %d/%d" % (item, i, numAcronym))
    # get_acronyms(item)
    try:
        possibilities = get_acronyms(item, acronyms[item])
    except Exception as e:

        print(e)
        print("Exception Occurred")
        continue
    if possibilities is None or len(possibilities) == 0:
        # del acronyms[item]
        print("No Possibilities :(")
        continue

    acronyms[item]["possibilities"] = possibilities
    # print(possibilities)
    print("%s More to go" % (len(acronyms) - i))

    new_acronyms[item] = acronyms[item]
    print("------------------------------------------------------------")
#    if i == 10:
#        break
    
json.dump(new_acronyms, open(folder_cs_wikipedia_generated+ "cs_wikipedia.json", "w"))

