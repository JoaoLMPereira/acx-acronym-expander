import os
import wikipedia
from bs4 import BeautifulSoup
import requests
import json
import re
import pickle
import sys
sys.setrecursionlimit(10000)

pickle_file = "pickle_file_test"
pickle_object = open(pickle_file, 'wb')
new_acronyms = {}

DATA_FILE_PATH = "../" + "data/acronyms.json"
SEARCH_URL = "https://en.wikipedia.org/w/index.php?title=Special:Search&go=Go&search="
DISAMBIGUATION_URL = "https://en.wikipedia.org/wiki/%s_(disambiguation)"

def get_acronyms(query):
    expansions = list()

    response = requests.get(DISAMBIGUATION_URL % str(query))
    query = query.lower()
    if response.status_code != 404:
        soup = BeautifulSoup(markup=response.text, features="lxml")
        if soup is None:
            return None
        div = soup.find("div", attrs={"class": "mw-parser-output"})
        all_uls = div.findAll("ul")

        for ul in all_uls:
            all_lis = ul.findAll("li")
            for li in all_lis:
                a = li.find("a")
                if a is None:
                    continue
                expansions.append(a)
    return expansions



new_acronyms = dict()
acronyms = json.load(open(DATA_FILE_PATH, mode="r"))
print("Total Acronyms: %s" % len(acronyms))
i = 1
for item in acronyms:
    #print("Acronyms for %s" % item)
    expansions = get_acronyms(item)
    #print(item, expansions)
    new_acronyms[item] = expansions
    print(new_acronyms)

pickle.dump(new_acronyms, pickle_object)

pickle_object.close()
#json.dump(new_acronyms, open("../data/new_acronyms.json", "w"))
