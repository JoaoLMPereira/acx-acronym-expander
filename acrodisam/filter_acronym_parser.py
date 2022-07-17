#  Make a parser which calls Leahs parser but also adds an extra layer that
#filters out acronyms and expansions not found on the wikipedia acronyms and
#expansions created in task 2 and acronyms and expansions not found on
#acronymfinder.com (this should be done after Wikipedia because we have to ask
#for a html page which is expensive)

#Leah parser + acronym finder (any acronyms that Leah finds expansions
#for that are not in acronym finder should be a separate list leahonly.
#The others should be in a list leahfinder: leahacronym and then
#all the expansions of acronymfinder.)

from __future__ import division
from sys import argv
import os
import re
import pprint
import sys
import importlib
import pickle
sys.path.append('..')
from AcronymExtractors.AcronymExtractor_v4 import AcronymExtractor_v4
from AcronymExpanders.Expander_fromText_v3 import Expander_fromText_v3
import AcronymFinder


def acronyms(article_text):
    #make sure there are no duplicated acronyms in acronym list
    new_acronym_list = []
    AcronymExtractor_Instance = AcronymExtractor_v4()
    acronyms = AcronymExtractor_Instance.get_acronyms(article_text)
    new_acronym_list = []
    for acronym in acronyms:
        if acronym.endswith('s'):
            acronym = acronym[:-1]
        if acronym not in new_acronym_list:
            new_acronym_list.append(acronym)
    return new_acronym_list

def leah_parser_expansions(acronym, article_text):
    #calls on Expander_fromText_v3 to give expansions from Leah's parser
    Expander_Instance = Expander_fromText_v3()
    expansions = Expander_Instance._expandInText(acronym=acronym, text=article_text)
    return expansions

def pickle_call():
    #uses pickle to open pickle file written from fetch_dynamic.py
    if os.path.getsize("pickle_file_test_copy") > 0:
        with open("pickle_file_test_copy", 'rb') as file_object:
            unpickler = pickle.Unpickler(file_object)
            new_dict = unpickler.load()
            #returns the dictionary of acronyms and their expansions from the
            #pickle file
            return new_dict

def wiki_filter(acronym, expansions):
    # checks to see if acronym expansions from Leah's parser are in pickle wiki file
    wiki_list = []
    parser_list = []
    file_dict = pickle_call()
    for expansion in expansions:
        if acronym in file_dict:
            acronym_expansion = file_dict[acronym]
            if expansion in str(acronym_expansion):
                wiki_list.append(expansion)
            else:
                parser_list.append(expansion)
        else:
            parser_list.append(expansion)
    #returns the list of acronyms that are in wikipedia and those that are not
    return [wiki_list, parser_list]

def call_on_AcronymFinder(acronym):
    #calls on AcronymFinder to check and see if the expansion is correct
    AcronymFinder_expansions = AcronymFinder.getExpansions(acronym)
    return AcronymFinder_expansions



def AcronymFinder_filter(acronym, expansions):
    #filters out acronym expansions from Leah's parser not found on acronymfinder.com
    AcronymFinder_list = []
    non_AF_list = expansions
    AF_expansions = call_on_AcronymFinder(acronym)
    for expansion in expansions:
        expansion = str(expansion)
        new_expansion = expansion.lower()
        for item in AF_expansions:
            item = item.lower()
            if new_expansion in item:
                AcronymFinder_list.append(expansion)
                non_AF_list.remove(expansion)

    return [AcronymFinder_list,non_AF_list]


def main_filter_function(acronyms, article_text):
    # creats three dictionaries: acronyms that are found in wikipedia,
    # acronyms found in acronymfinder.com, and acronyms found in neither
    wiki_dict = {}
    AcronymFinder_dict = {}
    in_neither_dict = {}
    for acronym in acronyms:
        expansions = leah_parser_expansions(acronym, article_text)
        wiki_list = wiki_filter(acronym, expansions)[0]
        parser_list = wiki_filter(acronym, expansions)[1]

        af_list = AcronymFinder_filter(acronym, parser_list)[0]
        in_neither_list = AcronymFinder_filter(acronym, parser_list)[1]
        if wiki_list:
            wiki_dict[acronym] = wiki_list
        if af_list:
            AcronymFinder_dict[acronym] = af_list
        if in_neither_list:
            in_neither_dict[acronym] = in_neither_list

    print("Acronyms found in wikipedia: ")
    print("\n".join("{}\t{}".format(k, v) for k, v in wiki_dict.items()),'\n')
    print("Acronyms found in AcronymFinder: ")
    print("\n".join("{}\t{}".format(k, v) for k, v in AcronymFinder_dict.items()), '\n')
    print("Acronyms found in neither: ")
    print("\n".join("{}\t{}".format(k, v) for k, v in in_neither_dict.items()))


def new_filter(acronyms, article_text):
    new_dict = {}
    for acronym in acronyms:
        wiki_list = []
        AcronymFinder_list = []
        file_dict = pickle_call()
        expansions = leah_parser_expansions(acronym, article_text)
        for expansion in expansions:
            if acronym in file_dict:
                acronym_expansion = file_dict[acronym]
                if expansion in str(acronym_expansion):
                    wiki_list.append(expansion)
        new_dict[acronym] = wiki_list
        AF_expansions = call_on_AcronymFinder(acronym)
        for expansion in expansions:
            expansion = str(expansion)
            new_expansion = expansion.lower()
            for item in AF_expansions:
                item = item.lower()
                if new_expansion in item:
                    AcronymFinder_list.append(expansion)
        new_dict[acronym] = AcronymFinder_list
    cleaner_dict = clean_dict(new_dict)
    return cleaner_dict

# The following fucntions are used to clean up the representations of the
# acronyms in a table
def list_to_string(list_item):
    # Converts a list to a string
    string = ""
    return ', '.join(str(x) for x in list_item)

def flatten(alist):
    # Flattens a list
     newlist = []
     for item in alist:
         if isinstance(item, list):
             newlist = newlist + flatten(item)
         else:
             newlist.append(item)
     return newlist

def list_change(alist):
    # uses two previous functions to convert a convoluted list into a clean string
    newlist = flatten(alist)
    result = list_to_string(newlist)
    return result

def clean_dict(dict):
    # cleans up a dict with values that have lists into a clean string for
    # representation
    new_dict = {}
    for key, val in dict.items():
        new_val = list_change(val)
        new_dict[key] = new_val
    return new_dict
