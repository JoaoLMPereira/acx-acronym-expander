"""
collection of functions used to manipulate the acronymdb dictionary
acronymdb is a dictionary in the format:
(acronym: [list of [acronym_expansion, article_id, expansion_idx_list]])

OLD FORMAT:(acronym: [array of [acronym_expansion, article_id, article_title]])
"""
import csv
import sys

import numpy

import logging

from sqlitedict import SqliteDict

from text_preparation import toUnicode
import pickle as pickle
from helper import AcronymExpansion
from string_constants import file_scraped_definitions_list, FILE_ACORNYMDB

logger = logging.getLogger(__name__)


def createFromScrapedDefinitions():
    csv.field_size_limit(sys.maxsize)

    acronymDB = {}
    loaded_acronyms = 0
    for definition_file in file_scraped_definitions_list:
        # open as csv file with headers
        acronym_csv = csv.DictReader(
            open(definition_file, "rb"), delimiter=",")

        for row in acronym_csv:
            acronym = toUnicode(row["acronym"])
            acronym_expansion = toUnicode(row["acronym_expansion"])
            article_id = toUnicode(row["article_id"])
            if(acronym not in acronymDB):
                acronymDB[acronym] = []
            acronymDB[acronym].append([acronym_expansion
                                       .strip().lower().replace('-', ' '), article_id])
            # , row["article_title"]]) # title was part of old format
            loaded_acronyms += 1
            if(loaded_acronyms % 10000 == 0):
                logger.debug("loaded %d acronyms", loaded_acronyms)

    logger.info("adding def_count values to acronymDB")
    defs_per_acronym = [0] * 1000
    insts_per_def = [0] * 1000
    #num_acronyms = len(acronymDB)
    for acronym, values_for_this_acronym in acronymDB.items():
        values_for_this_acronym = sorted(
            values_for_this_acronym, key=lambda x: x[0])

        def_count = 0
        inst_count = 0
        expansion_of_last_acronym = values_for_this_acronym[0][0]
        #, article_title]\ # title was part of old format in the line below
        for index, [acronym_expansion, article_id]\
                in enumerate(values_for_this_acronym):
            if AcronymExpansion.startsSameWay(acronym_expansion, expansion_of_last_acronym):
                inst_count += 1
                values_for_this_acronym[index].append(def_count)
                values_for_this_acronym[index][0] = expansion_of_last_acronym
            else:
                insts_per_def[min(inst_count, len(insts_per_def) - 1)] += 1
                inst_count = 0
                def_count += 1
                expansion_of_last_acronym = acronym_expansion
                values_for_this_acronym[index].append(def_count)
        defs_per_acronym[min(def_count, len(defs_per_acronym) - 1)] += 1
        acronymDB[acronym] = numpy.array(values_for_this_acronym)

    dump(acronymDB)
    logger.info("Dumped AcronymDB successfully")


# def is_same_expansion(true_exp, pred_exp):
#    true_exp = true_exp.strip().lower().replace("-", " ")
#    pred_exp = " ".join([word[:4] for word in pred_exp.split()])
#    true_exp = " ".join([word[:4] for word in true_exp.split()])
#    if(pred_exp == true_exp):
#        return True
#    #    ed = distance.edit_distance(pred_exp, true_exp)
#    #    if ed < 3:
#    #        return True
#    return False


def dump(acronymDB):
    pickle.dump(
        acronymDB, open(FILE_ACORNYMDB, "wb"), protocol=2)


def load(path=FILE_ACORNYMDB, storageType=None):
    """
    acronymdb is a dictionary in the format:
    (acronym: [list of [acronym_expansion, article_id]])
    """
    logger.debug("loading acronymDB from %s" % path)
    if storageType == "SQLite":
        return SqliteDict(path, flag='r')
    else:
        try:
            return pickle.load(open(path, "rb"))
        except pickle.UnpicklingError:
            logger.warn("File at %s is not a pickle file, trying to load sqlite instead.", path)
            return SqliteDict(path, flag='r')

def add_expansions_to_acronym_db(acronym_db, docId, acro_exp_dict):
    for acronym, expansion in acro_exp_dict.items():
        expList = acronym_db.setdefault(acronym, [])
        expList.append((expansion, docId))
        
if __name__ == "__main__":
    createFromScrapedDefinitions()
