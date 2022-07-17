'''
Calls the original code from Schwartz and Hearst in java
to process French documents.

@author: mprieur
'''


import logging
from AcroExpExtractors.AcroExpExtractor_Original_Schwartz_Hearst\
    import AcroExpExtractor_Original_Schwartz_Hearst
from string_constants import FR_MAPPING, FR_PREPOSITIONS
from helper import get_lang_dict, format_text


FRENCH_WORDS = get_lang_dict("FR")
logger = logging.getLogger(__name__)


class AcroExpExtractor_Original_Schwartz_HearstFR(AcroExpExtractor_Original_Schwartz_Hearst):
    """Python class calling the java acronym expansion extractor from Shwartz and Hearst.
    """
    def __init__(self):
        AcroExpExtractor_Original_Schwartz_Hearst.__init__(self)


    def valid_expansion(self, modified_exp, key):
        """Verify if a given expansion fit the acronym.
            Args :
                modified_exp (str): The expansion
                key (str): The acronym
            Res :
                boolean
        """
        key = key.replace('.', '')
        if '&' in key and 'et ' in modified_exp:
            key = key.replace('&', 'et')
        if len(modified_exp)>len(key):
            modified_exp = modified_exp.split(' ')
            if not modified_exp[-1].lower() in FR_PREPOSITIONS\
            and not modified_exp[0].lower() in FR_PREPOSITIONS:
                for word in modified_exp:
                    for index, letter in enumerate(word):
                        if letter.lower() == key[0].lower()\
                        and not(len(word) > 2\
                        and index == len(word)-1\
                        and letter == 's'):
                            key = key[1:]
                            if len(key) == 0:
                                return True
        return False


    def matching_verifications(self, pairs, res, key):
        """Verify that the extension returned match the acronym.
            Args :
                pairs (dict) : the pairs of acronym/extension
                res (string) : the extension
                key (string) : the acronym
        """
        if key not in pairs\
            and not (res[:3].lower() == "et " or res[-3:].lower() == " et")\
            and key[:2].lower() not in ["l’", "l'", "c'", "d'", "d’"]\
            and (key.lower() not in FRENCH_WORDS or key.isupper())\
            and " " not in key\
            and key.lower()+'s' not in res.lower()\
            and key.replace('.', '')[-1].lower() in res.lower().split(' ')[-1]:
                if res[:2].lower() in ["l’", "l'", "d'", "c'", "c’", "d’", "«\xa0"]:
                    res = res[2:]
                if self.valid_expansion(res, key)\
                and res.lower() != key.lower()\
                and not (len(key) > 3\
                and key[-1] == 's'\
                and key[:-1].lower() in res.lower()):
                    pairs[key] = res
        return pairs


    def sh_adaptation(self, pairs, text):
        """Call the Schwartz and hearst algorithm and apply post-process
            verifications.
            Args :
                pairs (dict) : acronym and their expansion
                text (str) : the text to process
            Returns :
                pairs (dict)
        """
        text = format_text(text).replace('-', ' ')
        java_map = self.algorithm(text)
        entry_set = java_map.entrySet()
        java_iterator = entry_set.iterator()

        while java_iterator.hasNext():
            entry = java_iterator.next()
            key = str(entry.key)
            res = str(entry.value).replace('"', '').replace("»", "").strip()
            pairs = self.matching_verifications(pairs, res, key)
        return pairs


    def get_acronym_expansion_pairs(self, text, lang="FR"):
        """Extract the (acronym, extansion) pairs from a given text.
            Args :
                text (str): the from which to extract the pairs
                lang (str): the language of the input text
            Returns:
                pairs (dict): the dict of pairs with the acronym as key
        """
        pairs = {}
        return self.sh_adaptation(pairs, text)
