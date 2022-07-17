import re

from AcronymExpanders import AcronymExpanderEnum
from AcronymExtractors.AcronymExtractor import AcronymExtractor
from helper import AcronymExpansion
import string_constants


class AcronymExtractor_v1(AcronymExtractor):
    """
    Finds acronyms and discards those which spell same as regular english words
    """
    def get_acronyms(self, text):
        english_words = set(word.strip().lower()
                            for word in open(string_constants.file_english_words))
        pattern = r'\b[A-Z]{3,8}s{0,1}\b'  # Limit length 8
        acronyms = re.findall(pattern, text)
        acronyms = [
            acronym for acronym in acronyms if acronym.lower() not in english_words]
        result = {}
        for acronym in set(acronyms):
            result[acronym.upper()] = []
        return result
