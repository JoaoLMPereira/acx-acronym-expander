import re

from AcronymExpanders import AcronymExpanderEnum
from AcronymExtractors.AcronymExtractor import AcronymExtractor
from helper import AcronymExpansion
import string_constants


class AcronymExtractor_v2(AcronymExtractor):
    """
    keeps acronyms which spell same as english words
    like: PEP (phosphoenolpyruvate), MOO, etc..
    """
    def __init__(self):
        self.pattern = r'\b[A-Z]{3,8}s{0,1}\b'# Limit length 8
        
    def get_acronyms(self, text):
        acronyms = re.findall(self.pattern, text)
        result = {}
        for acronym in set(acronyms):
            result[acronym.upper()] = []
        return result
