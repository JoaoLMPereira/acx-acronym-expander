from __future__ import division 
import re

from AcronymExtractors.AcronymExtractor_v2_small import AcronymExtractor_v2_small


class AcronymExtractor_v3_small(AcronymExtractor_v2_small):
    """
    keeps acronyms which spell same as english words
    decreases minimum acronym length to 2
    support acronyms with some lower-case characters
    #todo:support numbers (things like CCl4)
    """
    def __init__(self):
        self.pattern = r'\b[A-Z0-9]{2,8}s{0,1}\b'
    
    def isMoreThanHalfUpper(self, acronym):
        #acronyms of length 2 should be all capital
        if(len(acronym)<=2):
            return sum(1 for char in acronym if char.isupper() or char.isdigit())==2
        numUppers = sum(1 for char in acronym if char.isupper())
        return numUppers>=(len(acronym)/2)
        
    def get_acronyms(self, text):
        acronyms = re.findall(self.pattern, text, flags=re.IGNORECASE)
        acronyms = [item.upper() for item in acronyms if self.isMoreThanHalfUpper(item)]
        result = {}
        for acronym in set(acronyms):
            result[acronym] = []
        return result
