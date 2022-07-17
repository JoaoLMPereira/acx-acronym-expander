from __future__ import division
import sys
sys.path.append('..')
import re

from AcronymExtractors.AcronymExtractor import AcronymExtractor


class AcronymExtractor_v4(AcronymExtractor):
    """
    Leah Acronym extractor 20/6/2018
    """
    def get_acronyms(self, text):  # Find Acronyms in text
        pattern = r'\b[A-Z]{2,8}s{0,1}\b'
        acronyms = re.findall(pattern, text)

        acronyms = [acronym for acronym in acronyms if acronym.lower()]
        return acronyms
