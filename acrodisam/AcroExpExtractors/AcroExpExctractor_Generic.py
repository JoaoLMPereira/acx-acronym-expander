"""
Created on Jun 26, 2019

@author: jpereira
"""
from AcroExpExtractors.AcroExpExtractor import AcroExpExtractorRb


class AcroExpExtractor_Generic(AcroExpExtractorRb):
    def __init__(self, acronymExtractor, expansionExtractor):
        self.acronymExtractor = acronymExtractor
        self.expansionExtractor = expansionExtractor

    # returns a dictionary of {acronym:AcronymExpansion} pairs where AcronymExpansion is None if no expansion is found in text for the acronym
    def get_all_acronym_expansion(self, text):
        expansionDict = {}
        acronymsList = self.acronymExtractor.get_acronyms(text)
        for acronym in acronymsList:
            expansion = self.expansionExtractor._expandInText(
                text=text, acronym=acronym
            )
            if expansion:
                expansionDict[acronym] = expansion
            else:
                expansionDict[acronym] = None

        return expansionDict

    # returns a dictionary of valid {acronym:AcronymExpansion} pairs
    def get_acronym_expansion_pairs(self, text):
        expansionDict = {}
        acronymsList = self.acronymExtractor.get_acronyms(text)
        for acronym in acronymsList:
            expansion = self.expansionExtractor._expandInText(
                text=text, acronym=acronym
            )
            if expansion:
                expansionDict[acronym] = expansion

        return expansionDict
