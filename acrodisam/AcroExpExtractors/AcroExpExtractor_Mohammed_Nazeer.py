"""
An improved method for extracting acronym-definition pairs from biomedical Literature
Saneesh Mohammed
K A Abdul Nazeer 
@author: jpereira
"""
import logging
import re

from nltk.tokenize import word_tokenize

from AcroExpExtractors.AcroExpExtractor import AcroExpExtractorRb
from string_constants import (
    FILE_PREPOSITIONS,
    FILE_DETERMINERS,
    FILE_PARTICLES,
    FILE_CONJUNCTIONS,
)
from docutils.nodes import acronym, definition

log = logging.getLogger(__name__)

EXP_ACRO_BOUNDARY = {"(", ":", "-"}
SENTENCE_BOUNDARY = {":", ";", "!", "?", "."}


DIGIT_REPRESENTATIONS = {
    0: ["zero", "zeroth"],
    1: ["one", "first", "i"],
    2: ["two", "second", "ii"],
    3: ["three", "third", "iii"],
    4: ["four", "fourth", "iv"],
    5: ["five", "fifth", "v"],
    6: ["six", "sixth", "vi"],
    7: ["seven", "seventh", "vii"],
    8: ["eight", "eighth", "viii"],
    9: ["nine", "ninth", "ix"],
}

wordEndRegex = re.compile("\s")
# invalidSymbolsExp = re.compile("[;:,!?]")

acronymsToReject = set()
acronymsToReject |= {line.strip().lower() for line in open(FILE_PREPOSITIONS, "r")}
acronymsToReject |= {line.strip().lower() for line in open(FILE_DETERMINERS, "r")}
acronymsToReject |= {line.strip().lower() for line in open(FILE_PARTICLES, "r")}
acronymsToReject |= {line.strip().lower() for line in open(FILE_CONJUNCTIONS, "r")}


class AcroExpExtractor_Mohammed_Nazeer(AcroExpExtractorRb):
    def findNext(self, string, charList, start=0, reverse=False):
        if start > 0:
            string = string[start:]

        stringIterator = enumerate(string)
        if reverse == True:
            stringIterator = zip(string, range(len(string), 0, -1))
        else:
            stringIterator = enumerate(string)

        for p, c in stringIterator:
            if c in charList:
                yield p + start, c

    def extract_candidates(self, text):
        for (pos, char) in self.findNext(string=text, charList=EXP_ACRO_BOUNDARY):
            if char == "(":
                iteratorCloseParenthesis = self.findNext(
                    text, charList=[")"], start=pos
                )
                acronymStop, _ = next(iteratorCloseParenthesis, (None, None))
                if acronymStop == None:
                    continue

                acronym = text[pos + 1 : acronymStop]
                iteratorBeginSentence = self.findNext(
                    string=text, start=pos - 1, charList=SENTENCE_BOUNDARY, reverse=True
                )
                beginningOfSentence, _ = next(iteratorBeginSentence, (-1, None))

                expansion = text[beginningOfSentence + 1 : pos - 1]

            else:
                # acronymStart = wordEndRegex.search(text[:pos-1]).end() # TODO Fix
                # acronym = text[acronymStart: pos-1]
                # acronym = text[:pos-1].rsplit('\s', 1)[0]

                match = re.search(r"(\S+)\s*$", text[:pos])
                if match:
                    acronym = match.group(1)
                else:
                    continue

                iteratorEndSentence = self.findNext(
                    string=text, start=pos + 1, charList=SENTENCE_BOUNDARY
                )
                endOfSentence, _ = next(iteratorEndSentence, (len(text), None))
                expansion = text[pos + 1 : endOfSentence]

            if len(acronym) > len(expansion) or len(acronym.split()) > 2:
                swapAux = acronym
                acronym = expansion
                expansion = swapAux

            yield acronym.strip(), expansion.strip()

    def conditions(self, candidate):
        """
        Checking the validity of acronym - paper algorithm 2

        :param candidate: candidate abbreviation
        :return: True if this is a good candidate
        """

        if len(candidate) < 2:
            return False

        # An acronym can have up to two tokens
        tokens = candidate.split()
        if len(tokens) > 2:
            return False

        numChars = sum(c.isalnum() for c in candidate)

        if numChars > 10:
            return False

        numLetters = sum(c.isalpha() for c in candidate)
        if numLetters < 1:
            return False

        if candidate.lower() in acronymsToReject:
            return False

        return True

    def select_expansion(self, acronym, expansion):
        """
        Takes a definition candidate and an abbreviation candidate
        and returns True if the chars in the abbreviation occur in the definition

        Based on
        A simple algorithm for identifying abbreviation definitions in biomedical texts, Schwartz & Hearst
        :param definition: candidate definition
        :param abbrev: candidate abbreviation
        :return:
        """

        if acronym.lower() in expansion.lower().split():
            # raise ValueError('Abbreviation is full word of definition')
            return None

        acroIndex = len(acronym) - 1
        expIndex = len(expansion) - 1
        # acroSize = -len(acronym)

        while acroIndex >= 0:
            #             try:
            #                 expChar = expansion[expIndex].lower()
            #             except IndexError:
            #                 raise

            acroChar = acronym[acroIndex].lower()

            if not acroChar.isalnum() and not acroChar == "-":
                return None

            if acroChar.isnumeric():
                while (
                    expIndex >= 0 and not expansion[expIndex].isalnum()
                ):  # TODO maybe replace space by something else
                    expIndex -= 1
                endExpIndex = expIndex + 1
                # expIndex -= 1
                while (
                    expIndex >= 0 and expansion[expIndex].isalnum()
                ):  # TODO maybe replace space by something else
                    expIndex -= 1

                expIndex += 1
                expSubStr = expansion[expIndex:endExpIndex].lower()
                # expIndex -= 1
                digitRepresentations = DIGIT_REPRESENTATIONS[int(acroChar)]
                if expSubStr != acroChar and not expSubStr in digitRepresentations:
                    return None

            elif acroChar.isalpha():
                while expIndex >= 0 and expansion[expIndex].lower() != acroChar:
                    expIndex -= 1

            if expIndex < 0:
                return None

            acroIndex -= 1
            expIndex -= 1

        expIndex += 1
        newExpansion = expansion[expIndex:].strip()

        # Checking the validity of expansion - paper algorithm 3
        tokens = len(newExpansion.split())
        length = len(acronym)

        if tokens > min([length + 5, length * 2]):
            # raise ValueError("did not meet min(|A|+5, |A|*2) constraint")
            return None

        # if invalidSymbolsExp.search(newExpansion) != None:
        #   return None  # TODO test ; : , ! ? in expansion

        # Do not return definitions that contain unbalanced parentheses
        # if expansion.count('(') != expansion.count(')'):
        #    raise ValueError("Unbalanced parentheses not allowed in a definition")

        return newExpansion

    def extract_acronym_expansion_pairs(self, text):
        abbrev_map = dict()
        omit = 0
        written = 0
        # The paper says they can get acronyms and expansions from different lines
        # It is not specified how. We are using paragraph spliting for now.

        for acronym, expansion in self.extract_candidates(text):
            if self.conditions(acronym):

                try:
                    expansion = self.select_expansion(acronym, expansion)
                except (ValueError, IndexError) as e:
                    # log.debug("{} Omitting definition {} for candidate {}. Reason: {}".format(i, definition, candidate, e.args[0]))
                    omit += 1
                else:
                    if expansion != None:
                        abbrev_map[acronym] = expansion
                        written += 1
        return abbrev_map

    def get_all_acronyms(self, doc_text):
        tokens = word_tokenize(doc_text)

        return [
            t
            for t in tokens
            if t.isupper() and self.conditions(t) and not (len(t) == 2 and t[1] == ".")
        ]

    # Modified code from schwartz_hearst.extract_abbreviation_definition_pairs to return acronyms with no expansion
    def get_all_acronym_expansion(self, text):

        acronyms = self.get_all_acronyms(text)

        abbrev_map = {acronym: None for acronym in acronyms}
        omit = 0
        written = 0

        # The paper says they can get acronyms and expansions from different lines
        # It is not specified how. We are using paragraph spliting for now.

        for acronym, expansion in self.extract_candidates(text):
            if self.conditions(acronym):

                try:
                    expansion = self.select_expansion(acronym, expansion)
                except (ValueError, IndexError) as e:
                    # log.debug("{} Omitting definition {} for candidate {}. Reason: {}".format(i, definition, candidate, e.args[0]))
                    omit += 1
                else:
                    if expansion != None:
                        abbrev_map[acronym.upper()] = expansion
                        written += 1
        return abbrev_map

    def get_acronym_expansion_pairs(self, text):
        return self.extract_acronym_expansion_pairs(text)
