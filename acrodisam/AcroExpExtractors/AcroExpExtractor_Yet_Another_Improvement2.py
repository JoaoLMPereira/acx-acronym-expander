"""
An improved method for extracting acronym-definition pairs from biomedical Literature
Saneesh Mohammed
K A Abdul Nazeer 
@author: jpereira
"""
import logging
import re
import os

from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

from AcroExpExtractors.AcroExpExtractor import AcroExpExtractorRb
from abbreviations import schwartz_hearst
from string_constants import (
    FILE_PREPOSITIONS,
    FILE_DETERMINERS,
    FILE_PARTICLES,
    FILE_CONJUNCTIONS,
    FOLDER_DATA,
    FILE_JUST_ENGLISH_DICT,
)
from docutils.nodes import acronym, definition

# from nltk.corpus import words
# from distutils.command.clean import clean

# from nltk.corpus import wordnet

# english_words = set(word.strip().lower() for word in open(os.path.join(FOLDER_DATA, "wordsEn.txt")))

# word_set = set(words.words())

english_words2 = set(word.strip().casefold() for word in open(FILE_JUST_ENGLISH_DICT))

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
tokenizer = RegexpTokenizer(r"\w+")


acronymsToReject = set()
acronymsToReject |= {line.strip().lower() for line in open(FILE_PREPOSITIONS, "r")}
acronymsToReject |= {line.strip().lower() for line in open(FILE_DETERMINERS, "r")}
acronymsToReject |= {line.strip().lower() for line in open(FILE_PARTICLES, "r")}
acronymsToReject |= {line.strip().lower() for line in open(FILE_CONJUNCTIONS, "r")}


class AcroExpExtractor_Yet_Another_Improvement(AcroExpExtractorRb):
    def findNext(self, string, charList, start=0, reverse=False):

        if reverse == True:
            if start < len(string):
                remainString = string[:start]

            stringIterator = zip(range(start - 1, -1, -1), reversed(remainString))
        else:
            if start > 0:
                remainString = string[start:]
                stringIterator = zip(range(start, len(string)), remainString)
            else:
                stringIterator = enumerate(string)

        for p, c in stringIterator:
            if c in charList:
                yield p, c

    def findCorrespondingCloseParenthesis(self, text, pos):
        openParen = 1
        closeParen = 0
        iteratorParenthesis = self.findNext(text, charList=[")", "("], start=pos)
        while openParen != closeParen:
            acronymStop, c = next(iteratorParenthesis, (None, None))
            if acronymStop == None:
                return None

            if c == "(":
                openParen += 1
            else:
                closeParen += 1
        return acronymStop

    def hasAnyAlnum(self, string):
        for c in string:
            if c.isalnum():
                return True
        return False

    # We strip left and right by removing all non apha numeric chars unless they are brackets that will close or open inside the string
    def advancedStrip(self, string):
        # left strip
        p = 0
        for p, c in enumerate(string):
            if c.isalnum():
                break
            if c == "(":
                iterFindNext = self.findNext(string, ")", p)
                closePos, _ = next(iterFindNext, (len(string), None))
                # test if there is a alnum at the right of the )
                if closePos + 1 < len(string):
                    hasAlnum = self.hasAnyAlnum(string[closePos + 1 :])
                    if hasAlnum:
                        break

        string = string[p:]
        # right strip
        p = len(string) - 1
        for p, c in zip(range(len(string) - 1, -1, -1), reversed(string)):
            if c.isalnum():
                break
            if c == ")":
                iterFindNext = self.findNext(string, "(", p, reverse=True)
                openPos, _ = next(iterFindNext, (-1, None))
                # test if there is a alnum at the left of the (
                if openPos > 0:
                    hasAlnum = self.hasAnyAlnum(string[:openPos])
                    if hasAlnum:
                        break

        return string[: p + 1]

    def extract_candidates(self, text):
        for (pos, char) in self.findNext(string=text, charList=EXP_ACRO_BOUNDARY):
            acronymOnRight = True
            if char == "(":
                acronymStop = self.findCorrespondingCloseParenthesis(text, pos + 1)
                if acronymStop == None:
                    continue

                acronym = text[pos + 1 : acronymStop]
                iteratorBeginSentence = self.findNext(
                    string=text, start=pos, charList=SENTENCE_BOUNDARY, reverse=True
                )
                beginningOfSentence, _ = next(iteratorBeginSentence, (-1, None))

                expansion = text[beginningOfSentence + 1 : pos]

            else:
                # acronymStart = wordEndRegex.search(text[:pos-1]).end() # TODO Fix
                # acronym = text[acronymStart: pos-1]
                # acronym = text[:pos-1].rsplit('\s', 1)[0]
                if (
                    pos < 2
                    or pos >= len(text) - 1
                    or (
                        char == ":"
                        and not text[pos - 1].isspace()
                        and not text[pos + 1].isspace()
                    )
                    or (
                        char == "-"
                        and (not text[pos - 1].isspace() or not text[pos + 1].isspace())
                    )
                ):
                    continue

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
                acronymOnRight = False

            acronym = self.advancedStrip(acronym)
            expansion = self.advancedStrip(expansion)

            if len(acronym) > len(expansion) or len(acronym.split()) > 2:
                swapAux = acronym
                acronym = expansion
                expansion = swapAux
                acronymOnRight = not acronymOnRight

            # yield acronym.strip(), expansion.strip(), acronymOnRight
            yield acronym, expansion, acronymOnRight

    def conditions(self, candidate):
        """
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

        # In case we have lower case acronym we verify if they are English words
        # verify tokens split by non alfanum
        # for token in tokenizer.tokenize(candidate)

        if candidate[1:].islower():
            for token in tokens:
                if not token.casefold() in english_words2:
                    return True

            return False

        return True

    def select_expansion(self, acronym, expansion, acronymOnRight=True):

        # acronym = acronym.strip()
        i = 0
        while not acronym[i].isalnum():
            i += 1
        acronym = acronym[i:]

        inEpansion = True
        expanTokens = tokenizer.tokenize(expansion.lower())
        for acroToken in tokenizer.tokenize(acronym.lower()):
            if not acroToken in expanTokens:
                # raise ValueError('Abbreviation is full word of definition')
                inEpansion = False

        if inEpansion:
            return None

        if acronymOnRight:
            return self.select_expansion_right(acronym, expansion)
        else:
            return self.select_expansion_left(acronym, expansion)

    def select_expansion_right(self, acronym, expansion):
        """
        Takes a definition candidate and an abbreviation candidate
        and returns True if the chars in the abbreviation occur in the definition

        Based on
        A simple algorithm for identifying abbreviation definitions in biomedical texts, Schwartz & Hearst

        EXPANSION (ACRONYM)

        :param definition: candidate definition
        :param abbrev: candidate abbreviation
        :return:
        """
        # ignores special symbols and white spaces in acronym for matching purposes

        acroIndex = len(acronym) - 1
        expIndex = len(expansion) - 1
        firstMatch = -1
        # acroSize = -len(acronym)

        while acroIndex >= 0:
            #             try:
            #                 expChar = expansion[expIndex].lower()
            #             except IndexError:
            #                 raise

            acroChar = acronym[acroIndex].lower()

            if not acroChar.isalnum():
                acroIndex -= 1
                continue

            if acroChar.isnumeric():
                while 1:
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
                        expIndex -= 1
                        if firstMatch < 0 or expIndex < 0:
                            return None
                    else:
                        if firstMatch < 0:
                            firstMatch = expIndex + 1
                        break

            elif acroChar.isalpha():

                # if acroIndex = 0 we have to macth only with the beginning of a word
                while (expIndex >= 0 and expansion[expIndex].lower() != acroChar) or (
                    acroIndex == 0
                    and expIndex >= 1
                    and expansion[expIndex - 1].isalnum()
                ):
                    expIndex -= 1
                    if firstMatch < 0 and expansion[expIndex].isspace():
                        return None
                if firstMatch < 0:
                    firstMatch = expIndex + 1

            if expIndex < 0:
                return None

            acroIndex -= 1
            expIndex -= 1

        # we don't want to break sentences

        while expIndex >= 0 and not wordEndRegex.match(expansion[expIndex]):
            expIndex -= 1
        expIndex += 1

        while len(expansion) > firstMatch and not wordEndRegex.match(
            expansion[firstMatch]
        ):
            firstMatch += 1

        newExpansion = expansion[expIndex:firstMatch]
        tokens = len(newExpansion.split())
        length = len(acronym)

        if tokens > min([length + 5, length * 2]):
            # raise ValueError("did not meet min(|A|+5, |A|*2) constraint")
            return None

        # if newExpansion # TODO test ; : , ! ? in expansion

        # Do not return definitions that contain unbalanced parentheses
        # if expansion.count('(') != expansion.count(')'):
        #    raise ValueError("Unbalanced parentheses not allowed in a definition")

        return newExpansion

    def select_expansion_left(self, acronym, expansion):
        """
        Takes a definition candidate and an abbreviation candidate
        and returns True if the chars in the abbreviation occur in the definition

        Based on
        A simple algorithm for identifying abbreviation definitions in biomedical texts, Schwartz & Hearst
        :param definition: candidate definition
        :param abbrev: candidate abbreviation
        :return:
        """
        # ignores special symbols and white spaces in acronym for matching purposes
        if acronym.lower() in expansion.lower().split():
            # raise ValueError('Abbreviation is full word of definition')
            return None

        acroIndex = 0
        expIndex = 0
        acroEnd = len(acronym)
        expEnd = len(expansion)
        firstMatch = -1
        # acroSize = -len(acronym)

        while acroIndex < acroEnd:
            #             try:
            #                 expChar = expansion[expIndex].lower()
            #             except IndexError:
            #                 raise

            acroChar = acronym[acroIndex].lower()

            if not acroChar.isalnum():
                acroIndex += 1
                continue

            if acroChar.isnumeric():
                while 1:
                    while (
                        expIndex < expEnd and not expansion[expIndex].isalnum()
                    ):  # TODO maybe replace space by something else
                        expIndex += 1
                    startExpIndex = expIndex
                    # expIndex -= 1
                    while (
                        expIndex < expEnd and expansion[expIndex].isalnum()
                    ):  # TODO maybe replace space by something else
                        expIndex += 1

                    # expIndex += 1
                    expSubStr = expansion[startExpIndex:expIndex].lower()
                    # expIndex -= 1
                    digitRepresentations = DIGIT_REPRESENTATIONS[int(acroChar)]
                    if expSubStr != acroChar and not expSubStr in digitRepresentations:
                        expIndex += 1
                        if firstMatch < 0 or expIndex >= expEnd:
                            return None
                    else:
                        if firstMatch < 0:
                            firstMatch = expIndex
                        break

            elif acroChar.isalpha():
                # if acroIndex = 0 we have to macth only with the beginning of a word
                while (
                    expIndex < expEnd and expansion[expIndex].lower() != acroChar
                ) or (
                    acroIndex == 0
                    and expIndex >= 1
                    and expansion[expIndex - 1].isalnum()
                ):
                    expIndex += 1
                    if firstMatch < 0 and expansion[expIndex].isspace():
                        return None
                if firstMatch < 0:
                    firstMatch = expIndex

            if expIndex >= expEnd:
                return None

            acroIndex += 1
            expIndex += 1

        # we don't want to break sentences

        while expIndex < expEnd and not wordEndRegex.match(expansion[expIndex]):
            expIndex += 1

        while firstMatch >= 0 and not wordEndRegex.match(expansion[firstMatch]):
            firstMatch -= 1

        firstMatch += 1

        newExpansion = expansion[firstMatch:expIndex]
        tokens = len(newExpansion.split())
        length = len(acronym)

        if tokens > min([length + 5, length * 2]):
            # raise ValueError("did not meet min(|A|+5, |A|*2) constraint")
            return None

        return newExpansion

    def extract_acronym_expansion_pairs(self, text, abbrev_map=None):
        if abbrev_map == None:
            abbrev_map = {}
        # omit = 0
        # written = 0
        # The paper says they can get acronyms and expansions from different lines
        # It is not specified how. We are using paragraph spliting for now.
        sentence_iterator = schwartz_hearst.yield_lines_from_doc(text)
        for text in sentence_iterator:
            for acronym, expansion, acronymOnRight in self.extract_candidates(text):
                # log.debug("Found candidate: acronym: " + acronym + " expansion: "+ expansion)
                if self.conditions(acronym):

                    try:
                        newExpansion = self.select_expansion(
                            acronym, expansion, acronymOnRight
                        )
                    except (ValueError, IndexError) as e:
                        # log.debug("{} Omitting definition {} for candidate {}. Reason: {}".format(i, definition, candidate, e.args[0])
                        newExpansion = None
                        # omit += 1
                    if newExpansion != None:
                        abbrev_map[acronym] = self.advancedStrip(newExpansion)
                        # written += 1
        return abbrev_map

    def get_all_acronyms(self, doc_text):
        tokens = word_tokenize(doc_text)
        return [t for t in tokens if t.isupper() and self.conditions(t)]
        # return [t for t in tokens if t.isupper() and self.conditions(t) and not (len(t) == 2 and t[1] == '.')]

    # Modified code from schwartz_hearst.extract_abbreviation_definition_pairs to return acronyms with no expansion
    def get_all_acronym_expansion(self, text):

        acronyms = self.get_all_acronyms(text)

        abbrev_map = {acronym: None for acronym in acronyms}

        return self.extract_acronym_expansion_pairs(text, abbrev_map)

    def get_acronym_expansion_pairs(self, text):
        return self.extract_acronym_expansion_pairs(text)

    def get_best_expansion(self, acro, text):
        return super().get_best_expansion(acro, text)
