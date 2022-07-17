"""
An implementation of the rule-based in expander provided for the competition:
Amir Pouran Ben Veyseh, Franck Dernoncourt, Thien Huu Nguyen1, Walter Chang and Leo Anthony Celi.
"Acronym Identification and Disambiguation Shared Tasks for Scientific Document Understanding"

This in-expander was provided by the authors of the competition as a baseline and is inspired by the Schwartz and Hearst algorithm.
Original code can be found in this repository:
https://github.com/amirveyseh/AAAI-21-SDU-shared-task-1-AI

This in expander is also used as one of the base models that SciDr (AcroExpExtractor_Sci_Dr) uses to get features for the training of its CRF ensemble.
"""


import logging
import re

import spacy
from DatasetParsers.process_tokens_and_bio_tags import create_diction
from nltk import word_tokenize

from AcroExpExtractors.AcroExpExtractor import (
    AcroExpExtractorRb,
)

tokenizer = word_tokenize

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 30000000


class AcroExpExtractor_Character_Match_AAAI(AcroExpExtractorRb):
    def _process_raw_input(self, text):
        return [t.text for t in nlp(text) if len(t.text.strip()) > 0]

    def _predict(self, tokenized_input):
        predictions = ["O"] * len(tokenized_input)
        for i, t in enumerate(tokenized_input):
            if (
                t[0].isupper()
                and len([c for c in t if c.isupper()]) / len(t) > 0.6
                and 2 <= len(t) <= 10
            ):
                predictions[i] = "B-short"
                long_cand_length = min([len(t) + 5, len(t) * 2])
                cand_long = []
                cand_long_index = []
                left = True
                if i < len(tokenized_input) - 3 and tokenized_input[i + 1] == "(":
                    left = False
                    for j in range(
                        i + 2, min([i + 2 + long_cand_length, len(tokenized_input)])
                    ):
                        if tokenized_input[j] != ")":
                            cand_long.append(tokenized_input[j])
                            cand_long_index.append(j)
                        else:
                            break
                elif (
                    1 < i < len(tokenized_input) - 3
                    and tokenized_input[i - 1] == "("
                    and tokenized_input[i + 1] == ")"
                ):
                    for k in range(0, long_cand_length):
                        j = i - 2 - k
                        if j > 0:
                            cand_long.insert(0, tokenized_input[j])
                            cand_long_index.insert(0, j)
                cand_long = " ".join(cand_long)
                long_form = ""
                if len(cand_long) > 0:
                    if left:
                        sIndex = len(t) - 1
                        lIndex = len(cand_long) - 1
                        while sIndex >= 0:
                            curChar = t[sIndex].lower()
                            if curChar.isdigit() or curChar.isalpha():
                                while (
                                    lIndex >= 0 and cand_long[lIndex].lower() != curChar
                                ) or (
                                    sIndex == 0
                                    and lIndex > 0
                                    and (
                                        cand_long[lIndex - 1].isdigit()
                                        or cand_long[lIndex - 1].isalpha()
                                    )
                                ):
                                    lIndex -= 1
                                if lIndex < 0:
                                    break
                                lIndex -= 1
                            sIndex -= 1
                        if lIndex >= 0:
                            try:
                                lIndex = cand_long.rindex(" ", 0, lIndex + 1) + 1
                            except:
                                lIndex = 0
                            if cand_long:
                                cand_long = cand_long[lIndex:]
                                long_form = cand_long
                    else:
                        sIndex = 0
                        lIndex = 0
                        if t[0].lower() == cand_long[0].lower():
                            while sIndex < len(t):
                                curChar = t[sIndex].lower()
                                if curChar.isdigit() or curChar.isalpha():
                                    while (
                                        lIndex < len(cand_long)
                                        and cand_long[lIndex].lower() != curChar
                                    ):
                                        lIndex += 1
                                    if lIndex >= len(cand_long):
                                        break
                                    lIndex += 1
                                sIndex += 1
                            if lIndex < len(cand_long):
                                try:
                                    lIndex = cand_long[lIndex:].index(" ") + lIndex + 1
                                except:
                                    lIndex = len(cand_long)
                                if cand_long:
                                    cand_long = cand_long[:lIndex]
                                    long_form = cand_long
                    if long_form:
                        long_form = long_form.split()
                        if left:
                            long_form_index = cand_long_index[-len(long_form) :]
                        else:
                            long_form_index = cand_long_index[: len(long_form)]
                        first = True
                        for j in range(len(tokenized_input)):
                            if j in long_form_index:
                                if first:
                                    predictions[j] = "B-long"
                                    first = False
                                else:
                                    predictions[j] = "I-long"
        return predictions

    def get_all_acronym_expansion(self, text):
        tokenized_input = self._process_raw_input(text)
        predictions = self._predict(tokenized_input)
        pairs = create_diction(tokenized_input, predictions)
        return pairs

    def get_acronym_expansion_pairs(self, text):
        tokenized_input = self._process_raw_input(text)
        predictions = self._predict(tokenized_input)
        pairs = create_diction(tokenized_input, predictions, all_acronyms=False)
        return pairs
