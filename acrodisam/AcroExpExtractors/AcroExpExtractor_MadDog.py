"""
MadDog Acronym Identification Implementation
Code modified from original code of Veyseh and Franck, available at: https://github.com/amirveyseh/MadDog

Run 'python -m spacy download en_core_web_sm' before using this in-expander to download spacy files
@author:JRCasanova
"""


import json
import logging
import re
import string

import spacy
from DatasetParsers.process_tokens_and_bio_tags import tokens_to_raw_text
from string_constants import FOLDER_WORDSETS

from AcroExpExtractors.AcroExpExtractor import (
    AcroExpExtractorRb,
)
from AcroExpExtractors.utils_MadDog import constant_MadDog as constant

logger = logging.getLogger(__name__)

# TODO lazy loading
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 30000000

with open(FOLDER_WORDSETS + "stopWords_MadDog.txt") as file:
    stop_words = [l.strip() for l in file.readlines()]


class AcroExpExtractor_MadDog(AcroExpExtractorRb):
    """Implementation of the MadDog rule based algorithm."""

    def __init__(self):
        pass

    def _short_extract(
        self, sentence, threshold, starting_lower_case, ignore_dot=False
    ):
        """Returns a list with the indexes of the acronyms present in sentence.

        Args:
            sentence (list): a list with a tokenized sentence
            threshold (float): a float representing the threshold of capitals that must be in a string for it to be considered an acronym
            starting_lower_case (bool): a bool indicating if an acronym can start with a lower case character
            ignore_dot (bool, optional): a bool indicating if dots in an acronym should be ignored. Defaults to False.

        Returns:
            list: a list with the indexes of the acronyms present in sentence.
        """
        shorts = []
        for i, t in enumerate(sentence):
            if ignore_dot:
                t = t.replace(".", "")
            if len(t) == 0:
                continue
            if not starting_lower_case:
                if (
                    t[0].isupper()
                    and len([c for c in t if c.isupper()]) / len(t) > threshold
                    and 2 <= len(t) <= 10
                ):
                    shorts.append(i)
            else:
                if (
                    len([c for c in t if c.isupper()]) / len(t) > threshold
                    and 2 <= len(t) <= 10
                ):
                    shorts.append(i)
        return shorts

    def _extract_cand_long(
        self,
        sentence,
        token,
        ind,
        ignore_punc=False,
        add_punc=False,
        small_window=False,
    ):
        """Extracts a candidate long form for a given short form. Only long forms of the format "long form (short form)" or "short form (long form)" are extracted.

        Args:
            sentence (list): a list with a tokenized sentence
            token (str): the short form
            ind (int): the index of the short form in the sentence
            ignore_punc (bool, optional): bool indicating if punctuation should be ignored in the extraction process (excludes "(" and ")"). Defaults to False.
            add_punc (bool, optional): bool indicating if "=" and ":" should not be considered punctuation when ignore_punc is True. Defaults to False.
            small_window (bool, optional): bool indicating if a small or big window should be considered around the short form when extracting a long form. Defaults to False.

        Returns:
            list: a list with the tokenized extracted long form
            list: a list with the indexes of the extracted long form in sentence
            bool: a bool indicating the extracted long form is to the left or right of the short form
        """

        if not small_window:
            long_cand_length = min([len(token) + 10, len(token) * 3])
        else:
            long_cand_length = min([len(token) + 5, len(token) * 2])
        cand_long = []
        cand_long_index = []
        left = True
        right_ind = 1
        left_ind = 1
        if add_punc:
            excluded_puncs = ["=", ":"]
        else:
            excluded_puncs = []
        if ignore_punc:
            while ind + right_ind < len(sentence) and sentence[ind + right_ind] in [
                p
                for p in string.punctuation
                if p != "(" and p != ")" and p not in excluded_puncs
            ]:
                right_ind += 1
            while ind - left_ind > 0 and sentence[ind - left_ind] in [
                p
                for p in string.punctuation
                if p != "(" and p != ")" and p not in excluded_puncs
            ]:
                left_ind -= 1
        if ind < len(sentence) - 2 - right_ind and (
            sentence[ind + right_ind] == "("
            or sentence[ind + right_ind] == "="
            or sentence[ind + right_ind] in excluded_puncs
        ):
            left = False
            for j in range(
                ind + right_ind + 1,
                min([ind + right_ind + 1 + long_cand_length, len(sentence)]),
            ):
                if sentence[j] != ")":
                    cand_long.append(sentence[j])
                    cand_long_index.append(j)
                else:
                    break
        elif (
            1 < ind - (left_ind - 1)
            and ind + right_ind < len(sentence)
            and (
                (sentence[ind - left_ind] == "(" and sentence[ind + right_ind] == ")")
                or sentence[ind - left_ind] in excluded_puncs
            )
        ):
            for k in range(0, long_cand_length):
                j = ind - left_ind - 1 - k
                if j > -1:
                    cand_long.insert(0, sentence[j])
                    cand_long_index.insert(0, j)
        return cand_long, cand_long_index, left

    def _extract_high_recall_cand_long(
        self, sentence, token, ind, small_window=False, left=False
    ):
        """Extracts a candidate long form for a given short form.

        Args:
            sentence (list): a list with a tokenized sentence
            token (str): the short form
            ind (int): the index of the short form in the sentence
            small_window (bool, optional): bool indicating if a small or big window should be considered around the short form when extracting a long form. Defaults to False.
            left (bool, optional): a bool indicating the long form is to the left or right of the short form. Defaults to False.

        Returns:
            list: a list with the tokenized extracted long form
            list: a list with the indexes of the extracted long form in sentence
            bool: a bool indicating the extracted long form is to the left or right of the short form
        """

        long_cand_length = min([len(token) + 10, len(token) * 3])
        cand_long = []
        cand_long_index = []
        if not left:
            for j in range(ind + 1, min([ind + long_cand_length, len(sentence)])):
                cand_long.append(sentence[j])
                cand_long_index.append(j)
        else:
            for k in range(0, long_cand_length):
                j = ind - 1 - k
                if j > -1:
                    cand_long.insert(0, sentence[j])
                    cand_long_index.insert(0, j)
        return cand_long, cand_long_index, left

    def _create_diction(
        self, sentence, labels, all_acronyms=True, tag="", map_chars=False, diction={}
    ):
        """Creates a dictionary with acronyms as keys and expansions as values from text annotated with BIO tags.

        Args:
            sentence (list): a list with a tokenized sentence
            labels (list): a list where each element is a BIO tag: "B-long", "I-long", "B-short", "I-short", "O"
            all_acronyms (bool, optional): a bool to indicate if the dictionary should contain all the acronyms (no expansion in text) or not. Defaults to True.
            tag (str, optional): a string indicating the techinique used to extract the expansion. Defaults to "".
            map_chars (bool, optional): a bool indicating if the map_chars function should be used. Defaults to False.
            diction (dict, optional): a dictionary that already has some of the acronym-expansion pairs. Defaults to {}.

        Returns:
            dict:
             a dict where the keys are acronyms and the values are a list with the expansion the indexes for the acronym, the indexes for the expansion and the
             technique used for extraction.
        """

        shorts = []
        longs = []
        isShort = True
        phr = []
        for i in range(len(sentence)):
            if (
                labels[i] == "O"
                or (isShort and "long" in labels[i])
                or (not isShort and "short" in labels[i])
                or (labels[i].startswith("B"))
            ):
                if len(phr):
                    if isShort:
                        shorts.append((phr[0], phr[-1]))
                    else:
                        longs.append((phr[0], phr[-1]))
                    phr = []
            if "short" in labels[i]:
                isShort = True
                phr.append(i)
            if "long" in labels[i]:
                isShort = False
                phr.append(i)
        if len(phr):
            if isShort:
                shorts.append((phr[0], phr[-1]))
            else:
                longs.append((phr[0], phr[-1]))
        acr_long = {}
        for long in longs:
            best_short = []
            ## check if the long form is already mapped in given diction
            if long in diction and diction[long] in shorts:
                best_short = diction[long]
            best_dist = float("inf")
            if not best_short:
                best_short_cands = []
                for short in shorts:
                    long_form = self._character_match(
                        sentence[short[0]],
                        sentence[long[0] : long[1] + 1],
                        list(range(long[1] + 1 - long[0])),
                        output_string=True,
                        is_candidate=False,
                    )
                    if long_form:
                        best_short_cands.append(short)
                if len(best_short_cands) == 1:
                    best_short = best_short_cands[0]
            if not best_short and map_chars:
                best_short_cands = []
                for short in shorts:
                    long_form = self._map_chars(
                        sentence[short[0]], sentence[long[0] : long[1] + 1]
                    )
                    if long_form:
                        best_short_cands.append(short)
                if len(best_short_cands) == 1:
                    best_short = best_short_cands[0]
            if not best_short:
                best_short_cands = []
                for short in shorts:
                    is_mapped = self._map_chars_with_capitals(
                        sentence[short[0]], sentence[long[0] : long[1] + 1]
                    )
                    if is_mapped:
                        best_short_cands.append(short)
                if len(best_short_cands) == 1:
                    best_short = best_short_cands[0]
            if (
                not best_short
                and long[1] < len(sentence) - 2
                and sentence[long[1] + 1] == "("
                and "short" in labels[long[1] + 2]
            ):
                for short in shorts:
                    if short[0] == long[1] + 2:
                        best_short = short
                        break
            if (
                not best_short
                and long[0] > 1
                and sentence[long[0] - 1] == "("
                and "short" in labels[long[0] - 2]
            ):
                for short in shorts:
                    if short[1] == long[0] - 2:
                        best_short = short
                        break
            if not best_short:
                for short in shorts:
                    if short[0] > long[1]:
                        dist = short[0] - long[1]
                    else:
                        dist = long[0] - short[1]
                    if dist < best_dist:
                        best_dist = dist
                        best_short = short
            if best_short:
                short_form_info = " ".join(sentence[best_short[0] : best_short[1] + 1])
                long_form_info = [
                    " ".join(sentence[long[0] : long[1] + 1]),
                    best_short,
                    [long[0], long[1]],
                    tag,
                    1,
                ]
                if short_form_info in acr_long:
                    long_form_info[4] += 1
                acr_long[short_form_info] = long_form_info
        if all_acronyms:
            for short in shorts:
                acr = " ".join(sentence[short[0] : short[1] + 1])
                if acr not in acr_long:
                    acr_long[acr] = ["", short, [], tag, 1]
        return acr_long

    def _map_chars(self, acronym, long):
        """Matches the acronym to the expansion based on the number of initials of the expansion that are in the acronym.

        Args:
            acronym (str): the acronym
            long (list): a list where each element is a word of the expansion

        Returns:
            str: the long if it was matched with the acronym, otherwise None is returned.
        """

        capitals = []
        for c in acronym:
            if c.isupper():
                capitals.append(c.lower())
        initials = [w[0].lower() for w in long]
        ratio = len([c for c in initials if c in capitals]) / len(initials)
        if ratio >= 0.6:
            return long
        else:
            return None

    def _map_chars_with_capitals(self, acronym, long):
        """Matches the acronym to the long-form which has the same initial capitals as the acronym

        Args:
            acronym (str): the acronym
            long (list): a list where each element is a word of the expansion

        Returns:
            bool: True if the acronym is matched to the expansion and False otherwise
        """
        capitals = []
        for c in acronym:
            if c.isupper():
                capitals.append(c.lower())
        long_capital_initials = []
        for w in long:
            if w[0].isupper():
                long_capital_initials.append(w[0].lower())
        if len(capitals) == len(long_capital_initials) and all(
            capitals[i] == long_capital_initials[i] for i in range(len(capitals))
        ):
            return True
        else:
            return False

    def _schwartz_extract(
        self,
        sentence,
        shorts,
        remove_parentheses,
        ignore_hyphen=False,
        ignore_punc=False,
        add_punc=False,
        small_window=False,
        no_stop_words=False,
        ignore_righthand=False,
        map_chars=False,
        default_diction=False,
    ):
        """Returns a dictionary of acronym-definition pairs present in sentence. If an acronym does not have a definition present in text, the expansion is None.

        This function is based on the original Java implementation of the Schwartz and Hearst extraction algorithm.

        Args:
            sentence (list): a list with a tokenized sentence
            shorts (list): a list with the indexes of acronyms in sentence
            remove_parentheses (bool): a bool indicating if parenthesis should be removed from expansions.
            ignore_hyphen (bool, optional): a bool indicating if hypens in text should be ignored. Defaults to False.
            ignore_punc (bool, optional): bool indicating if punctuation should be ignored in the extraction process (excludes "(" and ")"). Defaults to False.
            add_punc (bool, optional): bool indicating if "=" and ":" should not be considered punctuation when ignore_punc is True. Defaults to False.
            small_window (bool, optional): bool indicating if a small or big window should be considered around the short form when extracting a long form. Defaults to False.
            no_stop_words (bool, optional): a bool indicating if an expansion is allowed to have stop words. Defaults to False.
            ignore_righthand (bool, optional): a bool indicating if what is right of the acronym should be ignored. Defaults to False.
            map_chars (bool, optional): a bool indicating if the map_chars function should be used. Defaults to False.
            default_diction (bool, optional): a bool indicating if dictionary with expansion indexes as keys and acronym indexes as keys should be created. Defaults to False.

        Returns:
            dict:
             a dict where the keys are acronyms and the values are a list with the expansion the indexes for the acronym, the indexes for the expansion and the
             technique used for extraction.
        """
        labels = ["O"] * len(sentence)
        diction = {}
        for i, t in enumerate(sentence):
            if i in shorts:
                labels[i] = "B-short"
                if ignore_hyphen:
                    t = t.replace("-", "")
                cand_long, cand_long_index, left = self._extract_cand_long(
                    sentence,
                    t,
                    i,
                    ignore_punc=ignore_punc,
                    add_punc=add_punc,
                    small_window=small_window,
                )
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
                        if lIndex >= -1:
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
                        if t[0].lower() == cand_long[0].lower() or ignore_righthand:
                            while sIndex < len(t):
                                curChar = t[sIndex].lower()
                                if curChar.isdigit() or curChar.isalpha():
                                    while (
                                        (
                                            lIndex < len(cand_long)
                                            and cand_long[lIndex].lower() != curChar
                                        )
                                        or (
                                            ignore_righthand
                                            and (
                                                sIndex == 0
                                                and lIndex > 0
                                                and (
                                                    cand_long[lIndex - 1].isdigit()
                                                    or cand_long[lIndex - 1].isalpha()
                                                )
                                            )
                                        )
                                        or (
                                            lIndex != 0
                                            and cand_long[lIndex - 1] != " "
                                            and " " in cand_long[lIndex:]
                                            and cand_long[
                                                cand_long[lIndex:].index(" ")
                                                + lIndex
                                                + 1
                                            ].lower()
                                            == curChar
                                        )
                                    ):
                                        lIndex += 1
                                        if lIndex >= len(cand_long):
                                            break
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
                    if remove_parentheses:
                        if "(" in long_form or ")" in long_form:
                            long_form = ""
                    long_form = long_form.split()
                    if no_stop_words and long_form:
                        if long_form[0].lower() in stop_words:
                            long_form = []
                    if long_form:
                        if left:
                            long_form_index = cand_long_index[-len(long_form) :]
                        else:
                            long_form_index = cand_long_index[: len(long_form)]
                        first = True
                        for j in range(len(sentence)):
                            if j in long_form_index:
                                if first:
                                    labels[j] = "B-long"
                                    first = False
                                else:
                                    labels[j] = "I-long"
                        if default_diction:
                            diction[(long_form_index[0], long_form_index[-1])] = (i, i)
        return self._create_diction(
            sentence, labels, tag="Schwartz", map_chars=map_chars, diction=diction
        )

    def _bounded_schwartz_extract(
        self,
        sentence,
        shorts,
        remove_parentheses,
        ignore_hyphen=False,
        ignore_punc=False,
        add_punc=False,
        small_window=False,
        no_stop_words=False,
        ignore_righthand=False,
        map_chars=False,
        high_recall=False,
        high_recall_left=False,
        tag="Bounded Schwartz",
        default_diction=False,
    ):
        """Returns a dictionary of acronym-definition pairs present in sentence. If an acronym does not have a definition present in text, the expansion is None.

        This function uses the same rules as schwartz and hearst but for the format "expansion (acronym)" it will select expansions whose last word is used to form the acronym.

        Args:
            sentence (list): a list with a tokenized sentence
            shorts (list): a list with the indexes of acronyms in sentence
            remove_parentheses (bool): a bool indicating if parenthesis should be removed from expansions.
            ignore_hyphen (bool, optional): a bool indicating if hypens in text should be ignored. Defaults to False.
            ignore_punc (bool, optional): bool indicating if punctuation should be ignored in the extraction process (excludes "(" and ")"). Defaults to False.
            add_punc (bool, optional): bool indicating if "=" and ":" should not be considered punctuation when ignore_punc is True. Defaults to False.
            small_window (bool, optional): bool indicating if a small or big window should be considered around the short form when extracting a long form. Defaults to False.
            no_stop_words (bool, optional): a bool indicating if an expansion is allowed to have stop words. Defaults to False.
            ignore_righthand (bool, optional): a bool indicating if what is right of the acronym should be ignored. Defaults to False.
            map_chars (bool, optional): a bool indicating if the map_chars function should be used. Defaults to False.
            high_recall (bool, optional): a bool indicating if high recall extraction should be done. Defaults to False.
            high_recall_left (bool, optional): a bool indicating if the expansion is to the left or right of the acronym. Defaults to False.
            tag (str, optional): a string indicating the techinique used to extract the expansion. Defaults to "Bounded Schwartz".
            default_diction (bool, optional): a bool indicating if dictionary with expansion indexes as keys and acronym indexes as keys should be created. Defaults to False.

        Returns:
            dict:
             a dict where the keys are acronyms and the values are a list with the expansion the indexes for the acronym, the indexes for the expansion and the
             technique used for extraction.
        """

        labels = ["O"] * len(sentence)
        diction = {}
        for i, t in enumerate(sentence):
            if i in shorts:
                labels[i] = "B-short"
                if ignore_hyphen:
                    t = t.replace("-", "")
                if high_recall:
                    (
                        cand_long,
                        cand_long_index,
                        left,
                    ) = self._extract_high_recall_cand_long(
                        sentence, t, i, small_window=small_window, left=high_recall_left
                    )
                else:
                    cand_long, cand_long_index, left = self._extract_cand_long(
                        sentence,
                        t,
                        i,
                        ignore_punc=ignore_punc,
                        add_punc=add_punc,
                        small_window=small_window,
                    )
                cand_long = " ".join(cand_long)
                long_form = ""
                ## findBestLongForm
                if len(cand_long) > 0:
                    if left:
                        sIndex = len(t) - 1
                        lIndex = len(cand_long) - 1
                        first_ind = len(cand_long)
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
                                if first_ind == len(cand_long):
                                    first_ind = lIndex
                                if lIndex < 0:
                                    break
                                lIndex -= 1
                            sIndex -= 1
                        if (
                            lIndex >= 0
                            or lIndex == -1
                            and cand_long[0].lower() == t[0].lower()
                        ):
                            try:
                                lIndex = cand_long.rindex(" ", 0, lIndex + 1) + 1
                                try:
                                    rIndex = (
                                        cand_long[first_ind:].index(" ") + first_ind
                                    )
                                except:
                                    rIndex = len(cand_long)
                            except:
                                lIndex = 0
                                try:
                                    rIndex = (
                                        cand_long[first_ind:].index(" ") + first_ind
                                    )
                                except:
                                    rIndex = len(cand_long)
                            if cand_long:
                                index_map = {}
                                word_ind = 0
                                for ind, c in enumerate(cand_long):
                                    if c == " ":
                                        word_ind += 1
                                    index_map[ind] = word_ind
                                last_word_index = index_map[rIndex - 1]
                                cand_long = cand_long[lIndex:rIndex]
                                long_form = cand_long
                    else:
                        sIndex = 0
                        lIndex = 0
                        first_ind = -1
                        if t[0].lower() == cand_long[0].lower() or ignore_righthand:
                            while sIndex < len(t):
                                curChar = t[sIndex].lower()
                                if curChar.isdigit() or curChar.isalpha():
                                    while (
                                        (
                                            lIndex < len(cand_long)
                                            and cand_long[lIndex].lower() != curChar
                                        )
                                        or (
                                            ignore_righthand
                                            and (
                                                sIndex == 0
                                                and lIndex > 0
                                                and (
                                                    cand_long[lIndex - 1].isdigit()
                                                    or cand_long[lIndex - 1].isalpha()
                                                )
                                            )
                                        )
                                        or (
                                            lIndex != 0
                                            and cand_long[lIndex - 1] != " "
                                            and " " in cand_long[lIndex:]
                                            and cand_long[
                                                cand_long[lIndex:].index(" ")
                                                + lIndex
                                                + 1
                                            ].lower()
                                            == curChar
                                        )
                                    ):
                                        lIndex += 1
                                        if lIndex >= len(cand_long):
                                            break
                                    if first_ind == -1:
                                        first_ind = lIndex
                                    if lIndex >= len(cand_long):
                                        break
                                    lIndex += 1
                                sIndex += 1
                            if lIndex < len(cand_long) or (
                                first_ind < len(cand_long)
                                and lIndex == len(cand_long)
                                and cand_long[-1] == t[-1]
                            ):
                                try:
                                    lIndex = cand_long[lIndex:].index(" ") + lIndex + 1
                                except:
                                    lIndex = len(cand_long)
                                if cand_long:
                                    if not ignore_righthand:
                                        first_ind = 0
                                    index_map = {}
                                    word_ind = 0
                                    for ind, c in enumerate(cand_long):
                                        if c == " ":
                                            word_ind += 1
                                        index_map[ind] = word_ind
                                    first_word_index = index_map[first_ind]
                                    cand_long = cand_long[first_ind:lIndex]
                                    long_form = cand_long
                    if remove_parentheses:
                        if "(" in long_form or ")" in long_form:
                            long_form = ""
                    long_form = long_form.split()
                    if no_stop_words and long_form:
                        if long_form[0].lower() in stop_words:
                            long_form = []
                    if long_form:
                        if left:
                            long_form_index = cand_long_index[
                                last_word_index
                                - len(long_form)
                                + 1 : last_word_index
                                + 1
                            ]
                        else:
                            long_form_index = cand_long_index[
                                first_word_index : first_word_index + len(long_form)
                            ]
                        first = True
                        for j in range(len(sentence)):
                            if j in long_form_index:
                                if first:
                                    labels[j] = "B-long"
                                    first = False
                                else:
                                    labels[j] = "I-long"
                        if default_diction:
                            diction[(long_form_index[0], long_form_index[-1])] = (i, i)
        return self._create_diction(
            sentence, labels, tag=tag, map_chars=map_chars, diction=diction
        )

    def _high_recall_schwartz(
        self,
        sentence,
        shorts,
        remove_parentheses,
        ignore_hyphen=False,
        ignore_punc=False,
        add_punc=False,
        small_window=False,
        no_stop_words=False,
        ignore_righthand=False,
        map_chars=False,
    ):
        """Returns a dictionary of acronym-definition pairs present in sentence. If an acronym does not have a definition present in text, the expansion is None.

        This function uses bounded schwartz rules for acronyms which are not necessarily in parentheses.

        Args:
            sentence (list): a list with a tokenized sentence
            shorts (list): a list with the indexes of acronyms in sentence
            remove_parentheses (bool): a bool indicating if parenthesis should be removed from expansions.
            ignore_hyphen (bool, optional): a bool indicating if hypens in text should be ignored. Defaults to False.
            ignore_punc (bool, optional): bool indicating if punctuation should be ignored in the extraction process (excludes "(" and ")"). Defaults to False.
            add_punc (bool, optional): bool indicating if "=" and ":" should not be considered punctuation when ignore_punc is True. Defaults to False.
            small_window (bool, optional): bool indicating if a small or big window should be considered around the short form when extracting a long form. Defaults to False.
            no_stop_words (bool, optional): a bool indicating if an expansion is allowed to have stop words. Defaults to False.
            ignore_righthand (bool, optional): a bool indicating if what is right of the acronym should be ignored. Defaults to False.
            map_chars (bool, optional): a bool indicating if the map_chars function should be used. Defaults to False.

        Returns:
            dict:
             a dict where the keys are acronyms and the values are a list with the expansion the indexes for the acronym, the indexes for the expansion and the
             technique used for extraction.
        """

        pairs_left = self._bounded_schwartz_extract(
            sentence,
            shorts,
            remove_parentheses,
            ignore_hyphen=True,
            ignore_punc=ignore_punc,
            add_punc=add_punc,
            small_window=small_window,
            no_stop_words=no_stop_words,
            ignore_righthand=ignore_righthand,
            map_chars=True,
            high_recall=True,
            high_recall_left=True,
            tag="High Recall Schwartz",
        )
        pairs_right = self._bounded_schwartz_extract(
            sentence,
            shorts,
            remove_parentheses,
            ignore_hyphen=True,
            ignore_punc=ignore_punc,
            add_punc=add_punc,
            small_window=small_window,
            no_stop_words=no_stop_words,
            ignore_righthand=ignore_righthand,
            map_chars=True,
            high_recall=True,
            high_recall_left=False,
            tag="High Recall Schwartz",
        )
        for acr, lf in pairs_right.items():
            if len(lf[0]) > 0 and (
                acr not in pairs_left or len(pairs_left[acr][0]) == 0
            ):
                pairs_left[acr] = lf
        res = {}
        for acr, lf in pairs_left.items():
            if (
                acr == "".join([w[0] for w in lf[0].split() if w[0].isupper()])
                or acr.lower()
                == "".join(
                    w[0]
                    for w in lf[0].split()
                    if w not in string.punctuation and w not in stop_words
                ).lower()
            ):
                res[acr] = lf
        return res

    def _character_match(
        self,
        acronym,
        long,
        long_index,
        left=False,
        output_string=False,
        is_candidate=True,
    ):
        """Matches the initials of the expansion to the capitals in the acronym in order.

        Args:
            acronym (str): the acronym
            long (list): a list where each element is a word of the expansion
            long_index (list): a list with the indexes of the expansion
            left (bool, optional): a bool indicating if the expansion is to the left of the acronym. Defaults to False.
            output_string (bool, optional): a bool indicating if the long form should be outputed in a string format or in a list containing the indexes. Defaults to False.
            is_candidate (bool, optional): a bool indicating if the long is already a candidate expansion. Defaults to True.

        Returns:
            list: a list with the indexes of the expansion in a sentence or if output_string is True the expansion in string format
        """
        capitals = []
        long_form = []
        for c in acronym:
            if c.isupper():
                capitals.append(c)
        if not is_candidate:
            long_capital_initials = []
            for w in long:
                if w[0].isupper():
                    long_capital_initials.append(w[0])
        if left:
            capitals = capitals[::-1]
            long = long[::-1]
            long_index = long_index[::-1]
        for j, c in enumerate(capitals):
            if j >= len(long):
                long_form = []
                break
            else:
                if long[j][0].lower() == c.lower():
                    long_form.append(long_index[j])
                else:
                    long_form = []
                    break
        if not is_candidate:
            if (
                len(long_capital_initials) != len(long_form)
                and len(long_capital_initials) > 0
            ):
                long_form = []
        long_form.sort()
        if output_string:
            if long_form:
                return long[long_form[0] : long_form[-1] + 1]
            else:
                return ""
        else:
            return long_form

    def _high_recall_character_match(
        self,
        sentence,
        shorts,
        all_acronyms,
        ignore_hyphen=False,
        map_chars=False,
        default_diction=False,
    ):
        """Returns a dictionary of acronym-definition pairs present in sentence. If an acronym does not have a definition present in text, the expansion is None.

        This function finds the expansion for acronyms that are not surrounded by parentheses by using the scrict rule
        of character matching (i.e. the initial of the sequence of the words in the candidate expansion should form the acronym).

        Args:
            sentence (list): a list with a tokenized sentence
            shorts (list): a list with the indexes of acronyms in sentence
            all_acronyms (bool): a bool to indicate if the dictionary should contain all the acronyms (no expansion in text) or not.
            ignore_hyphen (bool, optional): a bool indicating if hypens in text should be ignored. Defaults to False.
            map_chars (bool, optional): a bool indicating if the map_chars function should be used. Defaults to False.
            default_diction (bool, optional): a bool indicating if dictionary with expansion indexes as keys and acronym indexes as keys should be created. Defaults to False.

        Returns:
            dict:
             a dict where the keys are acronyms and the values are a list with the expansion the indexes for the acronym, the indexes for the expansion and the
             technique used for extraction.
        """

        labels = ["O"] * len(sentence)
        diction = {}
        for i, t in enumerate(sentence):
            if i in shorts:
                labels[i] = "B-short"
                if ignore_hyphen:
                    t = t.replace("-", "")
                capitals = []
                for c in t:
                    if c.isupper():
                        capitals.append(c)
                cand_long = sentence[max(i - len(capitals) - 10, 0) : i]
                long_form = ""
                long_form_index = []
                for j in range(max(len(cand_long) - len(capitals), 0)):
                    if "".join(w[0] for w in cand_long[j : j + len(capitals)]) == t:
                        long_form = " ".join(cand_long[j : j + len(capitals)])
                        long_form_index = list(
                            range(
                                max(max(i - len(capitals) - 10, 0) + j, 0),
                                max(max(i - len(capitals) - 10, 0) + j, 0)
                                + len(capitals),
                            )
                        )
                        break
                if not long_form:
                    cand_long = sentence[i + 1 : len(capitals) + i + 10]
                    for j in range(max(len(cand_long) - len(capitals), 0)):
                        if "".join(w[0] for w in cand_long[j : j + len(capitals)]) == t:
                            long_form = " ".join(cand_long[j : j + len(capitals)])
                            long_form_index = list(
                                range(i + 1 + j, i + j + len(capitals) + 1)
                            )
                            break
                long_form = long_form.split()
                if long_form:
                    if long_form[0] in stop_words or long_form[-1] in stop_words:
                        long_form = []
                    if any(lf in string.punctuation for lf in long_form):
                        long_form = []
                    if __name__ != "__main__":
                        NPs = [np.text for np in nlp(" ".join(sentence)).noun_chunks]
                        long_form_str = " ".join(long_form)
                        if all(long_form_str not in np for np in NPs):
                            long_form = []
                if long_form:
                    for j in long_form_index:
                        labels[j] = "I-long"
                    labels[long_form_index[0]] = "B-long"
                    if default_diction:
                        diction[(long_form_index[0], long_form_index[-1])] = (i, i)
        return self._create_diction(
            sentence,
            labels,
            all_acronyms=all_acronyms,
            tag="high recall character match",
            map_chars=map_chars,
            diction=diction,
        )

    def _character_match_extract(
        self,
        sentence,
        shorts,
        all_acronyms,
        check_all_capitals=False,
        ignore_hyphen=False,
        ignore_punc=False,
        map_chars=False,
        default_diction=False,
    ):
        """Returns a dictionary of acronym-definition pairs present in sentence. If an acronym does not have a definition present in text, the expansion is None.

        Args:
            sentence (list): a list with a tokenized sentence
            shorts (list): a list with the indexes of acronyms in sentence
            all_acronyms (bool): a bool to indicate if the dictionary should contain all the acronyms (no expansion in text) or not.
            check_all_capitals (bool, optional): a bool indicating if the number of capitals in an acronym should be compared with a token. Defaults to False.
            ignore_hyphen (bool, optional): a bool indicating if hypens in text should be ignored. Defaults to False.
            ignore_punc (bool, optional): bool indicating if punctuation should be ignored in the extraction process (excludes "(" and ")"). Defaults to False.
            map_chars (bool, optional): a bool indicating if the map_chars function should be used. Defaults to False.
            default_diction (bool, optional): a bool indicating if dictionary with expansion indexes as keys and acronym indexes as keys should be created. Defaults to False.

        Returns:
            dict:
             a dict where the keys are acronyms and the values are a list with the expansion the indexes for the acronym, the indexes for the expansion and the
             technique used for extraction.
        """
        labels = ["O"] * len(sentence)
        diction = {}
        for i, t in enumerate(sentence):
            if i in shorts:
                labels[i] = "B-short"
                if ignore_hyphen:
                    t = t.replace("-", "")
                if check_all_capitals:
                    if len(t) != len([c for c in t if c.isupper()]):
                        continue
                cand_long, cand_long_index, left = self._extract_cand_long(
                    sentence, t, i, ignore_punc=ignore_punc
                )
                long_form = []
                if cand_long:
                    long_form = self._character_match(
                        t, cand_long, cand_long_index, left, is_candidate=True
                    )
                if long_form:
                    labels[long_form[0]] = "B-long"
                    for l in long_form[1:]:
                        labels[l] = "I-long"
                    if default_diction:
                        diction[(long_form[0], long_form[-1])] = (i, i)
        return self._create_diction(
            sentence,
            labels,
            all_acronyms=all_acronyms,
            tag="character match",
            map_chars=map_chars,
            diction=diction,
        )

    def _filterout_roman_numbers(self, diction):
        """Removes acronyms that are roman numerals from the given dictionary.

        Args:
            diction (dict): a dictionary with acronym-expansion pairs

        Returns:
            [dict]:  a dictionary with acronym-expansion pairs with acronym that are roman numerals removed
        """

        acronyms = set(diction.keys())
        for acr in acronyms:
            # instead of all roman acronyms we remove only 1 to 20:
            if acr in [
                "I",
                "II",
                "III",
                "IV",
                "V",
                "VI",
                "VII",
                "VIII",
                "IX",
                "X",
                "XI",
                "XII",
                "XIII",
                "XIV",
                "XV",
                "XVI",
                "XVII",
                "XVIII",
                "XIX",
                "XX",
            ]:
                del diction[acr]
        return diction

    def _remove_punctuations(self, diction):
        """Removes head and trailing punctuations from the expansions in the given dictionary.

        Args:
            diction (dict): a dictionary with acronym-expansion pairs

        Returns:
            [dict]:  a dictionary with acronym-expansion pairs where the expansions do not contain head and trailing punctuations
        """

        for acr, info in diction.items():
            if len(info[0]) > 0:
                if info[0][0] in string.punctuation:
                    info[0] = info[0][2:]
                    info[2][0] = info[2][0] + 1
                    info[3] = "remove punctuation"
            if len(info[0]) > 0:
                if info[0][-1] in string.punctuation:
                    info[0] = info[0][:-2]
                    info[2][1] = info[2][1] - 1
                    info[3] = "remove punctuation"

        return diction

    def _initial_capitals_extract(
        self,
        sentence,
        shorts,
        all_acronyms,
        ignore_hyphen=False,
        map_chars=False,
        default_diction=False,
    ):
        """Returns a dictionary of acronym-definition pairs present in sentence. If an acronym does not have a definition present in text, the expansion is None.

        This function finds expansions for acronyms that are in the format "expansion (acronym)" or "(acronym) expansion"
        and where the capitals in the expansion could form the acronym.

        Args:
            sentence (list): a list with a tokenized sentence
            shorts (list): a list with the indexes of acronyms in sentence
            all_acronyms (bool): a bool to indicate if the dictionary should contain all the acronyms (no expansion in text) or not.
            ignore_hyphen (bool, optional): a bool indicating if hypens in text should be ignored. Defaults to False.
            map_chars (bool, optional): a bool indicating if the map_chars function should be used. Defaults to False.
            default_diction (bool, optional): a bool indicating if dictionary with expansion indexes as keys and acronym indexes as keys should be created. Defaults to False.

        Returns:
            dict:
             a dict where the keys are acronyms and the values are a list with the expansion the indexes for the acronym, the indexes for the expansion and the
             technique used for extraction.
        """

        labels = ["O"] * len(sentence)
        diction = {}
        for i, t in enumerate(sentence):
            if i in shorts:
                labels[i] = "B-short"
                if ignore_hyphen:
                    t = t.replace("-", "")
                capitals = []
                for c in t:
                    if c.isupper():
                        capitals.append(c)
                cand_long, cand_long_index, left = self._extract_cand_long(
                    sentence, t, i
                )
                capital_initials = []
                capital_initials_index = []
                for j, w in enumerate(cand_long):
                    lll = labels[i + j - len(cand_long) - 1]
                    if w[0].isupper() and labels[i + j - len(cand_long) - 1] == "O":
                        capital_initials.append(w[0])
                        capital_initials_index.append(j)
                if "".join(capital_initials) == t:
                    long_form = cand_long[
                        capital_initials_index[0] : capital_initials_index[-1] + 1
                    ]
                    long_form_index = cand_long_index[
                        capital_initials_index[0] : capital_initials_index[-1] + 1
                    ]
                    for lfi in long_form_index:
                        labels[lfi] = "I-long"
                    labels[long_form_index[0]] = "B-long"
                    if default_diction:
                        diction[(long_form_index[0], long_form_index[-1])] = (i, i)
        return self._create_diction(
            sentence,
            labels,
            all_acronyms=all_acronyms,
            tag="Capital Initials",
            map_chars=map_chars,
            diction=diction,
        )

    def _hyphen_in_acronym(self, sentence, shorts):
        """Returns a new list of indexes for acronyms forms where acronyms that have hypens have been merged.

        Args:
            sentence (list): a list with a tokenized sentence
            shorts (list): a list with the indexes of acronyms in sentence

        Returns:
            (list): a list with the indexes of acronyms in sentence but with hypened acronyms merged
        """

        new_shorts = []
        for short in shorts:
            i = short + 1
            next_hyphen = False
            while i < len(sentence) and sentence[i] == "-":
                next_hyphen = True
                i += 1
            j = short - 1
            before_hyphen = False
            while j > 0 and sentence[j] == "-":
                before_hyphen = True
                j -= 1
            if i < len(sentence) and sentence[i].isupper() and next_hyphen:
                for ind in range(short + 1, i + 1):
                    new_shorts += [ind]
            if j > -1 and sentence[j].isupper() and before_hyphen:
                for ind in range(j, short):
                    new_shorts += [ind]

        shorts.extend(new_shorts)
        return shorts

    def _merge_hyphened_acronyms(self, sentence, labels=[]):
        """Returns a new sentence and labels where hyphened acronyms have been merged.

        Args:
            sentence (list): a list with a tokenized sentence
            labels (list, optional): a list where each element is a BIO tag: "B-long", "I-long", "B-short", "I-short", "O". Defaults to [].

        Returns:
            (list): a list with a tokenized sentence where hyphened acronyms have been merged.
            (list): a list where each element is a BIO tag: "B-long", "I-long", "B-short", "I-short", "O".
        """

        new_sentence = []
        new_labels = []
        merge = False
        shorts = self._short_extract(sentence, 0.6, True)
        shorts += self._hyphen_in_acronym(sentence, shorts)

        for i, t in enumerate(sentence):
            if i in shorts and i - 1 in shorts and i + 1 in shorts and t == "-":
                merge = True
                if len(new_sentence) > 0:
                    new_sentence[-1] += "-"
                else:
                    new_sentence += ["-"]
                continue
            if merge:
                if len(new_sentence) > 0:
                    new_sentence[-1] += t
                else:
                    new_sentence += [t]
            else:
                new_sentence.append(t)
                if labels:
                    new_labels.append(labels[i])
            merge = False

        return new_sentence, new_labels

    def _add_embedded_acronym(self, diction, shorts, sentence):
        """Adds acronyms with no expansion to the provided dictionary

        Args:
            diction (dict): a dictionary of acronym-expansion pairs
            shorts (list): a list with the indexes of acronyms in sentence
            sentence (list): a list where each element is a token from a sentence

        Returns:
            dict: a dictionary of acronym-expansion pairs
        """

        short_captured = []
        long_captured = []
        for acr, info in diction.items():
            short_captured.append(info[1][0])
            if info[2]:
                long_captured.extend(list(range(info[2][0], info[2][1])))
        for short in shorts:
            if (
                short not in short_captured
                and short in long_captured
                and sentence[short] not in diction
            ):
                diction[sentence[short]] = ["", (short, short), [], "embedded acronym"]
        return diction

    def _extract_templates(self, sentence, shorts, map_chars=False):
        """Returns a dictionary of acronym-definition pairs present in sentence. If an acronym does not have a definition present in text, the expansion is None.

        Extracts acronym-definition pairs based on templates.

        Args:
            sentence (list): a list with a tokenized sentence
            shorts (list): a list with the indexes of acronyms in sentence
            map_chars (bool, optional):  a bool indicating if the map_chars function should be used. Defaults to False.

        Returns:
            dict:
             a dict where the keys are acronyms and the values are a list with the expansion the indexes for the acronym, the indexes for the expansion and the
             technique used for extraction.
        """

        labels = ["O"] * len(sentence)
        for i, t in enumerate(sentence):
            if i in shorts:
                labels[i] = "B-short"
                capitals = []
                for c in t:
                    if c.isupper():
                        capitals.append(c)
                if i < len(sentence) - len(capitals) - 2:
                    if sentence[i + 1] == "stands" and sentence[i + 2] == "for":
                        if "".join(
                            w[0] for w in sentence[i + 3 : i + 3 + len(capitals)]
                        ) == "".join(capitals):
                            labels[i + 3 : i + 3 + len(capitals)] = ["I-long"] * len(
                                capitals
                            )
                            labels[i + 3] = "B-long"
        return self._create_diction(
            sentence, labels, all_acronyms=False, tag="Template", map_chars=map_chars
        )

    def _update_pair(self, old_pair, new_pair):
        """Updates old_pair dicionary with acronym-expansion pairs with the new_pair dicionary.

        Args:
            old_pair (dict): a dict where the keys are acronyms and the values are a list with the expansion the indexes for the acronym, the indexes for the expansion and the
             technique used for extraction.
            new_pair (dict): a dict where the keys are acronyms and the values are a list with the expansion the indexes for the acronym, the indexes for the expansion and the
             technique used for extraction.

        Returns:
            dict:
             a dict where the keys are acronyms and the values are a list with the expansion the indexes for the acronym, the indexes for the expansion and the
             technique used for extraction.
        """
        for acr, info in new_pair.items():
            if acr not in old_pair:
                old_pair[acr] = info
            else:
                info[4] = max(info[4], old_pair[acr][4])
                old_pair[acr] = info
        return old_pair

    def _extract(self, sentence, active_rules):
        """Returns a dictionary of acronym-definition pairs present in sentence. If an acronym does not have a definition present in text, the expansion is None.

        Args:
            sentence (list): a list with a tokenized sentence
            active_rules (dict): a dictionary where the keys are rules (str) and the values are booleans indicating if the rule should be used or not.

        Returns:
            dict:
             a dict where the keys are acronyms and the values are a list with the expansion the indexes for the acronym, the indexes for the expansion and the
             technique used for extraction.
        """
        shorts = self._short_extract(
            sentence,
            0.6,
            active_rules["starting_lower_case"],
            ignore_dot=active_rules["ignore_dot"],
        )
        if active_rules["low_short_threshold"]:
            shorts += self._short_extract(
                sentence,
                0.50,
                active_rules["starting_lower_case"],
                ignore_dot=active_rules["ignore_dot"],
            )
        if active_rules["hyphen_in_acronym"]:
            shorts += self._hyphen_in_acronym(sentence, shorts)
        pairs = {}
        if active_rules["schwartz"]:
            pairs = self._schwartz_extract(
                sentence,
                shorts,
                active_rules["no_parentheses"],
                ignore_punc=active_rules["ignore_punc_in_parentheses"],
                add_punc=active_rules["extend_punc"],
                small_window=active_rules["small_window"],
                no_stop_words=active_rules["no_beginning_stop_word"],
                ignore_righthand=active_rules["ignore_right_hand"],
                map_chars=active_rules["map_chars"],
                default_diction=active_rules["default_diction"],
            )
        if active_rules["bounded_schwartz"]:
            bounded_pairs = self._bounded_schwartz_extract(
                sentence,
                shorts,
                active_rules["no_parentheses"],
                ignore_punc=active_rules["ignore_punc_in_parentheses"],
                add_punc=active_rules["extend_punc"],
                small_window=active_rules["small_window"],
                no_stop_words=active_rules["no_beginning_stop_word"],
                ignore_righthand=active_rules["ignore_right_hand"],
                map_chars=active_rules["map_chars"],
                default_diction=active_rules["default_diction"],
            )
            pairs = self._update_pair(pairs, bounded_pairs)
        if active_rules["high_recall_schwartz"]:
            hr_paris = self._high_recall_schwartz(
                sentence,
                shorts,
                active_rules["no_parentheses"],
                ignore_punc=active_rules["ignore_punc_in_parentheses"],
                add_punc=active_rules["extend_punc"],
                small_window=active_rules["small_window"],
                no_stop_words=active_rules["no_beginning_stop_word"],
                ignore_righthand=active_rules["ignore_right_hand"],
                map_chars=active_rules["map_chars"],
            )
            pairs = self._update_pair(pairs, hr_paris)
        if active_rules["character"]:
            character_pairs = self._character_match_extract(
                sentence,
                shorts,
                not active_rules["schwartz"],
                check_all_capitals=active_rules["check_all_capitals"],
                ignore_punc=active_rules["ignore_punc_in_parentheses"],
                map_chars=active_rules["map_chars"],
                default_diction=active_rules["default_diction"],
            )
            pairs = self._update_pair(pairs, character_pairs)
        if active_rules["high_recall_character_match"]:
            character_pairs = self._high_recall_character_match(
                sentence,
                shorts,
                not active_rules["schwartz"],
                map_chars=active_rules["map_chars"],
                default_diction=active_rules["default_diction"],
            )
            acronyms = character_pairs.keys()
            for acr in acronyms:
                if acr not in pairs or len(pairs[acr][0]) == 0:
                    pairs[acr] = character_pairs[acr]
        if active_rules["initial_capitals"]:
            character_pairs = self._initial_capitals_extract(
                sentence,
                shorts,
                not active_rules["schwartz"],
                map_chars=active_rules["map_chars"],
                default_diction=active_rules["default_diction"],
            )
            pairs = self._update_pair(pairs, character_pairs)
        if active_rules["template"]:
            template_pairs = self._extract_templates(
                sentence, shorts, map_chars=active_rules["map_chars"]
            )
            pairs = self._update_pair(pairs, template_pairs)
        if active_rules["capture_embedded_acronym"]:
            pairs = self._add_embedded_acronym(pairs, shorts, sentence)
        if active_rules["roman"]:
            pairs = self._filterout_roman_numbers(pairs)
        if active_rules["remove_punctuation"]:
            pairs = self._remove_punctuations(pairs)
        return pairs

    def _deep_strip(self, text):
        """Tokenizes the input text

        This portion of the code was present in the original MadDog, specifically in the server.py and predict.py files.

        Args:
            text (str): the text to be tokenized

        Returns:
            list: the tokenized text
        """

        new_text = ""
        for c in text:
            if len(c.strip()) > 0:
                new_text += c
            else:
                new_text += " "
        new_text = new_text.replace('"', "'")
        tokens = [t.text for t in nlp(new_text) if len(t.text.strip()) > 0]
        return tokens

    def _clean_up_pairs(self, acro_exp_pairs):
        """Removes everything but the expansion for an acronym from the given acro_exp_pairs dictionary.

        Args:
            acro_exp_pairs (dict):
             a dict where the keys are acronyms and the values are a list with the expansion the indexes for the acronym, the indexes for the expansion and the
             technique used for extraction.

        Returns:
            dict: a dictionary where the keys are acronyms and the values are the expansions. If the acronym does not have an expansion the value is None.
        """

        for acro in acro_exp_pairs.keys():
            if acro_exp_pairs[acro][0] == "":
                acro_exp_pairs[acro] = None
            else:
                acro_exp_pairs[acro] = acro_exp_pairs[acro][0]
        return acro_exp_pairs

    def _remove_acr_no_exp(self, acro_exp_pairs):
        """Removes acronym with no expansion from the given acro_exp_pairs dictionary.

        Args:
            acro_exp_pairs (dict): a dictionary where the keys are acronyms and the values are the expansions. If the acronym does not have an expansion the value is None.

        Returns:
            dict: a dictionary where the keys are acronyms and the values are the expansions.
        """

        new_acro_exp_pairs = {}
        for acro in acro_exp_pairs.keys():
            if acro_exp_pairs[acro] != None:
                new_acro_exp_pairs[acro] = acro_exp_pairs[acro]
        return new_acro_exp_pairs

    def get_all_acronym_expansion(self, text):
        """Returns a dicionary where each key is an acronym (str) and each value is an expansion (str). The expansion is None if no expansion is found.

        Args:
            text (str): the text to extract acronym-expansion pairs from

        Returns:
            dict: a dicionary where each key is an acronym (str) and each value is an expansion (str). The expansion is None if no expansion is found.
        """
        tokens = self._deep_strip(text)
        if constant.RULES["merge_hyphened_acronyms"]:
            tokens, _ = self._merge_hyphened_acronyms(tokens)
        acro_expansion_pairs = self._extract(tokens, constant.RULES)
        acro_expansion_pairs = self._clean_up_pairs(acro_expansion_pairs)
        return acro_expansion_pairs

    def get_acronym_expansion_pairs(self, text):
        """Returns a dicionary where each key is an acronym (str) and each value is an expansion (str).

        Args:
            text (str): the text to extract acronym-expansion pairs from

        Returns:
            dict: a dicionary where each key is an acronym (str) and each value is an expansion (str).
        """
        tokens = self._deep_strip(text)
        if constant.RULES["merge_hyphened_acronyms"]:
            tokens, _ = self._merge_hyphened_acronyms(tokens)
        acro_expansion_pairs = self._extract(tokens, constant.RULES)
        acro_expansion_pairs = self._clean_up_pairs(acro_expansion_pairs)
        acro_expansion_pairs = self._remove_acr_no_exp(acro_expansion_pairs)
        return acro_expansion_pairs


if __name__ == "__main__":
    acroExp = AcroExpExtractor_MadDog()

    r = acroExp.get_acronym_expansion_pairs(
        "A relational database is a digital database based on the relational model of data, as proposed by E. F. Codd in 1970. A software system used to maintain relational databases is a relational database management system (RDBMS). Many relational database systems have an option of using the SQL for querying and maintaining the database."
    )
    print(r)

    r = acroExp.get_acronym_expansion_pairs(
        "Interaction between Set1p and checkpoint protein Mec3p in DNA repair and telomere functions.\nThe yeast protein Set1p, inactivation of which alleviates telomeric position effect (TPE), contains a conserved SET domain present in chromosomal proteins involved in epigenetic control of transcription. Mec3p is required for efficient DNA-damage-dependent checkpoints at G1/S, intra-S and G2/M (refs 3-7). We show here that the SET domain of Set1p interacts with Mec3p. Deletion of SET1 increases the viability of mec3delta mutants after DNA damage (in a process that is mostly independent of Rad53p kinase, which has a central role in checkpoint control) but does not significantly affect cell-cycle progression. Deletion of MEC3 enhances TPE and attenuates the Set1delta-induced silencing defect. Furthermore, restoration of TPE in a Set1delta mutant by overexpression of the isolated SET domain requires Mec3p. Finally, deletion of MEC3 results in telomere elongation, whereas cells with deletions of both SET1 and MEC3 do not have elongated telomeres. Our findings indicate that interactions between SET1 and MEC3 have a role in DNA repair and telomere function."
    )
    print(r)

    r = acroExp.get_acronym_expansion_pairs(
        "Topology and functional domains of the yeast pore membrane protein Pom152p.\nIntegral membrane proteins associated with the nuclear pore complex (NPC) are likely to play an important role in the biogenesis of this structure. Here we have examined the functional roles of domains of the yeast pore membrane protein Pom152p in establishing its topology and its interactions with other NPC proteins. The topology of Pom152p was evaluated by alkaline extraction, protease protection, and endoglycosidase H sensitivity assays. The results of these experiments suggest that Pom152p contains a single transmembrane segment with its N terminus (amino acid residues 1-175) extending into the nuclear pore and its C terminus (amino acid residues 196-1337) positioned in the lumen of the nuclear envelope. The functional role of these different domains was investigated in mutants that are dependent on Pom152p for viability. The requirement for Pom152p in strains containing mutations allelic to the NPC protein genes NIC96 and NUP59 could be alleviated by Pom152p's N terminus, independent of its integration into the membrane. However, complementation of a mutation in NUP170 required both the N terminus and the transmembrane segment. Furthermore, mutations in NUP188 were rescued only by full-length Pom152p, suggesting that the lumenal structures play an important role in the function of pore-side NPC structures."
    )

    print(r)
    text = """
2019–20 Columbus Blue Jackets season

The 2019–20 Columbus Blue Jackets season is the 20th season for the National Hockey League franchise that was established on June 25, 1997.

The preseason schedule was published on June 18, 2019. The September 29 game between the Blue Jackets and the St. Louis Blues was cancelled due to issues with the team's flight.
The regular season schedule was published on June 25, 2019.

Denotes player spent time with another team before joining the Blue Jackets. Stats reflect time with the Blue Jackets only.
Denotes player was traded mid-season. Stats reflect time with the Blue Jackets only.
Bold/italics denotes franchise record.
    """

    r = acroExp.get_acronym_expansion_pairs(text)
    print(r)

    text = """
Goh Keng Swee Command and Staff College                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                               
The Goh Keng Swee Command and Staff College (GKS CSC) is one of five officer schools of the SAFTI Military Institute of the Singapore Armed Forces (SAF).                                                                                                                      
                                                                                                                                                                                                                                                                               
Formerly known as the Singapore Command and Staff College (SCSC), the inaugural Command and Staff Course commenced in 1969 and the College was officially opened in February 1970 by the Prime Minister of Singapore, Lee Kuan Yew, at its birthplace at Fort Canning. One of i
ts first commanders was Lieutenant Colonel Ronald Wee Soon Whatt. After relocating to Marina Hill in the seventies and Seletar Camp in the eighties, it finally moved into its present premises in SAFTI in 1995. It was later named after Goh Keng Swee in 2011.              
                                                                                                                                                                                                                                                                               
The College conducts the Command and Staff Course (CSC) for career officers and the National Service Command and Staff Course (NSCSC) for selected reserve officers who have demonstrated potential for higher command and staff appointments in the SAF. Annually, a number of
 International Officers from the region are invited to attend the 10.5-month-long CSC. In 2009, students from Indonesia, South Korea, Malaysia, Philippines, China, Thailand, Brunei, India, Vietnam, New Zealand, Australia and the United States, attended the course.       
                                                                                                                                                                                                                                                                               
The GKSCSC vision is: "World Class College, First Class Experience."      
    """
    r = acroExp.get_acronym_expansion_pairs(text)
    print(r)
    # print(algorithm(["/home/jpereira/Downloads/yeast_abbrev_unlabeled.txt"]))
    text = """Dennis Sullivan

Dennis Parnell Sullivan (born February 12, 1941) is an American mathematician. He is known for work in topology, both algebraic and geometric, and on dynamical systems. He holds the Albert Einstein Chair at the City University of New York Graduate Center, and is a professor at Stony Brook University.

He received his B.A. in 1963 from Rice University and his doctorate in 1966 from Princeton University. His Ph.D. thesis, entitled "Triangulating homotopy equivalences", was written under the supervision of William Browder, and was a contribution to surgery theory. He was a permanent member of the Institut des Hautes Études Scientifiques from 1974 to 1997.

Sullivan is one of the founders of the surgery method of classifying high-dimensional manifolds, along with Browder, Sergei Novikov and C. T. C. Wall. In homotopy theory, Sullivan put forward the radical concept that spaces could directly be "localised", a procedure hitherto applied to the algebraic constructs made from them. He founded (along with Daniel Quillen) rational homotopy theory.

The Sullivan conjecture, proved in its original form by Haynes Miller, states that the classifying space "BG" of a finite group "G" is sufficiently different from any finite CW complex "X", that it maps to such an "X" only 'with difficulty'; in a more formal statement, the space of all mappings "BG" to "X", as pointed spaces and given the compact-open topology, is weakly contractible. This area has generated considerable further research. (Both these matters are discussed in his 1970 MIT notes.)

In 1985, he proved the No wandering domain theorem. The Parry–Sullivan invariant is named after him and the English mathematician Bill Parry.

In 1987, he proved Thurston's conjecture about the approximation
of the Riemann map by circle packings together with Burton Rodin.
"""
    r = acroExp.get_all_acronym_expansion(text)
    print(r)
