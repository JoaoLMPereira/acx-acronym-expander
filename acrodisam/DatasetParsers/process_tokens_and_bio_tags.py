"""Contains functions to handle tokens and bio tags

The function create_diction allows to convert a text that is annotated in BIO tags of the form: "B-long", "I-long", "B-short", "I-short", "O"
to a dicionary where each key is an acronym and each value is an expansion. The function tokens_to_raw_text detokenizes a list with tokens into raw text

Code modified from original code of Veyseh and Franck, available at: https://github.com/amirveyseh/MadDog

Created May, 2021

@author:JRCasanova
"""

import re


def _match_capital_initials(acronym, long):
    """Match capitals in acronym to initial capitals in the expansion in order.

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


def _ratio_match_initials(acronym, long, threshold):
    """Matches the acronym to the expansion based on the number of initials of the expansion that are in the acronym.

    Evaluates if the number of initials of the expansion that are in the acronym are above a certain threshold.
    Order of the mapping between the initials of the expansion and the characters of the acronym is not checked.

    Args:
        acronym (str): the acronym
        long (list): a list where each element is a word of the expansion
        threshold (float): a threshold for the number of initials of the expansion that must be in the acronym

    Returns:
        bool:
         True if the number of initials of the expansion that are in the acronym are above the given thereshold and False otherwise
    """

    initials = [w[0].lower() for w in long]
    ratio = len([c for c in initials if c in acronym]) / len(initials)
    return ratio >= threshold


def _match_initials_capital_acro(acronym, long):
    """Matches the initials of the expansion to the capitals in the acronym in order.

    Args:
        acronym (str): the acronym
        long (list): a list where each element is a word of the expansion

    Returns:
        bool: True if the acronym is matched to the expansion and False otherwise
    """

    capitals = []
    long_form = []
    for c in acronym:
        if c.isupper():
            capitals.append(c)
    long_capital_initials = []
    for w in long:
        if w[0].isupper():
            long_capital_initials.append(w[0])
    for j, c in enumerate(capitals):
        if j >= len(long):
            return False
        else:
            if long[j][0].lower() == c.lower():
                long_form.append(long[j])
            else:
                return False
    if len(long_capital_initials) != len(long_form) and len(long_capital_initials) > 0:
        return False
    return True


def _find_best_acro_for_exp(
    exp_indexes, acro_list, sentence, labels, threshold=0.6, ratio_match=True
):
    """Tries to find the best acronym for an expansion.

    Args:
        exp_indexes (tuple):
         a tuple that has two positions and holds the indexes where the expansion starts and ends
        acro_list (list):
         a list where each element is a tuple that holds the indexes where an acronym starts and ends
        sentence (list):
         a list where each element is a token from a sentence
        labels (list):
         a list where each element is a BIO tag: "B-long", "I-long", "B-short", "I-short", "O"
        threshold (float, optional):
         a threshold for the ratio of initials of the expansion that must be in the acronym. If the ratio is above the threshold
         the acronym and expansion are matched. To be used when ratio_match is set to True. Defaults to 0.6.
        ratio_match (bool, optional):
         a bool to indicate if the ratio of initials of the expansion that are in the acronym is to be taken into account when
         trying to find the best acronym for an expansion. Defaults to True.

    Returns:
        tuple:
         a tuple that has two positions and holds the indexes where the acronym that matches the expansion starts and ends. If no acronym is found an empty tuple is returned
    """

    exp = sentence[exp_indexes[0] : exp_indexes[1] + 1]

    for acro_indexes in acro_list:
        acronym = " ".join(sentence[acro_indexes[0] : acro_indexes[1] + 1])

        if _match_initials_capital_acro(acronym, exp):
            return acro_indexes

    for acro_indexes in acro_list:
        acronym = " ".join(sentence[acro_indexes[0] : acro_indexes[1] + 1])

        if _match_capital_initials(acronym, exp):
            return acro_indexes

    if ratio_match:
        for acro_indexes in acro_list:
            acronym = " ".join(sentence[acro_indexes[0] : acro_indexes[1] + 1])

            if _ratio_match_initials(acronym, exp, threshold):
                return acro_indexes

    if (
        exp_indexes[1] < len(sentence) - 2
        and sentence[exp_indexes[1] + 1] == "("
        and "short" in labels[exp_indexes[1] + 2]
    ):
        for acro_indexes in acro_list:
            if acro_indexes[0] == exp_indexes[1] + 2:
                return acro_indexes

    if (
        exp_indexes[0] > 1
        and sentence[exp_indexes[0] - 1] == "("
        and "short" in labels[exp_indexes[0] - 2]
    ):
        for acro_indexes in acro_list:
            if acro_indexes[1] == exp_indexes[0] - 2:
                return acro_indexes

    for acro_indexes in acro_list:
        if acro_indexes[0] > exp_indexes[1]:
            dist = acro_indexes[0] - exp_indexes[1]
        else:
            dist = exp_indexes[0] - acro_indexes[1]
        if dist < 3:
            return acro_indexes

    return ()


def create_diction(
    sentence, labels, all_acronyms=True, ratio_match=True, threshold=0.6
):
    """Creates a dictionary with acronyms as keys and expansions as values from text annotated with BIO tags.

    Args:
        sentence (list): a list where each element is a token from a sentence
        labels (list): a list where each element is a BIO tag: "B-long", "I-long", "B-short", "I-short", "O"
        all_acronyms (bool, optional):
         a bool to indicate if the dictionary should contain all the acronyms (no expansion in text) or not. Defaults to True.
        ratio_match (bool, optional):
         a bool to indicate if the ratio of initials of the expansion that are in the acronym is to be taken into account when
         trying to find the best acronym for an expansion. Defaults to True.
        threshold (float, optional):
         a threshold for the ratio of initials of the expansion that must be in the acronym. If the ratio is above the threshold
         the acronym and expansion are matched. To be used when ratio_match is set to True. Defaults to 0.6.

    Returns:
        dict: a dict where the keys are acronyms and the values are expansions
    """

    shorts = []
    longs = []
    isShort = True
    label_indexes = []
    for i in range(len(sentence)):
        if (
            labels[i] == "O"
            or (isShort and "long" in labels[i])
            or (not isShort and "short" in labels[i])
            or (labels[i].startswith("B"))
        ):
            if len(label_indexes):
                if isShort:
                    shorts.append((label_indexes[0], label_indexes[-1]))
                else:
                    longs.append((label_indexes[0], label_indexes[-1]))
                label_indexes = []
        if "short" in labels[i]:
            isShort = True
            label_indexes.append(i)
        if "long" in labels[i]:
            isShort = False
            label_indexes.append(i)
    if len(label_indexes):
        if isShort:
            shorts.append((label_indexes[0], label_indexes[-1]))
        else:
            longs.append((label_indexes[0], label_indexes[-1]))
    acr_long = {}
    for long in longs:
        best_short = _find_best_acro_for_exp(
            long, shorts, sentence, labels, threshold, ratio_match
        )

        if best_short:
            acr = " ".join(sentence[best_short[0] : best_short[1] + 1])
            exp = " ".join(sentence[long[0] : long[1] + 1])
            if acr not in acr_long:
                acr_long[acr] = exp
    if all_acronyms:
        for short in shorts:
            acr = " ".join(sentence[short[0] : short[1] + 1])
            if acr not in acr_long:
                acr_long[acr] = None
    return acr_long


def tokens_to_raw_text(tokens):
    """Tranforms a list with tokens into raw text

    Args:
        tokens (list): a list where each element is a token from a sentence

    Returns:
        str: a string that is the raw text from the tokens
    """

    sentence = " ".join(tokens)

    # handling punctuation with wrong spaces
    sentence = (
        sentence.replace(" ,", ",")
        .replace(" .", ".")
        .replace(" !", "!")
        .replace(" ?", "?")
        .replace(" :", ":")
        .replace(" - ", "-")
        .replace(" / ", "/")
        .replace(" ;", ";")
        .replace("( ", "(")
        .replace(" )", ")")
        .replace("[ ", "[")
        .replace(" ]", "]")
    )

    # handling phrases that are in quotes
    # From: " hello there "
    # To: "hello there"
    sentence = re.sub(r'" (.*) "', r'"\1"', sentence)

    # same but for the case when quotes are done with ''
    sentence = re.sub(r"' (.*) '", r"'\1'", sentence)

    # fixing possessive case with extra space
    # From: the AI 's goal
    # To: the AI's goal
    sentence = sentence.replace(" 's", "'s")

    return sentence


def bioless_to_bio(tags):
    """Receives a list with tags of type "long","short" or "O" and transforms it into a list of BIO tags.

    The BIO tag format is comprised of three tags: "B"(beginning), "I"(inside), "O"(outside).
    In this case the input tags "long","short" and "O" are turned into: "B-long", "I-long", "B-short", "I-short", and "O" respectively.

    For example:
        input list: ["long","long","long","long","O","short","short","short","O"]
        output list: ["B-long","I-long","I-long","I-long","O","B-short","I-short","I-short","O"]

    Args:
        tags (list): a list of tags that are either "long","short" or "O"

    Returns:
        list: a list of tags derived from the input list. The list returned contains BIO tags.
    """
    fixed = []
    cont = None
    for tag in tags:
        if tag == "O":
            fixed.append(tag)
            cont = None
        else:
            if cont == tag:
                fixed.append("I-" + tag)
            else:
                fixed.append("B-" + tag)
                cont = tag
    return fixed


def biouless_to_bio(tags):
    """Receives a list with tags of type "U-long","U-short" or "O" and transforms it into a list of BIO tags.

    The BIO tag format is comprised of three tags: "B"(beginning), "I"(inside), "O"(outside).

    For example:
        input list: ["U-long","U-long","U-long","O","O","U-short","U-short","U-short","O"]
        output list: ["B-long","I-long","I-long","O","O","B-short","I-short","I-short","O"]

    Args:
        tags (list): a list of tags that are in the BIOUL format

    Returns:
        list: a list of tags derived from the input list. The list returned contains BIO tags.
    """
    fixed = []
    cont = None
    for tag in tags:
        if tag == "O":
            fixed.append(tag)
            cont = None
        else:
            if cont == tag:
                fixed.append(tag.replace("U", "I"))
            else:
                fixed.append(tag.replace("U", "B"))
                cont = tag
    return fixed


def bioul_to_bio(tags):
    """Receives a list with tags of type BIOUL and transforms it into a list of BIO tags.

    The BIOUL tag format is comprised of five tags: B(beginning), I(inside), O(outside), U(unit), "L"(last).
    The BIO tag format is comprised of three tags: "B"(beginning), "I"(inside), "O"(outside).

    For example:
        input list: ["B-long","I-long","L-long","O","O","B-short","I-short","L-short","O", "U-short"]
        output list: ["B-long","I-long","I-long","O","O","B-short","I-short","I-short","O", "B-short"]

    Args:
        tags (list): a list of tags that are in the BIOUL format

    Returns:
        list: a list of tags derived from the input list. The list returned contains BIO tags.
    """
    fixed_tags_dict = {
        "U-short": "B-short",
        "U-long": "B-long",
        "L-short": "I-short",
        "L-long": "I-long",
    }
    return [fixed_tags_dict.get(tag, tag) for tag in tags]
