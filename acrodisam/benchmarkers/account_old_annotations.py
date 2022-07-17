from nltk.metrics.distance import edit_distance
from nltk.tokenize.regexp import RegexpTokenizer

tokenizer = RegexpTokenizer(r"\w+")


# this function was not used when calculating user performance metrics
def account_for_older_annotations(
    first_part_acro, second_part_acro, predicted_expansions, actual_expansion
):
    """Transforms annotations of the old formart (first 68 articles) into new format

    Args:
        first_part_acro (str): first part of an acronym when splitted by "/" or "-"
        second_part_acro (str): second part of an acronym when splitted by "/" or "-"
        predicted_expansions (dict): dictionary with the an user's annotations for an article
        actual_expansion (str): the actual expansion of the acronym

    Returns:
        dict: a dictionary with the first and second part of the acronym popped
        int: an integer that can be either 0 or 1 indicating if it's a true positive
        int: an integer that can be either 0 or 1 indicating if it's a false positive
        int: an integer that can be either 0 or 1 indicating if it's a false negative
    """

    tp = 0
    fp = 0
    fn = 0

    actual_expansion_tk = tokenizer.tokenize(actual_expansion)

    if first_part_acro.isnumeric():
        second_part_exp = predicted_expansions.pop(second_part_acro, None)
        if second_part_exp != None:
            second_part_exp = tokenizer.tokenize(second_part_exp)
            second_part_exp.reverse()
            second_part_exp = " ".join(second_part_exp)
            actual_expansion_tk.reverse()
            tmp_actual_expansion = ""
            fp = 1
            # iterating in reverse so it's easier to compare
            for token in actual_expansion_tk:
                tmp_actual_expansion += token
                if edit_distance(tmp_actual_expansion, second_part_exp) <= 2:
                    tp = 1
                    fp = 0
                    break
                tmp_actual_expansion += " "
        else:
            fn = 1
    elif second_part_acro.isnumeric():
        first_part_exp = predicted_expansions.pop(first_part_acro, None)
        if first_part_exp != None:
            tmp_actual_expansion = ""
            fp = 1
            for token in actual_expansion_tk:
                tmp_actual_expansion += token
                if edit_distance(tmp_actual_expansion, first_part_exp) <= 2:
                    tp = 1
                    fp = 0
                    break
                tmp_actual_expansion += " "
        else:
            fn = 1
    else:
        first_part_exp = predicted_expansions.pop(first_part_acro, None)
        second_part_exp = predicted_expansions.pop(second_part_acro, None)
        if first_part_exp != None and second_part_exp != None:
            # checking if first part is in actual expansion
            tmp_actual_expansion_first = ""
            fp = 1
            match = False
            old_distance = 0
            for token in actual_expansion_tk:
                tmp_actual_expansion_first += token
                if (
                    not match
                    and edit_distance(tmp_actual_expansion_first, first_part_exp) <= 2
                ):
                    tp = 1
                    fp = 0
                    match = True
                    old_distance = edit_distance(
                        tmp_actual_expansion_first, first_part_exp
                    )
                if match:
                    if (
                        edit_distance(tmp_actual_expansion_first, first_part_exp)
                        > old_distance
                    ):
                        # removing the last token and whitespace before it
                        tmp_actual_expansion_first = tmp_actual_expansion_first[
                            : len(tmp_actual_expansion_first) - len(token) - 1
                        ]
                        break
                    else:
                        old_distance = edit_distance(
                            tmp_actual_expansion_first, first_part_exp
                        )
                tmp_actual_expansion_first += " "
            if tp == 1 and fp == 0:
                second_part_exp = tokenizer.tokenize(second_part_exp)
                second_part_exp.reverse()
                second_part_exp = " ".join(second_part_exp)
                actual_expansion_tk.reverse()
                tmp_actual_expansion_second = ""
                fp = 1
                match = False
                old_distance = 0
                # now checking if second part is in actual expansion
                for token in actual_expansion_tk:
                    tmp_actual_expansion_second += token
                    if (
                        not match
                        and edit_distance(tmp_actual_expansion_second, second_part_exp)
                        <= 2
                    ):
                        tp = 1
                        fp = 0
                        match = True
                        old_distance = edit_distance(
                            tmp_actual_expansion_second, second_part_exp
                        )

                    if match:
                        if (
                            edit_distance(tmp_actual_expansion_second, second_part_exp)
                            > old_distance
                        ):
                            tmp_actual_expansion_second = tmp_actual_expansion_second[
                                : len(tmp_actual_expansion_second) - len(token) - 1
                            ]
                            break
                        else:
                            old_distance = edit_distance(
                                tmp_actual_expansion_second, second_part_exp
                            )
                    tmp_actual_expansion_second += " "
                # finally we check if the maximum number of tokens
                # between the first and second part is no more than 2
                if (
                    len(actual_expansion_tk)
                    - (
                        len(tokenizer.tokenize(tmp_actual_expansion_first.strip()))
                        + len(tokenizer.tokenize(tmp_actual_expansion_second.strip()))
                    )
                ) > 2:
                    fp = 1
                    tp = 0
        else:
            fn = 1

    return predicted_expansions, tp, fp, fn