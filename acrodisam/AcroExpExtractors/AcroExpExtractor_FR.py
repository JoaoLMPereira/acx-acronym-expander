"""Custom Acronym Exp Extractor Optimized to extract
in-expansion in a french text.

@author : maxime prieur
"""

import re
import nltk
from helper import get_lang_dict
from string_constants import FR_MAPPING, APOSTROPHE_LETTERS, FR_PREPOSITIONS as PREPOSITIONS, \
    SEPARATOR_CHAR, ROMAN_NUMERALS, FR_VOWELS as VOWELS
from AcroExpExtractors.AcroExpExtractor_Original_Schwartz_HearstFR\
    import AcroExpExtractor_Original_Schwartz_HearstFR


FRENCH_WORDS = get_lang_dict("FR")
TOKENIZER = nltk.load('tokenizers/punkt/french.pickle')

class AcroExpExtractor_FR(AcroExpExtractor_Original_Schwartz_HearstFR):
    """ Acronym Expansion Extractor using custom heuristics and the
        original Schwartz and Hearst Algorithm.
    """

    def __init__(self):
        AcroExpExtractor_Original_Schwartz_HearstFR.__init__(self)


    def is_roman_numeral(self, word):
        """Check if the word is a Roman numeral.
            Args :
                word (str) : the word to check
            Returns :
                (boolean)
        """
        for letter in word:
            if letter not in ROMAN_NUMERALS:
                return False
        return True


    def fr_special_char_replacer(self, text):
        """Pre-process the input text to improve extraction
            by replacing french special chars as 'ç', 'é'...
            Args:
                text (str): the input text
            Returns:
                new_text (str): the processed text
        """
        new_text = ""
        for letter in text:
            new_text = new_text+FR_MAPPING.get(letter, letter)
        return new_text


    def text_to_tokenized_list(self, text):
        """Split a given text into a list of sentences tokenized into words.
            Args :
                text (str) : the text to split
            Returns :
                sentences (list) : the splitted text
        """
        tokenized_text = TOKENIZER.tokenize(text)
        sentences = []
        for token in tokenized_text:
            sentences.extend(token.split('\n\n'))
        sentences = [sentence.replace('\n', '') for sentence in sentences]
        return sentences


    def can_be_acronym(self, word):
        """Verify if a word is a possible acronym.
            Args :
                word (str) : the word to process
            Returns :
                (boolean)
        """
        return "'" not in word\
           and not word.isdigit()\
           and (word.isupper()\
                or (word.lower() not in FRENCH_WORDS\
                    and word.lower() not in PREPOSITIONS)\
                    and len(word) > 1)\
           and word not in SEPARATOR_CHAR\
           and word not in APOSTROPHE_LETTERS\
           and self.has_letter(word)\
           and not self.is_roman_numeral(word)


    def get_acronym_candidates(self, words):
        """Select the possible acronym in a sentence.
            Args:
                words (list): the words from the sentence
            Returns:
                possible_acronyms (list): the candidates as acronyms
        """
        possible_acronyms = [word for word in words if self.can_be_acronym(word)]
        return possible_acronyms


    def tokenization(self, sentence):
        """Split the sentence into token which are either words or non letter char.
            Args:
                sentence (str): the sentence to split
            Returns:
                words (list) : the splitted sentence
        """
        words = []
        count = 0
        sentence = re.sub('\s[A-Z]\.\s', ' ', sentence).replace('.', '')
        while count < len(sentence):
            prev_count = count
            while count < len(sentence)\
                  and sentence[count] not in SEPARATOR_CHAR\
                  and sentence[count] != '-':
                count += 1
            if count < len(sentence)-1 and sentence[count] == "'":
                words.append(sentence[prev_count:count+1])
                count += 1
            else:
                if prev_count != count:
                    words.append(sentence[prev_count:count])
                if count < len(sentence):
                    words.append(sentence[count])
                    count += 1
        return words


    def is_valid_extraction(self,
                            not_found,
                            word,
                            part_sentence,
                            start_index,
                            end_index,
                            candidates=[]):
        """ Check if some conditions are respected to return the found expansion.
            Args :
                not_found (boolean)
                word (str) : the acronym
                part_sentence (list) : the left or right part of the sentence
                start_index (int) : the index of the first word of the expansion
                end_index (int) : the index of the last word of the expansion
                candidates (list) : the possible acronyms
            Returns :
                (list) : the expansion if found
        """
        if not end_index is None and not start_index is None:
            return not_found == ''\
                and word not in part_sentence[start_index:end_index+1]\
                and part_sentence[start_index].lower() not in PREPOSITIONS\
                and part_sentence[end_index].lower() not in PREPOSITIONS\
                and ''.join(part_sentence[start_index:end_index+1]) not in candidates
        return False


    def check_left_part(self, word, left_part, sentence):
        """ Search for the expansion in the left part of the sentence.
            Args :
                word (str) : the acronym
                left_part (list) : Words in the left part of the sentence
                sentence (list) : the entire sentence
            Returns :
                (list) : the expansion if found
        """
        left_word = word
        count = 0
        separator_found = False
        end_index = -1
        start_index = None
        for ite in range(len(left_part)-1, -1, -1):
            if left_part[ite] not in SEPARATOR_CHAR and separator_found:
                if left_part[ite][0].lower() == left_word[-1].lower()\
                   and left_part[ite] not in APOSTROPHE_LETTERS\
                   and not (len(left_word) == len(word) and left_part[ite] in PREPOSITIONS):
                    if left_word == word:
                        end_index = ite
                    if len(left_word) > 1 or left_part[ite] not in PREPOSITIONS:
                        left_word = left_word[:-1]
                if end_index > 0 and left_part[ite] != "l'" and separator_found:
                    count += 1
                if len(left_word) == 0:
                    start_index = ite
                    break
            elif left_part[ite] in [")", ",", "(", "»", "«", '"', "–"]:
                separator_found = True
        if ((len(word) > 2\
        and count <= 2*len(word)+1)\
        or (len(word) == 2\
        and count <= 2*len(word)))\
        and self.is_valid_extraction(left_word, word, left_part, start_index, end_index+1):
            contain_upper = False
            for token in left_part[start_index:end_index+3]:
                if token.isupper():
                    contain_upper = True
                elif token not in SEPARATOR_CHAR and contain_upper:
                    return []
            if len(sentence) > end_index+2\
            and sentence[end_index+1] == ' '\
            and sentence[end_index+2][0].isupper()\
            and sentence[start_index][0].isupper():
                return [''.join(left_part[start_index:end_index+3])]
            extracted_exp = ''
            index = start_index
            remaining_word = word
            while len(remaining_word) > 0:
                extracted_exp = extracted_exp + left_part[index]
                if left_part[index][0].lower() == remaining_word[0].lower()\
                and not (len(remaining_word) == 1\
                    and (left_part[index] in APOSTROPHE_LETTERS or left_part[index] in PREPOSITIONS)):
                    remaining_word = remaining_word[1:]
                index += 1
            if len(left_part) > index+1\
            and left_part[index+1][0].isupper()\
            and left_part[start_index][0].isupper()\
            and left_part[index+1][0] == word[-1]:
                return [''.join(left_part[start_index:index+2])]
            return [extracted_exp]
        return []


    def check_right_part(self, word, right_part):
        """ Search for the expansion in the right part of the sentence.
            Args :
                word (str): the acronym
                right_part (list): list of the words iin the left part of the sentenece
            Returns :
                (list) : the expansion if found
        """
        right_word = word
        count = 0
        separator_found = False
        start_index = -1
        end_index = None
        for index, token in enumerate(right_part):
            if token not in SEPARATOR_CHAR and separator_found:
                if token[0].lower() == right_word[0].lower()\
                   and token not in APOSTROPHE_LETTERS\
                   and not (len(right_word) == 1 and token in PREPOSITIONS):
                    if right_word == word:
                        start_index = index
                    right_word = right_word[1:]
                if start_index > 0:
                    count += 1
                if len(right_word) == 0 or count >= 2*len(word)-1:
                    end_index = index
                    if index+1 < len(right_part) and right_part[index+1] == '-':
                        while index+1 < len(right_part)\
                        and right_part[index+1] not in SEPARATOR_CHAR:
                            end_index += 1
                            index += 1
                    break
            elif token in [")", ",", "(", "»", "«", "–", '"', ':']:
                separator_found = True
        if count <= 2*len(word)+1\
        and self.is_valid_extraction(right_word, word, right_part, start_index, end_index):
            contain_upper = False
            for token in right_part[start_index:end_index+3]:
                if token.isupper():
                    contain_upper = True
                elif token not in SEPARATOR_CHAR and contain_upper:
                    return []
            if len(right_part) > end_index+2\
            and right_part[end_index+1] == ' '\
            and right_part[end_index+2][0].isupper()\
            and right_part[start_index][0].isupper():
                return [''.join(right_part[start_index:end_index+3])]
            return [''.join(right_part[start_index:end_index+1])]
        return []


    def check_near(self, word, sentence, index_word):
        """ Look for correspondace with 2 letters acronyms where the first letter is upper.
            Args :
                word (str) : the possible acronym
                sentence (list) : the words from the same sentence of the candidate
                index_word (int) : the position of the acronym in the list
            Returns :
                expansion (list) : the expansion corresponding if found.
        """
        if index_word > 3\
        and sentence[index_word-4][0] == word[0].lower()\
        and sentence[index_word-2][0] == word[1].lower()\
        and sentence[index_word-2] not in PREPOSITIONS:
            return [''.join(sentence[index_word-4:index_word-1])]
        expansion = []
        if index_word > 1 and sentence[index_word-1] != '-':
            if sentence[index_word-2][0:2].lower() == word.lower():
                expansion.append(sentence[index_word-2])
            if word[0].lower() == sentence[index_word-2][0].lower()\
            and word[1] in sentence[index_word-2]:
                expansion.append(sentence[index_word-2])
        elif index_word+2 < len(sentence):
            if sentence[index_word+2][0:2].lower() == word.lower():
                expansion.append(sentence[index_word+2])
            elif word[0].lower() == sentence[index_word+2][0].lower()\
            and word[1] in sentence[index_word+2]:
                expansion.append(sentence[index_word+2])
        if word.isupper()\
        and index_word+1 < len(sentence)\
        and sentence[index_word+1] == '-':
            for ind, token in enumerate(sentence[:index_word]):
                if token[:2].lower() == word.lower()\
                and sentence[ind+1] == '-':
                    expansion.append(token)
        if len(word) == 2 and word.isupper()\
        and not (index_word > 0\
        and sentence[index_word-1] == '-'):
            for index in range(index_word-1):
                if sentence[index][0] == word[0]\
                and sentence[index+2][0] == word[1]\
                and len(sentence[index]) > 2\
                and len(sentence[index+2]) > 2:
                    expansion.append(''.join(sentence[index:index+3]))
        return expansion


    def check_previous_sentence(self, acro, sentence, combinaison):
        """ Extract the expansion if situated in the previous sentence.
        """
        not_found_letters = acro
        start_index = None
        count = 0
        for index, word in enumerate(sentence):
            if word not in SEPARATOR_CHAR:
                if word[0].lower() == not_found_letters[0].lower()\
                and word.lower() != acro.lower()\
                and word not in APOSTROPHE_LETTERS\
                and not index in combinaison\
                and not (len(not_found_letters) == 1 and word[0] in PREPOSITIONS):
                    if acro == not_found_letters:
                        start_index = index
                        combinaison.append(index)
                    if len(not_found_letters) > 1 or word not in PREPOSITIONS:
                        not_found_letters = not_found_letters[1:]
                if start_index is not None:
                    count += 1
                if not_found_letters == "" or count >= 2*len(acro):
                    break
        if start_index is not None\
        and not_found_letters == ''\
        and count < 2*len(acro)\
        and sentence[start_index] not in PREPOSITIONS:
            return [''.join(sentence[start_index:min(len(sentence), index+1)])], combinaison
        for index, word in enumerate(sentence):
            if word[0].lower() == acro[0].lower()\
            and index not in combinaison\
            and word.lower() != acro.lower()\
            and word not in APOSTROPHE_LETTERS:
                exp, combinaison = self.check_previous_sentence(acro, sentence, combinaison)
                return exp, combinaison
        return [], combinaison


    def check_one_letter(self, acro, sentence, index_word):
        """Search the expansion of a one letter acronym.
            Args :
                acro (str) : the acronym
                sentence (str) : the sentence containing the acronym
                index_word (int) : the index of the acronym in the sentence
            Returns :
                (list): the expansion if found
        """
        separator_found = False
        if acro.isupper():
            for word in sentence[index_word:]:
                if separator_found\
                and word != ' '\
                and word[0].lower() == acro.lower()\
                and len(word) > 1:
                    return [word]
                elif separator_found and word != ' ':
                    break
                if word in ["«", "'", '"', ':']:
                    separator_found = True

        if index_word-1 > 0\
        and sentence[index_word-2][0].lower() == acro.lower()\
        and sentence[index_word-2].lower() not in PREPOSITIONS\
        and not sentence[index_word-1] == '(':
            return [sentence[index_word-2]]
        if index_word > 2\
        and sentence[index_word-1] == '"'\
        and sentence[index_word-3][0].lower() == acro.lower()\
        and len(sentence[index_word-3]) > 3:
            return [sentence[index_word-3]]
        return []


    def get_end_digit(self, word, sentence, index_word):
        """If an acronym contains a digit expansion, separate the 2 parts
            Args :
                word (str): the acroonym candidate
                sentence (list): the words in the sentence
                index_word (int): the index of the word in the sentence
            Returns :
                word (str): the firt part of the acronym
                (str): the digit part
        """
        if word[-1].isdigit():
            ite = len(word)-1
            while word[ite].isdigit():
                ite -= 1
            return word[:ite+1], word[ite+1:]
        elif len(sentence) > index_word+2\
        and sentence[index_word+1] == '-'\
        and sentence[index_word+2].isdigit():
            return word, sentence[index_word+2]
        return word, ''


    def all_in_one_word(self, word, sentence, index_word):
        """Find a pair if the acronym is in upper case and the expansion
            is only one word with the acronym letters in upper case
            Args :
                word (str): the possible acronym
                sentence (list): the sentence
                index_word (int): the index of the acronym in the sentence
            Returns :
                (list) : the corresponding expansion
        """
        if index_word+1 < len(sentence):
            right_part = sentence[index_word+1:]
            for ind, token in enumerate(right_part):
                if token == '('\
                and ind+1 < len(right_part)\
                and word != right_part[ind+1]:
                    for letter in right_part[ind+1]:
                        if letter.isupper() and letter == word[0]:
                            word = word[1:]
                            if len(word) == 0:
                                break
                    if word == '':
                        return [right_part[ind+1]]
                    break
        return []


    def check_if_in_parenthesis(self, acro, sentence, index_word):
        """Extract the word before parenthesis.
            Args :
                acro (str) : the acronym
                sentence (str) : the sentence containing the acronym
                index_word (int) : Index of the acronym in the sentence
            Returns :
                list : the expansion of the input acronym
        """
        if index_word > 0\
        and index_word+1 < len(sentence)\
        and sentence[index_word-1] == '('\
        and sentence[index_word+1] == ')':
            index = index_word-3
            word = ''
            while index >= 0 and (sentence[index].isalpha() or sentence[index] == '-'):
                word = sentence[index] + word
                index -= 1
            for letter in word:
                if letter.upper() == acro[0]:
                    acro = acro[1:]
                if acro == '':
                    return [word]
        return []


    def remove_end_digit(self, word):
        """Remove the end of the word if consisted of digits
            Args :
                word (str): Input word to process
            Return :
                word (str): the word without digit
        """
        while word[-1].isdigit():
            word = word[:-1]
        return word


    def contains_vowel(self, word):
        """Verify if a given word contains vowel
            Args :
                word (str): the word to check
            Returns :
                (boolean)
        """
        for vowel in VOWELS:
            if vowel in word:
                return True
        return False


    def search_expansions(self, word, sentence, index_word, prev_sentence, pairs):
        """ Give the recognized expansions in the same sentence as a a given acronym.
            Args :
                word (str) : the possible acronym
                left_part (list) : the left part of the sentence splitted by the candidate
                right_part (list) : the right part of the sentence splitted by the candidate
                candidates (list) : the possibles acronyms in the text
                sentence (list) : the words from the same sentence of the candidate
                index_word (int) : the position of the acronym in the list
            Returns :
                expansions (list): expansions found for the acronym
        """
        expansions = []
        end_with_digit = False
        word, digit_part = self.get_end_digit(word, sentence, index_word)
        if digit_part != '':
            end_with_digit = True
            if word in pairs:
                return [pairs[word][0]+' '+digit_part]
        if len(word) > 2 or (len(word) == 2 and word.isupper()):
            # For exemple : 3CBO = CCCBO
            if word[0].isdigit() and not word[1].isdigit() and len(word) > 2:
                word = int(word[0])*word[1]+word[2:]
            expansions.extend(self.check_left_part(word,
                                                   sentence[:index_word],
                                                   sentence))
            expansions.extend(self.check_right_part(word, sentence[index_word+1:]))
            if len(expansions) == 0 and word.isupper() and len(word) > 2:
                exp, _ = self.check_previous_sentence(word, prev_sentence, [])
                expansions.extend(exp)
            if end_with_digit and len(expansions) > 0:
                expansions[0] = expansions[0] + ' ' + digit_part
            if len(expansions) == 0 and len(word) > 2 and word.isupper():
                expansions.extend(self.check_if_in_parenthesis(word, sentence, index_word))
            if (len(expansions) > 0 and word.lower() != expansions[0].lower()):
                return expansions
        if len(word) == 2:
            exp = self.check_near(word, sentence, index_word)
            if len(exp) > 0 and exp[0].lower() != word.lower():
                if end_with_digit:
                    exp[0] = exp[0]+' '+digit_part
                return exp
        elif len(word) == 1:
            exp = self.check_one_letter(word, sentence, index_word)
            if len(exp) == 0\
            or self.remove_end_digit(exp[0]).lower() == word.lower()\
            or not self.contains_vowel(exp[0]):
                return []
            if end_with_digit:
                exp[0] = exp[0]+' '+digit_part
            return exp
        elif word.isupper():
            exp = self.all_in_one_word(word, sentence, index_word)
            if len(exp) > 0 and exp[0].lower() != word.lower():
                return exp
        return []


    def process_candidates(self, pairs, words, candidates, prev_sentence):
        """ Extract the pair if a corresponding extantion is found in the sentence.
            Args :
                words (list): the sentence tokenized
                candidates (list): the possible acronyms
            Returns :
                pairs (dict) : the extracted pairs
        """
        for ite, word in enumerate(words):
            if word in candidates:
                possible_expansions = self.search_expansions(word,
                                                             words,
                                                             ite,
                                                             prev_sentence,
                                                             pairs)
                if len(possible_expansions) > 0\
                and possible_expansions[0].split(' ')[-1].lower() not in PREPOSITIONS:
                    if ite+2 < len(words) and words[ite+1] == '-' and words[ite+2].isdigit():
                        word = ''.join(words[ite:ite+3])
                        if words[ite+2] not in possible_expansions[0]:
                            possible_expansions[0] = possible_expansions[0]+' '+words[ite+2]
                    if '/' in possible_expansions[0]:
                        index = possible_expansions[0].index('/')
                        if ' ' in possible_expansions[0][index:]:
                            next_index = possible_expansions[0][index:].index(' ')
                            word_to_delete = possible_expansions[0][index:index+next_index+1]
                            new_exp = possible_expansions[0].replace(word_to_delete, ' ')
                            if self.valid_expansion(new_exp, word):
                                possible_expansions[0] = new_exp
                    if word not in pairs:
                        pairs[word] = []
                    for ext in possible_expansions:
                        already_extracted = False
                        for found_ext in pairs[word]:
                            if found_ext.lower() == ext.lower():
                                already_extracted = True
                        if not already_extracted and len(ext) > len(word) and ':true' not in ext:
                            pairs[word].append(ext)
        return pairs


    def has_parenthesis(self, expansion):
        """Verify the presence on unclosed parenthesis
            Args :
                expansion (list): words in the expansion
            Returns :
                boolean
        """
        if ('(' in expansion and not ')' in expansion)\
            or (')' in expansion and not '(' in expansion)\
            or expansion[-5:] == "c'est":
            return False
        return True


    def count_valuable_words(self, expansion):
        """Counts the number of words that are neither prepositions nor linking words.
            Args :
                expansion (list): the expansion
            Returns :
                count (int): number of valuable words.
        """
        exp = expansion.replace('-', ' ').split(' ')
        count = 0
        for token in exp:
            if token.lower() not in PREPOSITIONS\
            and token not in SEPARATOR_CHAR\
            and token.lower() not in APOSTROPHE_LETTERS:
                count += 1
        return count


    def shortest_expansion(self, expansions):
        """Give the shortest expansion according to non prepositional or linking words.
            Args :
                expansions (list): the expansions
            Returns :
                shortest (str) : the shortest expansion
        """
        shortest_len = 100
        for expansion in expansions:
            length = self.count_valuable_words(expansion)
            if length < shortest_len:
                shortest_len = length
                shortest = expansion
        return shortest


    def most_suitable_expansions(self, pairs):
        """If several expansions are found for an acronym, return the most accurate.
            Args :
                pairs (dict): pairs of acronym as key and their corresponding expansions.
            Returns :
                new_dict (dict) : the pairs with only one expansion for each acronym.
        """
        new_dict = {}
        for pair, expansions in pairs.items():
            if len(expansions) == 1:
                if self.has_parenthesis(expansions[0]):
                    new_dict[pair] = expansions[0].replace('"', "").replace('-', ' ')
            elif len(expansions) > 1:
                matching_letter = False
                new_dict[pair] = expansions[0]
                possible = []
                for expansion in expansions:
                    cleaned_expansion = ''
                    for letter in expansion:
                        if letter in SEPARATOR_CHAR or letter == '-':
                            cleaned_expansion = cleaned_expansion + ' '
                        else:
                            cleaned_expansion = cleaned_expansion + letter
                    cleaned_expansion = cleaned_expansion.split()
                    if self.count_valuable_words(expansion) == len(pair)\
                    and not (pair[-1].isdigit()\
                    and not cleaned_expansion[-1].isdigit())\
                    and self.has_parenthesis(expansion):
                        matching_letter = True
                        possible.append(expansion.replace('-', ' '))
                if matching_letter:
                    new_dict[pair] = possible[0]
                else:
                    shortest = self.shortest_expansion(expansions)
                    if self.has_parenthesis(shortest):
                        new_dict[pair] = shortest
        return new_dict


    def extract_all_pairs(self, sentences):
        """Get the pairs within each sentences of the text.
            Args:
                sentences (list): the sentences stored in the list.
            Returns:
                pairs (dict): the dict of pairs with the acronym as value
        """
        pairs = {}
        processed_sentences = []
        for sentence in sentences:
            processed_sentences.append(self.tokenization(sentence))
        for index, sentence in enumerate(processed_sentences):
            acronym_candidates = self.get_acronym_candidates(sentence)
            prev_sentence = []
            if index > 0:
                prev_sentence = processed_sentences[index-1]
            pairs = self.process_candidates(pairs,
                                            sentence,
                                            acronym_candidates,
                                            prev_sentence)
            for word in range(len(sentence)-2):
                if sentence[word] in pairs\
                and sentence[word+1] == '-'\
                and sentence[word+2].isdigit():
                    pairs[''.join(sentence[word:word+3])] = [pairs[sentence[word]][0]\
                                                             +' '\
                                                             +sentence[word+2]]
        pairs = self.most_suitable_expansions(pairs)
        return pairs


    def get_acronym_expansion_pairs(self, text, lang=None):
        """Extract the (acronym, expansion) pairs from a given text.
            Args :
                text (str): the from which to extract the pairs
            Returns:
                pairs (dict): the dict of pairs with the acronym as key
        """
        # Custom extractor part
        processed_text = self.fr_special_char_replacer(text)
        text_in_list = self.text_to_tokenized_list(processed_text)
        pairs = self.extract_all_pairs(text_in_list)
        # Schwartz and Hearst Extractor
        return self.sh_adaptation(pairs, text)


    def get_all_acronyms(self, doc_text):
        """Extract all the possible acronyms inside a given text.
            Args:
                doc_text (str): the text
            Returns:
                acronyms (list): the extracted acronyms
        """
        acronyms = []
        text = [self.tokenization(sent) for sent in self.text_to_tokenized_list(self.fr_special_char_replacer(doc_text))]
        for sentence in text:
            for token in self.get_acronym_candidates(sentence):
                if token.isupper() and len(token)>1:
                    acronyms.append(token)
        acronyms = set(acronyms)
        return acronyms
