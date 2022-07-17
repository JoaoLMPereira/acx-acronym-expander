"""
Created on Jun 25, 2019
@author: jpereira
"""
import logging

from nltk.tokenize import word_tokenize

from AcroExpExtractors.AcroExpExtractor import AcroExpExtractorRb
from abbreviations import schwartz_hearst

from nltk.tokenize import RegexpTokenizer, sent_tokenize

log = logging.getLogger(__name__)


class AcroExpExtractor_Schwartz_Hearst(AcroExpExtractorRb):
    def get_all_acronyms(self, doc_text):
        tokens = word_tokenize(doc_text)

        return [
            t
            for t in tokens
            if t.isupper()
            and schwartz_hearst.conditions(t)
            and not (len(t) == 2 and t[1] == ".")
        ]

    # Modified code from schwartz_hearst.extract_abbreviation_definition_pairs to return acronyms with no expansion
    def get_all_acronym_expansion(self, doc_text):

        acronyms = self.get_all_acronyms(doc_text)

        abbrev_map = {acronym: None for acronym in acronyms}
        omit = 0
        written = 0

        sentence_iterator = enumerate(schwartz_hearst.yield_lines_from_doc(doc_text))

        for i, sentence in sentence_iterator:
            try:
                for candidate in schwartz_hearst.best_candidates(sentence):
                    try:
                        definition = schwartz_hearst.get_definition(candidate, sentence)
                    except (ValueError, IndexError) as e:
                        log.debug(
                            "{} Omitting candidate {}. Reason: {}".format(
                                i, candidate, e.args[0]
                            )
                        )
                        if candidate not in abbrev_map:
                            abbrev_map[candidate] = None
                        omit += 1
                    else:
                        try:
                            definition = schwartz_hearst.select_definition(
                                definition, candidate
                            )
                        except (ValueError, IndexError) as e:
                            log.debug(
                                "{} Omitting definition {} for candidate {}. Reason: {}".format(
                                    i, definition, candidate, e.args[0]
                                )
                            )
                        if candidate not in abbrev_map:
                            abbrev_map[candidate] = None
                            omit += 1
                        else:
                            abbrev_map[candidate] = definition
                            written += 1
            except (ValueError, IndexError) as e:
                log.debug(
                    "{} Error processing sentence {}: {}".format(i, sentence, e.args[0])
                )
        log.debug(
            "{} abbreviations detected and kept ({} omitted)".format(written, omit)
        )
        return abbrev_map

    def get_acronym_expansion_pairs(self, text):
        newText = "\n".join(sent_tokenize(text))
        return schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=newText)

    def get_best_expansion(self, acro, text):
        best_long_form = ""

        text = text + " (" + acro + ")"

        acr_exp = self.get_acronym_expansion_pairs(text)

        if acro in acr_exp.keys():
            best_long_form = acr_exp[acro]

        return best_long_form
