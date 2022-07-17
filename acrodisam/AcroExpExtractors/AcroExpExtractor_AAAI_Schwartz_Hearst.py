import logging
from AcroExpExtractors.AcroExpExtractor import (
    AcroExpExtractorRb,
)
from nltk import word_tokenize
import re

tokenizer = word_tokenize

logger = logging.getLogger(__name__)


class AcroExpExtractor_AAAI_Schwartz_Hearst(AcroExpExtractorRb):
    """Implementation of the Schwartz and Hearts rule based algorithm."""

    def __init__(self):
        pass

    def _has_capital(self, string):
        """Checks if a string has a capital.

        Args:
            string (str): a string

        Returns:
            bool: True if string as capital, False otherwise.
        """
        for i in string:
            if i.isupper():
                return True
        return False

    def _is_valid_short_form(self, acr):
        """Checks if an acronym has a letter and the first character is alphanumeric.

        Args:
            acr (str): the acronym

        Returns:
            bool: True if acronym is valid, False otherwise.
        """
        if self._has_letter(acr) and (
            acr[0].isalpha() or acr[0].isdecimal() or acr[0] == "("
        ):
            return True
        else:
            return False

    def _has_letter(self, string):
        """Checks if a string as a letter.

        Args:
            string (str): a string.

        Returns:
            bool: True if the string as a letter, False otherwise.
        """
        for c in string:
            if c.isalpha():
                return True
        return False

    def _find_best_long_form(self, short_form, long_form):
        """Returns the best long form for a short form from a series of candidate long forms.

        Args:
            short_form (str): the acronym
            long_form (str): the candidate expansion
        Returns:
            str: the best long form for the short form. If one is not found None is returned.
        """
        l_index = len(long_form) - 1

        for s_index in range(len(short_form) - 1, -1, -1):
            curr_char = short_form[s_index].lower()
            if not (curr_char.isalpha() or curr_char.isdecimal()):
                continue
            while (l_index >= 0 and long_form[l_index].lower() != curr_char) or (
                s_index == 0
                and l_index > 0
                and (
                    long_form[l_index - 1].isalpha()
                    or long_form[l_index - 1].isdecimal()
                )
            ):
                l_index -= 1
            if l_index < 0:
                return None
            l_index -= 1
        l_index = long_form.rfind(" ", 0, l_index + 1) + 1
        return long_form[l_index:]

    def _extract_abbr_pair(self, short_form, long_form, candidates):
        """Extracts the best long form for a short form from several candidate long forms.

        Args:
            short_form (str): the acronym
            long_form (str): the candidate expansion
            candidates (dict): a dictionary whose keys are acronyms and values are expansions.

        Returns:
            dict: a dicionary whose keys are acronyms and values are expansions.
        """
        best_long_form = ""
        long_form_size, short_form_size = 0, 0
        if len(short_form) == 1:
            return candidates
        best_long_form = self._find_best_long_form(short_form, long_form)
        if best_long_form == None:
            return candidates
        best_long_tokens = re.split("[ \t\n\r\f-]", best_long_form)
        best_long_tokens = [x for x in best_long_tokens if x != ""]
        long_form_size = len(best_long_tokens)
        short_form_size = len(short_form)
        for i in range(short_form_size - 1, -1, -1):
            if not (short_form[i].isalpha() or short_form[i].isdecimal()):
                short_form_size -= 1
        if (
            len(best_long_form) < len(short_form)
            or best_long_form.find(short_form + " ") > -1
            or best_long_form.endswith(short_form)
            or long_form_size > short_form_size * 2
            or long_form_size > short_form_size + 5
            or short_form_size > 10
        ):
            return candidates

        candidates[short_form] = best_long_form
        return candidates

    def _extract_abbr_pairs_from_str(self, text):
        """Extracts acronyms and the corresponding expansions that are present in text.

        Args:
            text (str): the text to analyse

        Returns:
            dict: a dicionary whose keys are acronyms and values are expansions.
        """
        tmp_str, long_form, short_form, curr_sentence = "", "", "", ""
        (
            open_paren_index,
            close_paren_index,
            sentence_end,
            new_close_paren_index,
            tmp_index,
        ) = (
            -1,
            -1,
            -1,
            -1,
            -1,
        )
        new_paragraph = True
        candidates = {}
        try:
            text_split = text.split("\n")
            for phrase in text_split:
                if len(phrase) == 0 or new_paragraph and not phrase[0].isupper():
                    curr_sentence = ""
                    new_paragraph = True
                    continue
                new_paragraph = False
                phrase += " "
                curr_sentence += phrase
                open_paren_index = curr_sentence.find(" (")
                paren_cond = True
                while paren_cond:
                    if open_paren_index > -1:
                        open_paren_index += 1
                    sentence_end = max(
                        curr_sentence.rfind(". "), curr_sentence.rfind(", ")
                    )
                    if open_paren_index == -1 and not sentence_end == -1:
                        curr_sentence = curr_sentence[sentence_end + 2 :]
                    elif curr_sentence.find(")", open_paren_index) > -1:
                        close_paren_index = curr_sentence.find(")", open_paren_index)
                        sentence_end = max(
                            curr_sentence.rfind(". ", 0, open_paren_index + 1),
                            curr_sentence.rfind(", ", 0, open_paren_index + 1),
                        )
                        if sentence_end == -1:
                            sentence_end = -2
                        long_form = curr_sentence[sentence_end + 2 : open_paren_index]
                        short_form = curr_sentence[
                            open_paren_index + 1 : close_paren_index
                        ]
                    if len(short_form) > 0 or len(long_form) > 0:
                        if len(short_form) > 1 and len(long_form) > 1:
                            if (
                                short_form.find("(") > -1
                                and curr_sentence.find(")", close_paren_index + 1) > -1
                            ):
                                new_close_paren_index = curr_sentence.find(
                                    ")", close_paren_index + 1
                                )
                                short_form = curr_sentence[
                                    open_paren_index + 1 : new_close_paren_index
                                ]
                                close_paren_index = new_close_paren_index

                            if short_form.find(", ") > -1:
                                tmp_index = short_form.find(", ")
                                short_form = short_form[0:tmp_index]
                            if short_form.find("; ") > -1:
                                tmp_index = short_form.find("; ")
                                short_form = short_form[0:tmp_index]
                            short_tokens = re.split("[ \t\n\r\f]", short_form)
                            short_tokens = [x for x in short_tokens if x != ""]
                            if len(short_tokens) > 2 or len(short_form) > len(
                                long_form
                            ):
                                tmp_index = curr_sentence.rfind(
                                    " ", 0, open_paren_index - 2 + 1
                                )
                                tmp_str = curr_sentence[
                                    tmp_index + 1 : open_paren_index - 1
                                ]
                                long_form = short_form
                                short_form = tmp_str
                                if not self._has_capital(short_form):
                                    short_form = ""
                            if self._is_valid_short_form(short_form):
                                candidates = self._extract_abbr_pair(
                                    short_form.strip(), long_form.strip(), candidates
                                )

                        curr_sentence = curr_sentence[close_paren_index + 1 :]
                    elif open_paren_index > -1:
                        if (len(curr_sentence) - open_paren_index) > 200:
                            # Matching close paren was not found
                            curr_sentence = curr_sentence[open_paren_index + 1 :]
                        break
                    short_form = ""
                    long_form = ""
                    open_paren_index = curr_sentence.find(" (")
                    paren_cond = open_paren_index > -1
        except Exception:
            logger.exception(
                "Fatal error running Schwartz and Hearst AcroExpExtractor for text: "
                + text
            )

        return candidates

    def _split_tokens(self, tokens):
        """Split element according to specific chars.

        Args:
            tokens (list): list of tokens
        """
        for token in tokens:
            for element in re.split(
                "[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D/]",
                token,
            ):
                yield element

    def get_all_acronyms(self, text):
        """Extract all the possible acronyms inside a given text.

        Args:
            text (str): the text
        Returns:
            acronyms (list): the extracted acronyms
        """
        tokens = set(self._split_tokens(tokenizer(text)))
        acronyms = []
        for token in tokens:
            if len(token) < 2 or token[1:].islower():
                continue
            if len(token) < 3 and token[1] == ".":
                continue
            if self._is_valid_short_form(token):
                acronyms.append(token)
        return acronyms

    def get_all_acronym_expansion(self, text):
        """Returns a dicionary where each key is an acronym (str) and each value is an expansion (str). The expansion is None if no expansion is found.

        Args:
            text (str): the text to extract acronym-expansion pairs from

        Returns:
            dict:
             a dict that has acronyms as values and definitions as keys. Definition is None if no definition is found for the acronym in text.
             The returned dict is enconded in UTF-8 (the python3 standard)
        """
        acronyms = self.get_all_acronyms(text)
        abbrev_map = {acronym: None for acronym in acronyms}
        for acronym, expansion in self.get_acronym_expansion_pairs(text).items():
            abbrev_map[acronym] = expansion
        return abbrev_map

    def get_acronym_expansion_pairs(self, text):
        """Returns a dicionary where each key is an acronym (str) and each value is an expansion (str).

        Args:
            text (str): the text to extract acronym-expansion pairs from

        Returns:
            dict:
             a dict that has acronyms as values and definitions as keys. The returned dict is enconded in UTF-8 (the python3 standard)
        """
        return self._extract_abbr_pairs_from_str(text)


if __name__ == "__main__":
    acroExp = AcroExpExtractor_AAAI_Schwartz_Hearst()

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
