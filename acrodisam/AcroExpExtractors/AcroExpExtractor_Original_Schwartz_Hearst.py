"""
Calls the original code from Schwartz and Hearst in java

Aditional conditions to identify acronyms with no expansion in text

@author: jpereira
"""


import logging
import re

from AcroExpExtractors.AcroExpExtractor import AcroExpExtractorRb

from string_constants import FOLDER_ROOT

from nltk.tokenize import word_tokenize

# from string_constants import FILE_PREPOSITIONS, FILE_DETERMINERS, FILE_PARTICLES, FILE_CONJUNCTIONS, FOLDER_DATA, FILE_JUST_ENGLISH_DICT


logger = logging.getLogger(__name__)

"""
acronymsToReject = set()
acronymsToReject |= {line.strip().lower() for line in open(FILE_PREPOSITIONS, 'r')}
acronymsToReject |= {line.strip().lower() for line in open(FILE_DETERMINERS, 'r')}
acronymsToReject |= {line.strip().lower() for line in open(FILE_PARTICLES, 'r')}
acronymsToReject |= {line.strip().lower() for line in open(FILE_CONJUNCTIONS, 'r')}

english_words2 = set(word.strip().casefold() for word in open(FILE_JUST_ENGLISH_DICT))
"""


class AcroExpExtractor_Original_Schwartz_Hearst(AcroExpExtractorRb):
    def __init__(self):

        import jnius_config

        jnius_config.add_classpath(".", FOLDER_ROOT + "acrodisam/AcroExpExtractors/")
        from jnius import autoclass

        self.javaClass = autoclass("OriginalSchwartzHearst")
        self.instance = self.javaClass()
        self.algorithm = self.instance.extractAbbrPairsFromStr

    def has_capital(self, token):
        """Check if the token contains at least one capital letter.
        Args :
            token (str) : the element to process
        Returns :
            boolean
        """
        for char in token:
            if char.isupper():
                return True
        return False

    def has_letter(self, token):
        """Check if a given ttoken contains at least one char.
        Args :
            token (str) : the element to process
        Returns :
            boolean
        """
        for char in token:
            if char.isalpha():
                return True
        return False

    def split_tokens(self, tokens):
        """Split element according to specifics char.
        Args :
            tokens (List): list of token
        """
        for token in tokens:
            for element in re.split(
                "[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D/]",
                token,
            ):
                yield element

    def get_all_acronyms(self, doc_text):
        """Extract all the possible acronyms inside a given text.
        Args:
            doc_text (str): the text
        Returns:
            acronyms (list): the extracted acronyms
        """
        tokens = set(self.split_tokens(word_tokenize(doc_text)))
        acronyms = []
        for token in tokens:
            if len(token) < 2 or token[1:].islower():
                continue

            if len(token) < 3 and token[1] == ".":
                continue

            if self.has_letter(token) and token[0].isalnum():
                acronyms.append(token)
        return acronyms

    def get_all_acronym_expansion(self, text):
        """Analyse a given text to return all the acronym and their expansion
        if in the text.
            Args :
                text (str) : the text to process
            Returns :
                abbrev_map (dict) : A dictionnary with the acronym as key and expansion as value
        """
        acronyms = self.get_all_acronyms(text)
        abbrev_map = {acronym: None for acronym in acronyms}
        for acronym, expansion in self.get_acronym_expansion_pairs(text).items():
            abbrev_map[acronym] = expansion
        return abbrev_map

    def get_acronym_expansion_pairs(self, text):
        try:
            results = {}

            # javaText = self.javaString(text)

            # javaMap = self.algorithm(javaText)
            javaMap = self.algorithm(text)

            entrySet = javaMap.entrySet()

            javaIterator = entrySet.iterator()
            while javaIterator.hasNext():
                entry = javaIterator.next()
                results[str(entry.key)] = str(entry.value)

            return results
        except Exception:
            logger.exception("Fatal error in java for text: " + text)
            return results


if __name__ == "__main__":

    acroExp = AcroExpExtractor_Original_Schwartz_Hearst()

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
