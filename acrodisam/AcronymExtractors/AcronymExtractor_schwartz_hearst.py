from abbreviations import schwartz_hearst

from AcronymExtractors.AcronymExtractor import AcronymExtractor


class AcronymExtractor_schwartz_hearst(AcronymExtractor):
    """
    """
    def get_acronyms(self, text):  # Find Acronyms in text
        acronyms = []
    
        sentence_iterator = schwartz_hearst.yield_lines_from_doc(text)
    
        for sentence in sentence_iterator:
            acronyms.append(schwartz_hearst.best_candidates(sentence))
             
        return acronyms
