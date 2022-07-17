from AcronymExtractors.AcronymExtractor_v2 import AcronymExtractor_v2

class AcronymExtractor_v2_small(AcronymExtractor_v2):
    """
    Changes minimum acronym length to 2
    """
    def __init__(self):
        self.pattern = r'\b[A-Z]{2,8}s{0,1}\b'# Limit length 8