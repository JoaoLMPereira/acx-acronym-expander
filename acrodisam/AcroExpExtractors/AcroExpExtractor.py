"""
Created on Jun 26, 2019

@author: jpereira
"""

from abc import ABCMeta, abstractmethod
from inputters import TrainInDataManager


class AcroExpExtractorFactory(metaclass=ABCMeta):
    @abstractmethod
    def get_in_expander(
        self,
    ):
        pass


class AcroExpExtractor(metaclass=ABCMeta):
    @abstractmethod
    def get_all_acronym_expansion(self, text):
        """Returns a dicionary where each key is an acronym (str) and each value is an expansion (str). The expansion is None if no expansion is found.

        Args:
            text (str): the text to extract acronym-expansion pairs from
        """
        pass

    @abstractmethod
    def get_acronym_expansion_pairs(self, text):
        """Returns a dicionary where each key is an acronym (str) and each value is an expansion (str).

        Args:
            text (str): the text to extract acronym-expansion pairs from
        """
        pass

    def get_best_expansion(self, acro, text):
        """Returns the best expansion present in text for the given acro.

        Args:
            acro (str): the acronym to which an expansion is to be found in text
            text (str): a string where the expansion for the acronym might be present
        """
        best_long_form = ""

        text = text + " (" + acro + ")"

        acr_exp = self.get_acronym_expansion_pairs(text)

        if acro in acr_exp.keys():
            best_long_form = acr_exp[acro]

        return best_long_form


class AcroExpExtractorRb(AcroExpExtractor):
    pass


class AcroExpExtractorFactoryMl(AcroExpExtractorFactory):
    def __init__(self, train_data_manager_base: TrainInDataManager):
        self.training_data_name = train_data_manager_base.get_dataset_name()
        self.articles_raw_db = train_data_manager_base.get_raw_articles_db()
        self.article_acronym_db = train_data_manager_base.get_article_acronym_db()


class AcroExpExtractorMl(AcroExpExtractor):
    pass
