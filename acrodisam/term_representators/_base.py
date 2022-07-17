"""
Base abstract classes for Term Representators and their factories

@author: jpereira
"""
from abc import ABCMeta, abstractmethod
import itertools
from typing import Optional

from Logger import logging
from helper import ExecutionTimeObserver
from inputters import TrainOutDataManager, InputArticle


logger = logging.getLogger(__name__)


class TermRepresentator(metaclass=ABCMeta):
    """
    Abstract Class for Term Representators

    New Term Representators should override the transform method
    """

    @abstractmethod
    def tranform_acronym_terms(self, acronym_list: list, article: InputArticle):
        """
        Args:
            acronym_list: list of acronyms in input article to represent
            article: input article

        Yields:
            A representation of an acronym-article
        """

    @abstractmethod
    def tranform_expansion_terms(self, expansion_list: list, train_instance_list):
        """
        Args:
            expansion_list: list of expansions to represent
            train_instance_list: list of tain instances (article ids) that contains the expansions

        Yields:
            expansion representation
        """


class TermRepresentatorAcronymIndependent(TermRepresentator):
    """
    Abstract class that generalizes Term Representators whose acronym-article
     representation only depends on the article
    """

    def tranform_acronym_terms(self, acronym_list, article):
        return itertools.repeat(self._transform_input_text(article), len(acronym_list))

    @abstractmethod
    def _transform_input_text(self, article: InputArticle):
        pass

    def tranform_expansion_terms(self, expansion_list, train_instance_list):
        for expansion in expansion_list:
            yield self._transform_expansion_term(expansion)

    @abstractmethod
    def _transform_expansion_term(self, expansion):
        pass


class TermRepresentatorFactory(
    metaclass=ABCMeta
):  # pylint: disable=too-few-public-methods
    """
    Abstract Class for Text Representators Factories

    Factories should override the get_text_representator method
    """

    @abstractmethod
    def get_term_representator(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: Optional[ExecutionTimeObserver] = None,
    ) -> TermRepresentator:
        """
        Generates a new term representator for a given data

        Args:
            train_data_manager: manager of the train data, used to create the representator
            execution_time_observer: executions time saver
        """
