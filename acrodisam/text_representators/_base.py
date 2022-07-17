"""
Base abstract classes for Text Representators and their factories

Created on Aug 31, 2018

@author: jpereira
"""
from abc import ABCMeta, abstractmethod
import itertools
from typing import Optional

from Logger import logging
from inputters import TrainOutDataManager, InputArticle
from helper import ExecutionTimeObserver, TrainInstance


logger = logging.getLogger(__name__)


class TextRepresentator(metaclass=ABCMeta):
    """
    Abstract Class for Text Representators

    New Text Representators should override the transform method
    """

    @abstractmethod
    def tranform_test_instance(self, acronym_list: list, article: InputArticle):
        """"""

    def tranform_train_instances(self, train_instance_list: list):
        for train_instance in train_instance_list:
            yield self._transform_train_instance(train_instance)

    @abstractmethod
    def _transform_train_instance(self, train_instance: TrainInstance):
        """"""

    # @abstractmethod
    # def transform(self, X: List[AcronymExpansion]):
    #    """
    #    transforms each element into a representation

    #   :param x: list of elements to transform
    #    """


class TextRepresentatorAcronymIndependent(TextRepresentator):
    def tranform_test_instance(self, acronym_list: list, article: InputArticle):
        text_representation = self._transform_input_text(article)
        return itertools.repeat(text_representation, len(acronym_list))

    @abstractmethod
    def _transform_input_text(self, article: InputArticle):
        """"""


class TextRepresentatorFactory(
    metaclass=ABCMeta
):  # pylint: disable=too-few-public-methods
    """
    Abstract Class for Text Representators Factories

    Factories should override the get_text_representator method
    """

    @abstractmethod
    def get_text_representator(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: Optional[ExecutionTimeObserver] = None,
    ) -> TextRepresentator:
        """
        Generates a new representator for a given database

        :param articles_db: dict like database that contains articles text as
         article_id:article_text
        :param article_acronym_db: dict like database that contains acronym
        expansion pairs by article as article_id:{acronym:expansion}
        :param fold: fold identifier
        :param execution_time_observer: executions time saver
        """
