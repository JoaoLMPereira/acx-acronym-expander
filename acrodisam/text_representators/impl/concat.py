"""
Text representator that concatenates the representations returned by other text representators
"""
from typing import Optional
from itertools import chain

from Logger import logging
from helper import grouped_longest, ExecutionTimeObserver
from inputters import TrainOutDataManager
import numpy as np
from term_representators import TermRepresentator, get_term_representator_factory
from term_representators._base import TermRepresentatorFactory
from text_representators import get_text_representator_factory

from .._base import TextRepresentator, TextRepresentatorFactory


logger = logging.getLogger(__name__)


class FactoryConcatRepresentators(
    TextRepresentatorFactory
):  # pylint: disable=too-few-public-methods
    """
    Text representator that concanetas the output of
     other text representators into a single vector
    """

    def __init__(self, *args, **kwargs):
        """
        :param args: List of text representators names and their arguments
        first list element and impair elements should contain the text representator name as string
        in the next position of the list should be the arguments to be passed to that representator
         in list or dict format.
         [name_1, args_1, name_2, args_2, ..., name_n, args_n]
         where name_x is a text representator name
         and args_n are the arguments for text representator x
        :param kwargs: to be passed to each text representator,
        includes general run information
        """
        self.representator_factories = []
        for representator_name, representator_args in grouped_longest(args, 2):
            rep_args_list = [] if representator_args is None else representator_args
            try:
                representator_factory = get_text_representator_factory(
                    representator_name, *rep_args_list, **kwargs
                )
            except (ValueError, TypeError, ModuleNotFoundError):
                term_representator_factory = get_term_representator_factory(
                    representator_name, *rep_args_list, **kwargs
                )
                representator_factory = _TermToTextRepresentatorFactory(
                    term_representator_factory
                )

            self.representator_factories.append(representator_factory)

    def get_text_representator(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: Optional[ExecutionTimeObserver] = None,
    ) -> TextRepresentator:
        representators = [
            factory.get_text_representator(
                train_data_manager=train_data_manager,
                execution_time_observer=execution_time_observer,
            )
            for factory in self.representator_factories
        ]

        return _RepresentatorConcat(representators)


class _TermToTextRepresentatorFactory(TextRepresentatorFactory):
    def __init__(self, term_representator_factory: TermRepresentatorFactory):
        self.term_representator_factory = term_representator_factory

    def get_text_representator(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: Optional[ExecutionTimeObserver] = None,
    ) -> TextRepresentator:
        term_representator = self.term_representator_factory.get_term_representator(
            train_data_manager=train_data_manager,
            execution_time_observer=execution_time_observer,
        )
        return _TermToTextRepresentator(term_representator)


class _TermToTextRepresentator(TextRepresentator):
    """
    We need this to concat term representations to text representation
    """

    def __init__(self, term_representator: TermRepresentator):
        self.term_representator = term_representator

    def tranform_test_instance(self, acronym_list, article):
        return self.term_representator.tranform_acronym_terms(acronym_list, article)

    def tranform_train_instances(self, train_instance_list):
        expansion_list = list({instance.expansion for instance in train_instance_list})

        expansions_representations = list(
            self.term_representator.tranform_expansion_terms(
                expansion_list, train_instance_list
            )
        )

        for instance in train_instance_list:
            idx = expansion_list.index(instance.expansion)
            yield expansions_representations[idx]

    def _transform_train_instance(self, train_instance):
        pass


class _RepresentatorConcat:
    def __init__(self, representators):
        super().__init__()
        self.representators = representators

    def tranform_test_instance(self, acronym_list, article):
        list_arrays = [
            r.tranform_test_instance(acronym_list, article) for r in self.representators
        ]
        for a in zip(*list_arrays):
            yield list(chain.from_iterable(a))

    def tranform_train_instances(self, train_instance_list):
        list_arrays = [
            r.tranform_train_instances(train_instance_list) for r in self.representators
        ]
        for instance_representations in zip(*list_arrays):
            concat_vec = list(chain.from_iterable(instance_representations))
            yield concat_vec

    def _transform_train_instance(self, train_instance):
        # TODO we have to implement this for now, change interface
        pass
