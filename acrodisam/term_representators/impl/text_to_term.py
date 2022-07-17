"""
Makes a term representator out of a text representator by summing the text representations for a given term
"""
from typing import Optional

from pydantic import validate_arguments

from helper import ExecutionTimeObserver
from inputters import TrainOutDataManager
from text_representators import TextRepresentator
from text_representators import get_text_representator_factory

from .._base import TermRepresentator, TermRepresentatorFactory


class FactoryTextToTerm(
    TermRepresentatorFactory
):  # pylint: disable=too-few-public-methods
    """
    Factory that creates a term representator from a text representator
    """

    @validate_arguments
    def __init__(self, *args, **kw):
        """
        Args:
            text_representator: a text representator factory
            run_config: general run configurations
        """
        self.text_representator_fac = get_text_representator_factory(*args, **kw)

    def get_term_representator(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: Optional[ExecutionTimeObserver] = None,
    ):
        text_representator = self.text_representator_fac.get_text_representator(
            train_data_manager=train_data_manager,
            execution_time_observer=execution_time_observer,
        )

        return _RepresentatorTextToTerm(text_representator)


class _RepresentatorTextToTerm(
    TermRepresentator
):  # pylint: disable=too-few-public-methods
    def __init__(self, text_representator: TextRepresentator):
        super().__init__()
        self.text_representator = text_representator

    def tranform_acronym_terms(self, acronym_list, article):
        return self.text_representator.tranform_test_instance(acronym_list, article)

    def tranform_expansion_terms(self, expansion_list, train_instance_list):
        transformed_train_instances = self.text_representator.tranform_train_instances(
            train_instance_list
        )
        representation_per_expansion = {}

        for instance, transformed_instance in zip(
            train_instance_list, transformed_train_instances
        ):

            expansion = instance.expansion
            if expansion not in representation_per_expansion:
                representation_per_expansion[expansion] = transformed_instance
            else:
                representation_per_expansion[expansion] += transformed_instance

        return [representation_per_expansion[expansion] for expansion in expansion_list]
