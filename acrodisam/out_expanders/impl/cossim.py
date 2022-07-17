"""
Selects the expansion whose vector obtains the closer cosine distance to the test instance
"""
from sklearn.metrics.pairwise import cosine_similarity

import helper
from inputters import TrainOutDataManager
import numpy as np
from term_representators import get_term_representator_factory
from text_representators import TextRepresentator, get_text_representator_factory

from .._base import (
    OutExpanderWithTextRepresentator,
    OutExpanderWithTermRepresentator,
    OutExpanderFactory,
)


class FactoryExpanderCossim(OutExpanderFactory):
    def __init__(
        self,
        representator: str = "document_context_vector",
        representator_args=None,
        *args,
        **kwargs
    ):
        """
        Args:
            representator: Text or term representator name
            representator_args: Arguments for the representator
        """
        new_args, new_kwargs = helper.get_args_to_pass(representator_args, args, kwargs)
        self.text_representator = True
        try:
            self.representator_factory = get_text_representator_factory(
                representator, *new_args, **new_kwargs
            )
        except ModuleNotFoundError:
            self.representator_factory = get_term_representator_factory(
                representator, *new_args, **new_kwargs
            )
            self.text_representator = False

    def get_expander(
        self, train_data_manager: TrainOutDataManager, execution_time_observer=None
    ):
        if self.text_representator:
            text_representator = self.representator_factory.get_text_representator(
                train_data_manager=train_data_manager,
                execution_time_observer=execution_time_observer,
            )

            return _ExpanderCossimText(text_representator)

        term_representator = self.representator_factory.get_term_representator(
            train_data_manager=train_data_manager,
            execution_time_observer=execution_time_observer,
        )

        return _ExpanderCossimTerm(term_representator)


class _ExpanderCossimText(OutExpanderWithTextRepresentator):
    def fit(self, X_train, y_train):
        self.x_train_list = list(X_train)
        self.y_train = y_train

    def predict(self, X_test, acronym):
        if isinstance(X_test, np.ndarray):
            X_test = X_test.reshape(1, -1,order= 'c')
        else:
            aux = list()
            aux.append(X_test)
            X_test = aux
            
        sim_matrix = cosine_similarity(self.x_train_list, X_test)
        argmax_array = np.argmax(sim_matrix, axis=0)
        labels = [self.y_train[idx] for idx in argmax_array]
        confidences = sim_matrix.take(argmax_array)
        return labels, confidences

    def predict_confidences(self, text_representation, acronym, distinct_expansions):
        confidences_dict = {}
        sim_matrix = cosine_similarity(self.x_train_list, text_representation.reshape(1, -1))
        for exp, conf in zip(self.y_train, sim_matrix):
            confidences_dict[exp] = max(confidences_dict.get(exp,0), conf[0].item())
        
        return confidences_dict


class _ExpanderCossimTerm(OutExpanderWithTermRepresentator):

    def fit(self, X_train, y_train):
        self.X_train = list(X_train)
        self.y_train = y_train

    def predict(self, acronym_representation, acronym):
        if isinstance(acronym_representation, np.ndarray):
            X_test = acronym_representation.reshape(1, -1,order= 'c')
        else:
            aux = list()
            aux.append(acronym_representation)
            X_test = aux
            
        sim_matrix = cosine_similarity(self.X_train, X_test)
        argmax_array = np.argmax(sim_matrix, axis=0)
        labels = [self.y_train[idx] for idx in argmax_array]
        confidences = sim_matrix.take(argmax_array)
        return labels, confidences


    def predict_confidences(self, acronym_representation, acronym, distinct_expansions):
        confidences_dict = {}
        sim_matrix = cosine_similarity(self.X_train, acronym_representation.reshape(1, -1))
        for exp, conf in zip(distinct_expansions, sim_matrix):
            confidences_dict[exp] = conf[0]
        
        return confidences_dict

