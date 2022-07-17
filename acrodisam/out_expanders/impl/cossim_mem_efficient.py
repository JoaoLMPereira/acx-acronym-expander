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

# in case of failiuer compares pairs in a lazy fashion
def cosine_memory_error_recover(input_a, input_b):
        
    #switch for more efficiency
    aux = None
    if len(input_b) > 1 and len(input_a) > len(input_b):
        aux = input_b
        input_b = input_a
        input_a = aux
        
    cum_results = np.empty([len(input_a), len(input_b)])
    for j, b in enumerate(input_b):
        if not isinstance(b, np.ndarray):
            b = np.array(b)
        
        for i, a in enumerate(input_a):
            if not isinstance(a, np.ndarray):
                a = np.array(a)
            
            result = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
            cum_results[i,j] = result[0][0]
    if aux:
        cum_results = cum_results.T
    return cum_results


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
        
        sim_matrix = cosine_memory_error_recover(self.x_train_list, X_test)
        argmax_array = np.argmax(sim_matrix, axis=0)
        labels = [self.y_train[idx] for idx in argmax_array]
        confidences = sim_matrix.take(argmax_array)
        return labels, confidences

    def predict_confidences(self, x_test, acronym, distinct_expansions):
        # classifier_proba = CalibratedClassifierCV(self.classifier, 'sigmoid', 'prefit')
        confidences_dict = {}
        sim_matrix = cosine_memory_error_recover(self.x_train_list, x_test.reshape(1, -1))
        for exp, conf in zip(distinct_expansions, sim_matrix):
            confidences_dict[exp] = conf[0]
        
        return confidences_dict




class _ExpanderCossimTerm(OutExpanderWithTermRepresentator):

    def fit(self, X_train, y_train):
        self.X_train = list(X_train)
        self.y_train = y_train

    def predict(self, X_test, acronym):

        sim_matrix = cosine_memory_error_recover(self.X_train, X_test.reshape(1, -1))
        argmax_array = np.argmax(sim_matrix, axis=0)
        labels = [self.y_train[idx] for idx in argmax_array]
        confidences = sim_matrix.take(argmax_array)
        return labels, confidences


    def predict_confidences(self, acronym_reresentation, acronym, distinct_expansions):
        # classifier_proba = CalibratedClassifierCV(self.classifier, 'sigmoid', 'prefit')
        confidences_dict = {}
        sim_matrix = cosine_memory_error_recover(self.X_train, acronym_reresentation.reshape(1, -1))
        for exp, conf in zip(distinct_expansions, sim_matrix):
            confidences_dict[exp] = conf[0]
        
        return confidences_dict

    def predict_proba(self, X_test, acronym):
        # classifier_proba = CalibratedClassifierCV(self.classifier, 'sigmoid', 'prefit')
        return cosine_memory_error_recover(self.X_train, X_test)
