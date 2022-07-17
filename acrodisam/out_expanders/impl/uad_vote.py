"""
UAD based technique applied for each sentence contain an acronym and then soft voting is used to predict the final expansion
"""
from helper import ExecutionTimeObserver
from inputters import TrainOutDataManager
from out_expanders._base import OutExpanderArticleInput
from term_representators.impl.uad import FactoryUAD

from .._base import OutExpanderFactory
from .cossim import _ExpanderCossimTerm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class FactoryUADVote(OutExpanderFactory):
    def __init__(self, *args, **kw):
        self.representator_factory = FactoryUAD(*args, **kw)

    def get_expander(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: ExecutionTimeObserver = None,
    ):
        representator = self.representator_factory.get_term_representator(
            train_data_manager=train_data_manager,
            execution_time_observer=execution_time_observer,
        )
        return _ExpanderUADVote(representator)


class _ExpanderUADVote(_ExpanderCossimTerm):
    def __init__(self, representator):
        super().__init__(representator)
        self.representator = representator

    def predict_cosine(self, acronym_representation, acronym):

        sim_matrix = cosine_similarity(self.X_train, acronym_representation)
        argmax_array = np.argmax(sim_matrix, axis=0)
        labels = [self.y_train[idx] for idx in argmax_array]
        confidences = sim_matrix.take(argmax_array)
        return labels, confidences

    def predict_soft_vote(self, acronym_representation, acronym):
        sim_matrix = cosine_similarity(self.X_train, acronym_representation).sum(axis=1)
        argmax_idx = np.argmax(sim_matrix, axis=0)
        label = self.y_train[argmax_idx]
        confidence = sim_matrix.take(argmax_idx)
        return label, confidence / len(acronym_representation)

    def process_article(self, out_expander_input: OutExpanderArticleInput):

        predicted_expansions = []

        x_train_list = out_expander_input.get_train_instances_list()
        expansions_list = out_expander_input.distinct_expansions_list
        acronyms_list = out_expander_input.acronyms_list

        x_test_list = [
            self.representator.transform_test_sentences(
                acronym, out_expander_input.article.get_raw_text()
            )
            for acronym in acronyms_list
        ]

        for acronym, sentence_representation_list, expansions_set, x_train in zip(
            acronyms_list, x_test_list, expansions_list, x_train_list
        ):
            expansions = list(expansions_set)
            x_transformed = self.representator.tranform_expansion_terms(
                expansions, x_train
            )

            self.fit(x_transformed, expansions)

            result, confidence = self.predict_soft_vote(sentence_representation_list, acronym)

            predicted_expansions.append((result, confidence))
        return predicted_expansions
    
    
    def predict_confidences(self, acronym_representation, acronym, distinct_expansions):
        confidences_dict = {}
        sim_matrix = cosine_similarity(self.X_train, acronym_representation).sum(axis=1)
        for exp, conf in zip(distinct_expansions, sim_matrix):
            confidences_dict[exp] = conf / len(acronym_representation)
        
        return confidences_dict
    
    def process_article_return_confidences(self, out_expander_input: OutExpanderArticleInput):
        x_train_list = out_expander_input.get_train_instances_list()
        expansions_list = out_expander_input.distinct_expansions_list
        acronyms_list = out_expander_input.acronyms_list
        
        x_test_list = [
            self.representator.transform_test_sentences(
                acronym, out_expander_input.article.get_raw_text()
            )
            for acronym in acronyms_list
        ]
        
        confidences_list = []
        for acronym, sentence_representation_list, expansions_set, x_train in zip(acronyms_list,
                                                                            x_test_list,
                                                                            expansions_list,
                                                                            x_train_list):
            expansions = list(expansions_set)
            x_transformed = self.term_representator.tranform_expansion_terms(expansions, x_train)

            self.fit(x_transformed, expansions)

            confidences_dict = self.predict_confidences(sentence_representation_list,
                                                        acronym,
                                                        expansions_set)

            confidences_list.append(confidences_dict)

        return confidences_list
