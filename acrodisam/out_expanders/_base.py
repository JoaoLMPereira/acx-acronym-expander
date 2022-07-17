from abc import ABCMeta, abstractmethod

import helper
import numpy as np
from string_constants import min_confidence
from text_representators import TextRepresentator
from text_representators import get_text_representator_factory
from typing import List
from helper import TestInstance, TrainInstance, ExecutionTimeObserver
from term_representators import TermRepresentator
from inputters import TrainOutDataManager, InputArticle


class OutExpanderArticleInput:
    def __init__(self, test_article_id, article: InputArticle):
        self.test_article_id = test_article_id
        self.article = article

        self.acronyms_list = []
        self.train_instances_ids_list = []
        self.train_instances_expansions_list = []
        self.distinct_expansions_list = []

    def add_acronym(
        self,
        acronym,
        train_intances_ids,
        train_instance_expansions,
        distinct_expansions,
    ):

        self.acronyms_list.append(acronym)
        self.train_instances_ids_list.append(train_intances_ids)
        self.train_instances_expansions_list.append(train_instance_expansions)
        self.distinct_expansions_list.append(distinct_expansions)

    def get_test_instances(self):
        for acronym in self.acronyms_list:
            yield TestInstance(
                self.test_article_id, self.article.get_preprocessed_text(), acronym
            )

    def _get_train_instances(self, acronym, ids_list, exp_list):
        for article_id, exp in zip(ids_list, exp_list):
            yield TrainInstance(article_id, acronym, exp)

    def get_train_instances_list(self):
        for acronym, ids_list, exp_list in zip(
            self.acronyms_list,
            self.train_instances_ids_list,
            self.train_instances_expansions_list,
        ):
            yield list(self._get_train_instances(acronym, ids_list, exp_list))

    def iterator_all(self):
        return zip(
            self.acronyms_list,
            self.train_instances_ids_list,
            self.train_instances_expansions_list,
            self.distinct_expansions_list,
        )


class OutExpander(metaclass=ABCMeta):
    
    def process_article_return_confidences(self, out_expander_input: OutExpanderArticleInput):
        """
            Models should always override this method whenever it is possible

            For a given input (document and additional data like the acronyms present)
            returns the confidences for each acronym and known expansion (that is available in the 
            DB)
            confidences can be a probability between 0 and 1
            else is any real value whose expansion with max value is the predicted expansions
            when any of those is possible, 1 is assigned to the predicted expansion
              and 0 to the others
        """
        predicted_expansions = self.process_article(out_expander_input)
        confidences_list = []
        for predicted_expansion, distinct_expansions in zip(predicted_expansions, 
                                                            out_expander_input.distinct_expansions_list):
            confidences_dict = {}
            for expansion in distinct_expansions:
                if expansion == predicted_expansion[0]:
                    confidences_dict[expansion] = predicted_expansion[1] if predicted_expansion[1] > 0 else 1
                else:
                    confidences_dict[expansion] = 0
            confidences_list.append(confidences_dict)
        return confidences_list
    
    @abstractmethod
    def process_article(self, out_expander_input: OutExpanderArticleInput):
        """
        test_instances_list = out_expander_input.get_test_instances()
        x_train_list = out_expander_input.get_train_instances_list()
        y_train_list = out_expander_input.train_instances_expansions_list
        expansions = []
        for test_instance, x_train, y_train in zip(test_instances_list, x_train_list, y_train_list):
            x_transformed = self.transform(x_train)

            self.fit(x_transformed, y_train)

            x_test = self.transform([test_instance])

            results, confidences = self.predict(x_test, test_instance.acronym)
            result = results[0]
            confidence = confidences[0]

            expansions.append((result, confidence))
        return expansions
        """

        # @abstractmethod
        # def transform(self, X):
        """
        transforms input list to form accepted by fit and predict function
        inputs:
        X (list): of helper.ExpansionChoice
        returns:
        result (list): of inputs to predict and fit functions
        """

        # @abstractmethod
        # def fit(self, X_train, y_train):
        """
        fits the current algo to training data
        inputs:
        X_train (list): of training input variables
        y_train (list): of training labels
        """

        # @abstractmethod
        # def predict(self, X_test, acronym):
        """
        predicts the labels possible for test data
        inputs:
        X_test (list): of input data
        acronym (unicode): the acronym for which the expansion is being predicted
        returns:
        labels (list): of labels for test data
        confidences (list): corresponding un-normalized confidence values
        """


class OutExpanderWithTermRepresentator(OutExpander):
    def __init__(self, term_representator: TermRepresentator):
        super().__init__()
        self.term_representator = term_representator

        # @abstractmethod
        # def transform(self, X):
        """
        transforms input list to form accepted by fit and predict function
        inputs:
        X (list): of helper.ExpansionChoice
        returns:
        result (list): of inputs to predict and fit functions
        """

    @abstractmethod
    def fit(self, X_train, y_train):
        """
        fits the current algo to training data
        inputs:
        X_train (list): of training input variables
        y_train (list): of training labels
        """

    @abstractmethod
    def predict(self, acronym_representation, acronym):
        """
        predicts the labels possible for test data
        inputs:
        X_test (list): of input data
        acronym (unicode): the acronym for which the expansion is being predicted
        returns:
        labels (list): of labels for test data
        confidences (list): corresponding un-normalized confidence values
        """

    def predict_confidences(self, acronym_representation, acronym, distinct_expansions):
        """
            Models should always override this method whenever it is possible

            Accepts an acronym representation (usually a list or np.array), the acronym 
            and the possible expansions
            Returns a dict of expansions with the confidences
        """
        results, confidences = self.predict(acronym_representation, acronym)
        predicted_expansion = results[0]

        confidences_dict = {}
        for expansion in distinct_expansions:
            if expansion == predicted_expansion:
                confidences_dict[expansion] = confidences[0] if confidences[0] > 0 else 1
            else:
                confidences_dict[expansion] = 0
        return confidences_dict


    def process_article_return_confidences(self, out_expander_input: OutExpanderArticleInput):
        x_train_list = out_expander_input.get_train_instances_list()
        expansions_list = out_expander_input.distinct_expansions_list
        acronyms_list = out_expander_input.acronyms_list
        confidences_list = []
        x_test_list = self.term_representator.tranform_acronym_terms(acronyms_list,
                                                                     out_expander_input.article)

        for acronym, acronym_representation, expansions_set, x_train in zip(acronyms_list,
                                                                            x_test_list,
                                                                            expansions_list,
                                                                            x_train_list):
            expansions = list(expansions_set)
            x_transformed = self.term_representator.tranform_expansion_terms(expansions, x_train)

            self.fit(x_transformed, expansions)

            confidences_dict = self.predict_confidences(acronym_representation,
                                                        acronym,
                                                        expansions_set)

            confidences_list.append(confidences_dict)

        return confidences_list


    def process_article(self, out_expander_input: OutExpanderArticleInput):

        predicted_expansions = []

        # test_instances_list = out_expander_input.get_test_instances()
        x_train_list = out_expander_input.get_train_instances_list()
        # y_train_list = out_expander_input.train_instances_expansions_list
        expansions_list = out_expander_input.distinct_expansions_list
        acronyms_list = out_expander_input.acronyms_list
        
        x_test_list = self.term_representator.tranform_acronym_terms(acronyms_list, out_expander_input.article)
        
        #trasnformed_intances = self.transform(test_instances_list, x_train_list, y_train_list)
        for acronym, acronym_representation, expansions_set, x_train in zip(acronyms_list, x_test_list, expansions_list, x_train_list ):
            expansions = list(expansions_set)
            # x_transformed = self.transform(x_train)
            x_transformed = self.term_representator.tranform_expansion_terms(
                expansions, x_train
            )

            self.fit(x_transformed, expansions)

            results, confidences = self.predict(acronym_representation, acronym)
            result = results[0]
            confidence = confidences[0]

            predicted_expansions.append((result, confidence))
        return predicted_expansions


# This is an abstract class, we don't need to override methods in parent abstract class
# following line disables that warning in pylint
# pylint: disable=W0223
class OutExpanderWithTextRepresentator(OutExpander):
    """
    for expanders which use a machine learning algo to expand acronyms
    """

    def __init__(self, text_representator: TextRepresentator):
        super().__init__()
        self.text_representator = text_representator

    @abstractmethod
    def fit(self, X_train, y_train):
        """
        fits the current algo to training data
        inputs:
        X_train (list): of training input variables
        y_train (list): of training labels
        """

    @abstractmethod
    def predict(self, X_test, acronym):
        """
        predicts the labels possible for test data
        inputs:
        X_test (list): of input data
        acronym (unicode): the acronym for which the expansion is being predicted
        returns:
        labels (list): of labels for test data
        confidences (list): corresponding un-normalized confidence values
        """

    # def transform(self, X):
    # if self.text_representator is not None:
    #    return self.text_representator.transform(X)
    # else:
    #    return X

    def process_article(self, out_expander_input: OutExpanderArticleInput):
        # test_instances_list = out_expander_input.get_test_instances()
        x_train_list = out_expander_input.get_train_instances_list()
        y_train_list = out_expander_input.train_instances_expansions_list
        expansions = []
        x_test_list = self.text_representator.tranform_test_instance(
            out_expander_input.acronyms_list, out_expander_input.article
        )
        for acronym, x_test, x_train, y_train in zip(
            out_expander_input.acronyms_list, x_test_list, x_train_list, y_train_list
        ):
            x_transformed = self.text_representator.tranform_train_instances(x_train)

            self.fit(x_transformed, y_train)

            results, confidences = self.predict(x_test, acronym)
            result = results[0]
            confidence = confidences[0]

            expansions.append((result, confidence))
        return expansions


    def predict_confidences(self, text_representation, acronym, distinct_expansions):
        """
            Models should always override this method whenever it is possible

            Accepts a text representation (usually a list or np.array) of the input
            , the acronym and the possible expansions
            Returns a dict of expansions with the confidences
        """

        results, confidences = self.predict(text_representation, acronym)
        predicted_expansion = results[0]

        confidences_dict = {}
        for expansion in distinct_expansions:
            if expansion == predicted_expansion:
                confidences_dict[expansion] = confidences[0] if confidences[0] > 0 else 1
            else:
                confidences_dict[expansion] = 0
        return confidences_dict

    def process_article_return_confidences(self, out_expander_input: OutExpanderArticleInput):
        x_train_list = out_expander_input.get_train_instances_list()
        y_train_list = out_expander_input.train_instances_expansions_list
        dictinct_exp_list = out_expander_input.distinct_expansions_list
        confidences_list = []
        x_test_list = self.text_representator.tranform_test_instance(
            out_expander_input.acronyms_list,
            out_expander_input.article)

        for acronym, x_test, x_train, y_train, distinct_expansions in zip(
            out_expander_input.acronyms_list,
            x_test_list,
            x_train_list,
            y_train_list,
            dictinct_exp_list):

            x_transformed = self.text_representator.tranform_train_instances(x_train)
            self.fit(x_transformed, y_train)
            confidences_dict = self.predict_confidences(x_test, acronym, distinct_expansions)
            confidences_list.append(confidences_dict)

        return confidences_list


    def _get_confidences_from_decision_function(self, labels, decisions):
        confidences = []
        for label, decision in zip(labels, decisions):
            confidence = min_confidence
            if hasattr(decision, "__iter__"):
                cl = self.classifier.classes_
                indx = np.where(cl == label)[0][0]
                confidence = decision[indx]
            else:
                confidence = abs(decision)
            confidences.append(confidence)

        return confidences

    # Implementation based on the random forest original predict method
    def _get_labels_and_confidences_from_proba(self, proba):
        idx_max = np.argmax(proba, axis=1)
        return self.classifier.classes_.take(idx_max, axis=0), proba.take(idx_max)


class OutExpanderFactory(metaclass=ABCMeta):
    @abstractmethod
    def get_expander(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: ExecutionTimeObserver = None,
    ):
        pass


# This is an abstract class, we don't need to override methods in parent abstract class
# following line disables that warning in pylint
# pylint: disable=W0223
class OutExpanderWithTextRepresentatorFactory(OutExpanderFactory):
    """
    to create expanders that a machine learning algo and a text extractor generative model (unsupervised) to expand acronyms
    """

    def __init__(
        self,
        text_representator: str = "document_context_vector",
        text_representator_args=None,
        *args,
        **kwargs
    ):
        """

        :param text_representator: Text representator name
        :param text_representator_args: Arguments for the text-representator
        """
        new_args, new_kwargs = helper.get_args_to_pass(
            text_representator_args, args, kwargs
        )

        self.representator_factory = get_text_representator_factory(
            text_representator, *new_args, **new_kwargs
        )

    def _get_representator(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: ExecutionTimeObserver = None,
    ) -> TextRepresentator:
        return self.representator_factory.get_text_representator(
            train_data_manager=train_data_manager,
            execution_time_observer=execution_time_observer,
        )
