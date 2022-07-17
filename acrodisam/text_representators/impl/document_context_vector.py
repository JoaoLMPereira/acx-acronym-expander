"""
Context Vector text representator for texts, each document/text has a context vector
"""
from collections import Counter
from typing import Optional

# from nltk.tokenize import word_tokenize
from sklearn.feature_extraction import DictVectorizer

from run_config import RunConfig
from inputters import TrainOutDataManager
from helper import TrainInstance, ExecutionTimeObserver
from text_preparation import get_expansion_without_spaces, preprocessed_text_tokenizer

import numpy as np

from .._base import TextRepresentatorFactory, TextRepresentator


class FactoryDocumentContextVector(
    TextRepresentatorFactory
):  # pylint: disable=too-few-public-methods
    """
    Text representator factory to generate context vectors from texts
    """

    def __init__(self, normalize_docs = False, run_config: Optional[RunConfig] = RunConfig()):
        self.normalize_docs = normalize_docs
        self.dataset_name = run_config.name
        self.save_and_load = run_config.save_and_load

    def get_text_representator(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: Optional[ExecutionTimeObserver] = None,
    ):

        vocabulary = DictVectorizer()

        # discover corpus and vectorize file word frequencies in a single pass
        if execution_time_observer:
            execution_time_observer.start()

        result_matrix = vocabulary.fit_transform(
            Counter(preprocessed_text_tokenizer(f))
            for f in train_data_manager.get_preprocessed_articles_db().values()
        )

        # get the maximum number of occurrences of the same word in a document,
        # this is for normalization purposes
        if self.normalize_docs:
            max_c = result_matrix.sum(axis=1).max()
        else:
            max_c = result_matrix.max()

        if execution_time_observer:
            execution_time_observer.stop()
        return _RepresentatorDocumentContextVector(
            train_data_manager.get_preprocessed_articles_db(), vocabulary, float(max_c)
        )


class _RepresentatorDocumentContextVector(TextRepresentator):
    """
    Represents each document by a context vector
    """

    def __init__(self, articles_db, vocabulary, max_c):
        super().__init__()
        self.vocabulary = vocabulary
        self.articles_db = articles_db
        self.max_c = max_c

    def tranform_test_instance(self, acronym_list, article):
        text = article.get_preprocessed_text()
        for acronym in acronym_list:
            yield self._transform_instance(acronym, text)

    def _transform_train_instance(self, train_instance):
        concept = get_expansion_without_spaces(train_instance.expansion)
        text = train_instance.getText(self.articles_db)
        return self._transform_instance(concept, text)

    def _transform_instance(self, concept, text):
        """
        Transforms an instance into a document context representation
        :param instance_x: train or test instance representing
         a document with an expansion or acronym
        """
        tokens = preprocessed_text_tokenizer(text.replace(concept, ""))

        return np.divide(
            self.vocabulary.transform(Counter(tokens)).toarray()[0], self.max_c
        )
