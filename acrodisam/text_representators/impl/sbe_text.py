"""
Created on Aug 29, 2018

@author: jpereira
"""
from typing import Optional

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from pydantic.decorator import validate_arguments
from pydantic.types import PositiveInt
from typing_extensions import Literal

from acronym_expander import RunConfig
import numpy as np
from text_preparation import get_expansion_without_spaces


from .._base import TextRepresentator, TextRepresentatorFactory
from inputters import InputArticle


class Factory_SBE(TextRepresentatorFactory):
    @validate_arguments
    def __init__(
        self,
        epoch: PositiveInt = 5,
        algorithm: Literal[0, 1] = 1,
        vector_size: PositiveInt = 200,
        word2vec_window_size: PositiveInt = 5,
        sbe_window_size: PositiveInt = 3,
        run_config: Optional[RunConfig] = RunConfig(),
    ):

        # SBE default epoch=5,  algorithm=1, vector_size=200, word2Vec_window_size=5, sbe_window_size=3
        self.dataset_name = run_config.name
        self.save_and_load = run_config.save_and_load

        self.epoch = epoch
        self.algorithm = algorithm

        self.vector_size = vector_size
        self.word2vec_window_size = word2vec_window_size
        self.sbe_window_size = sbe_window_size

    def get_text_representator(
        self,
        train_data_manager,
        execution_time_observer=None,
    ):
        tokenized_corpus = []
        for text in train_data_manager.get_preprocessed_articles_db().values():
            tokenized_corpus.append(word_tokenize(text))

        if execution_time_observer:
            execution_time_observer.start()
        word2vec_model = Word2Vec(
            tokenized_corpus,
            iter=self.epoch,
            size=self.vector_size,
            sg=self.algorithm,
            window=self.word2vec_window_size,
        )
        if execution_time_observer:
            execution_time_observer.stop()
        return RepresentatorSBE(train_data_manager.get_preprocessed_articles_db(), word2vec_model, self.sbe_window_size)


class RepresentatorSBE(TextRepresentator):
    """
    take doc2vec vectors of labelled articles
    """

    def __init__(self, articles_db, word2vec_model, window_size=3):
        super().__init__()
        self.articles_db = articles_db
        self.word2vec_model = word2vec_model
        self.window_size = window_size

    def tranform_test_instance(self, acronym_list, article: InputArticle):
        for acronym in acronym_list:
            yield self._transform_instance(acronym, article.get_preprocessed_text())

    def _transform_train_instance(self, train_instance):
        concept = get_expansion_without_spaces(train_instance.expansion)
        text = train_instance.getText(self.articles_db)
        return self._transform_instance(concept, text)

    def _transform_instance(self, concept, text):

        tmp_lst = text.split(concept)
        
        if len(tmp_lst) < 2:
            raise Exception("Acronym or Expansion: %s not found in text: %s" % (concept, text))

        vector_size = self.word2vec_model.vector_size

        tmp_ans = np.zeros(vector_size)
        # tmp_ans = array([0])
        for (idx, sen) in enumerate(tmp_lst):
            sen_lst = sen.split(" ")

            last_sen_lst = tmp_lst[idx - 1].split(" ")

            if idx != 0:
                tmp_cnt = 0
                for i in last_sen_lst[::-1]:
                    if i in self.word2vec_model:
                        tmp_cnt += 1
                        tmp_ans = tmp_ans + self.word2vec_model[i]
                        if tmp_cnt > self.window_size:
                            break

            tmp_cnt = 0
            for i in sen_lst:
                if i in self.word2vec_model:
                    tmp_cnt += 1
                    tmp_ans = tmp_ans + self.word2vec_model[i]
                    if tmp_cnt > self.window_size:
                        break

        return tmp_ans
