from typing import Optional

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from pydantic.decorator import validate_arguments
from pydantic.types import PositiveInt
from typing_extensions import Literal

from acronym_expander import RunConfig
from helper import ExecutionTimeObserver
from text_preparation import get_expansion_without_spaces
import numpy as np
from string_constants import FILE_WORD2VEC_GOOGLE_NEWS

from .._base import TermRepresentator, TermRepresentatorFactory
from inputters import TrainOutDataManager, InputArticle


class FactoryUAD(
    TermRepresentatorFactory
):  # pylint: disable=too-many-instance-attributes
    @validate_arguments
    def __init__(
        self,
        epoch: PositiveInt = 10,
        algorithm: Literal[0, 1] = 1,
        vector_size: PositiveInt = 300,
        word2vec_window_size: PositiveInt = 5,
        run_config: Optional[RunConfig] = RunConfig(),
    ):
        self.epoch = epoch
        self.algorithm = algorithm
        self.vector_size = vector_size
        self.word2vec_window_size = word2vec_window_size

        self.google_news = True

        self.dataset_name = run_config.name
        self.save_and_load = run_config.save_and_load

    def get_term_representator(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: Optional[ExecutionTimeObserver] = None,
    ) -> TermRepresentator:

        tokenized_corpus = []
        for text in train_data_manager.get_preprocessed_articles_db().values():

            tokenized_corpus.append(word_tokenize(text.lower()))

        if execution_time_observer:
            execution_time_observer.start()

        if self.google_news:
            word2vec_model = Word2Vec(
                min_count=1,
                iter=self.epoch,
                size=self.vector_size,
                sg=self.algorithm,
                window=self.word2vec_window_size,
                negative=5,
            )

            word2vec_model.build_vocab(
                tokenized_corpus
            )  # can be a non-repeatable, 1-pass generator

            word2vec_model.intersect_word2vec_format(
                FILE_WORD2VEC_GOOGLE_NEWS, lockf=1.0, binary=True
            )

            word2vec_model.train(
                tokenized_corpus,
                total_examples=word2vec_model.corpus_count,
                epochs=self.epoch,
            )

        else:
            word2vec_model = Word2Vec(
                tokenized_corpus,
                min_count=1,
                iter=self.epoch,
                size=self.vector_size,
                sg=self.algorithm,
                window=self.word2vec_window_size,
                negative=5,
            )

        if execution_time_observer:
            execution_time_observer.stop()
        return _RepresentatorUAD(word2vec_model, self.word2vec_window_size)


class _RepresentatorUAD(TermRepresentator):
    def __init__(self, word2vec_model, word2vec_window_size):
        self.word2vec_model = word2vec_model
        self.word2vec_window_size = word2vec_window_size

    def _get_vector(self, sentence):
        tmp_cnt = 0
        vector_size = self.word2vec_model.vector_size
        tmp_ans = np.zeros(vector_size)
        for i in sentence:
            if i in self.word2vec_model:
                tmp_cnt += 1
                tmp_ans = tmp_ans + self.word2vec_model[i]
                if tmp_cnt > self.word2vec_window_size:
                    return tmp_ans
        return tmp_ans

    def transform_test_sentences(self, acronym, text):
        """
        returns a vector for each sentence, i.e. fixed size window around each acronym occurrence
         in text
        """
        tmp_lst = text.split(acronym)

        last_sen_lst = []
        sentences_vect = []
        for (idx, sen) in enumerate(tmp_lst):
            sen_lst = sen.split(" ")

            sen_vec = self._get_vector(sen_lst)

            if idx != 0:
                last_sen_vec = self._get_vector(last_sen_lst[::-1])
                sen_vec = sen_vec + last_sen_vec

            last_sen_lst = sen_lst
            sentences_vect.append(sen_vec)

        return sentences_vect

    def tranform_acronym_terms(self, acronym_list: list, article: InputArticle):
        for acronym in acronym_list:
            yield sum(
                self.transform_test_sentences(acronym, article.get_preprocessed_text())
            )

    def tranform_expansion_terms(self, expansion_list, train_instance_list):
        for expansion in expansion_list:
            token = get_expansion_without_spaces(expansion)
            vec = self.word2vec_model[token.lower()]
            yield vec
