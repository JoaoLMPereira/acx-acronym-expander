"""
LDA text representator for texts, each document/text is represented by a set o latent topics
"""
from typing import Union, Optional

from gensim.matutils import sparse2full
from pydantic.decorator import validate_arguments
from typing_extensions import Literal

from DataCreators.LDAModel import create_model
from DataCreators.LDAModel import preProcessText
from acronym_expander import RunConfig
from helper import TrainInstance, ExecutionTimeObserver

from .._base import (
    TextRepresentator,
    TextRepresentatorAcronymIndependent,
    TextRepresentatorFactory,
)
from inputters import TrainOutDataManager, InputArticle


class FactoryLDA(TextRepresentatorFactory):  # pylint: disable=too-few-public-methods
    """
    Text representator factory to create LDA models
    """

    @validate_arguments
    def __init__(
        self,
        epochs: int = 1,
        num_topics: Union[Literal["log(nub_distinct_words)+1"], int] = 100,
        run_config: Optional[RunConfig] = RunConfig(),
    ):
        """

        :param epochs: Number of passes (epochs) through the corpus during training default=1
        :param num_topics: The number of requested latent topics to be extracted from the training
         corpus, default=100
        :param run_config: general run configurations
        """

        self.epochs = epochs
        self.num_topics = num_topics

        self.dataset_name = run_config.name
        self.save_and_load = run_config.save_and_load
        self.persistent_articles = run_config.persistent_articles

    def get_text_representator(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: Optional[ExecutionTimeObserver] = None,
    ) -> TextRepresentator:

        lda_model = create_model(
            process_pool=None,
            datasetName=self.dataset_name,
            articleDB=train_data_manager.get_preprocessed_articles_db(),
            num_topics=self.num_topics,
            numPasses=self.epochs,
            fold=train_data_manager.get_fold(),
            saveAndLoad=self.save_and_load,
            persistentArticles=self.persistent_articles,
            executionTimeObserver=execution_time_observer,
        )
        return _RepresentatorLDA(lda_model)


class _RepresentatorLDA(
    TextRepresentatorAcronymIndependent
):  # pylint: disable=too-few-public-methods
    """
    take LDA vectors of labelled articles
    """

    def __init__(self, lda_model):
        super().__init__()

        self.lda_model = lda_model.ldaModel
        self.dictionary = lda_model.dictionary
        self.article_id_to_lda_dict = lda_model.articleIDToLDADict

    def _transform_input_text(self, article: InputArticle):
        cleaned_words = preProcessText(article.get_preprocessed_text())
        bow = self.dictionary.doc2bow(cleaned_words)
        lda_vector = self.lda_model[bow]
        return self._get_dense_vector(lda_vector)

    def _transform_train_instance(self, train_instance: TrainInstance):
        lda_vector = self.article_id_to_lda_dict[train_instance.article_id]
        return self._get_dense_vector(lda_vector)

    def _get_dense_vector(self, sparse_vec):
        return sparse2full(sparse_vec, self.lda_model.num_topics)
