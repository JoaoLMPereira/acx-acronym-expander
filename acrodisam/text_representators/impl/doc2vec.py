"""
Doc2vec text representator for texts, each document/text has a unique identifier
"""
from typing import Optional

from pydantic import validate_arguments, PositiveInt
from typing_extensions import Literal

from DataCreators.Doc2VecModel import trainDoc2VecModel, preProcessText
from Logger import logging
from run_config import RunConfig
from inputters import TrainOutDataManager

from .._base import TextRepresentatorAcronymIndependent, TextRepresentatorFactory
from helper import ExecutionTimeObserver


# replace typing_extensions by typing in python 3.7+
logger = logging.getLogger(__name__)


class FactoryDoc2Vec(
    TextRepresentatorFactory
):  # pylint: disable=too-few-public-methods
    """
    Text representator factory to create Do2Vec models
    """

    @validate_arguments
    def __init__(  # pylint: disable=too-many-arguments
        self,
        epoch: PositiveInt = 50,
        algorithm: Literal["Skip-gram", "CBOW"] = "CBOW",
        vector_size: PositiveInt = 200,
        window_size: PositiveInt = 8,
        run_config: Optional[RunConfig] = RunConfig(),
    ):
        """

        :param epoch: Number of iterations (epochs) over the corpus
        :param algorithm: Defines the training algorithm
        :param vector_size: Dimensionality of the feature vectors
        :param window_size: The maximum distance between the current
        and predicted word within a sentence
        :param run_config: general run configurations
        """

        # self._set_default_values(run_config.name)

        self.epoch = epoch
        if algorithm == "Skip-gram":
            self.algorithm = 1
        elif algorithm == "CBOW":
            self.algorithm = 0
        else:
            raise TypeError("Algorithm value not known: %s" % str(algorithm))

        self.vector_size = vector_size
        self.window_size = window_size

        self.run_config = run_config

    def get_text_representator(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: Optional[ExecutionTimeObserver] = None,
    ):

        doc2vec_model, tags_map = trainDoc2VecModel(
            articleDB=train_data_manager.get_preprocessed_articles_db(),
            epochs=self.epoch,
            dm=self.algorithm,
            vector_size=self.vector_size,
            window=self.window_size,
            datasetName=self.run_config.name,
            fold=train_data_manager.get_fold(),
            saveAndLoad=self.run_config.save_and_load,
            persistentArticles=self.run_config.persistent_articles,
            executionTimeObserver=execution_time_observer,
        )
        return _RepresentatorDoc2Vec(doc2vec_model, tags_map)


class _RepresentatorDoc2Vec(TextRepresentatorAcronymIndependent):
    def __init__(self, doc2vec_model, tags_map=None):
        super().__init__()
        self.doc2vec_model = doc2vec_model
        if tags_map is None:
            self.convert_tag2id = lambda tag: tag
        else:
            self.convert_tag2id = lambda tag: tags_map[tag]

    def _transform_input_text(self, article):
        preprocessed_input = preProcessText(article.get_preprocessed_text())
        vect = self.doc2vec_model.infer_vector(preprocessed_input)
        return vect

    def _transform_train_instance(self, train_instance):
        doc_id = self.convert_tag2id(str(train_instance.article_id))
        vect = self.doc2vec_model[doc_id]
        return vect
