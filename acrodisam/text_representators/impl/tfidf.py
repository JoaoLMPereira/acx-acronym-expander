"""
TFIDF text representator for texts
"""
from typing import Optional, Union, Sequence

from pydantic.decorator import validate_arguments
from typing_extensions import Literal

from DataCreators.TFIDFModel import getTFIDFModelForArticles
from acronym_expander import RunConfig
from helper import ExecutionTimeObserver

from .._base import (
    TextRepresentator,
    TextRepresentatorFactory,
    TextRepresentatorAcronymIndependent,
)
from inputters import TrainOutDataManager, InputArticle


LOG_FEATURES = "log(nub_distinct_words)+1"


class FactoryTFIDF(TextRepresentatorFactory):  # pylint: disable=too-few-public-methods
    """
    Text representator factory to create TFIDF models
    """

    # @validate_arguments
    def __init__(
        self,
        features_policy: Union[
            Literal[LOG_FEATURES], int, Sequence[Union[int, float]]
        ] = (1.0, 0),
        ngram_range: Sequence[int] = (1, 1),
        run_config: Optional[RunConfig] = RunConfig(),
    ):
        """
        :param features_policy: policy for the features to retain in TFIDF
         if one value assumes TFIDF max_features otherwise if is a list/tuple
          of two values assumes max_df and min_df respectively, default=(1.0, 0)
         Additional details for each policies below:
            max_features : int, default=None
                If not None, build a vocabulary that only consider the top
                max_features ordered by term frequency across the corpus.

            max_df : float or int
                When building the vocabulary ignore terms that have a document
                frequency strictly higher than the given threshold (corpus-specific
                stop words).
                If float in range [0.0, 1.0], the parameter represents a proportion of
                documents, integer absolute counts.

            min_df : float or int
                When building the vocabulary ignore terms that have a document
                frequency strictly lower than the given threshold. This value is also
                called cut-off in the literature.
                If float in range of [0.0, 1.0], the parameter represents a proportion
                of documents, integer absolute counts.

        :param ngram_range: tuple (min_n, max_n), default=(1, 1)
            The lower and upper boundary of the range of n-values for different
            n-grams to be extracted. All values of n such that min_n <= n <= max_n
            will be used. For example an ``ngram_range`` of ``(1, 1)`` means only
            unigrams, ``(1, 2)`` means unigrams and bigrams, and ``(2, 2)`` means
            only bigrams.
        :param run_config: general run configurations
        """

        """
        featuresParam = args[0]
        rangeParams = args[1]

        listFeaturesParams = featuresParam.split("-")
        listRangeparams = rangeParams.split("-")

        if listFeaturesParams[0] == "log(nub_distinct_words)+1":
            self.max_features = listFeaturesParams[0]
            self.max_df = 1.0
            self.min_df = 0

        elif int(listFeaturesParams[0]) == 0:
            self.max_features = None
            self.max_df = float(listFeaturesParams[1])
            self.min_df = int(listFeaturesParams[2])
        else:
            self.max_features = int(listFeaturesParams[0])
            self.max_df = 1.0
            self.min_df = 0

        self.ngram_range = (int(listRangeparams[0]), int(listRangeparams[1]))
        """
        if features_policy == LOG_FEATURES:
            self.max_features = features_policy
            self.max_df = 1.0
            self.min_df = 0
        else:
            try:
                self.max_features = int(features_policy)
                self.max_df = 1.0
                self.min_df = 0
            except (ValueError, TypeError):
                if not isinstance(features_policy, str):
                    self.max_features = None
                    self.max_df = float(features_policy[0])
                    self.min_df = int(features_policy[1])
                else:
                    raise ValueError(  # pylint: disable=raise-missing-from
                        "Invalid value for features_policy argument: " + features_policy
                    )

        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))
        self.dataset_name = run_config.name
        self.save_and_load = run_config.save_and_load

    def get_text_representator(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: Optional[ExecutionTimeObserver] = None,
    ) -> TextRepresentator:

        vectorizer = getTFIDFModelForArticles(
            train_data_manager.get_preprocessed_articles_db(),
            ngram_range=self.ngram_range,
            max_df=self.max_df,
            min_df=self.min_df,
            max_features=self.max_features,
            datasetName=self.dataset_name,
            fold=train_data_manager.get_fold(),
            saveAndLoad=self.save_and_load,
            executionTimeObserver=execution_time_observer,
        )
        return _RepresentatorTFIDF(
            train_data_manager.get_preprocessed_articles_db(), vectorizer
        )


class _RepresentatorTFIDF(
    TextRepresentatorAcronymIndependent
):  # pylint: disable=too-few-public-methods
    def __init__(self, articles_db, vectorizer):
        super().__init__()
        self.articles_db = articles_db
        self.vectorizer = vectorizer

    def _transform_input_text(self, article: InputArticle):
        return self.vectorizer.transform([article.get_preprocessed_text()]).toarray()[0]

    def _transform_train_instance(self, train_instance):
        text = train_instance.getText(self.articles_db)
        return self.vectorizer.transform([text]).toarray()[0]

    def transform(self, X):
        texts = [item.getText(self.articles_db) for item in X]
        return self.vectorizer.transform(texts)
