"""
Acronym out expander proposed in article entitled Acronym Disambiguation: A Domain Independent Approach
"""
from cachetools import cached, LRUCache, keys
import gensim.models
from gensim.models.doc2vec import TaggedDocument

from DataCreators.Doc2VecModel import preProcessText
from Logger import logging
from helper import TrainInstance, TestInstance
from inputters import TrainOutDataManager
from out_expanders._base import OutExpanderArticleInput
from text_preparation import get_expansion_without_spaces

from .._base import OutExpanderFactory, OutExpander

logger = logging.getLogger(__name__)


class FactoryDoc2VecPerAcronym(
    OutExpanderFactory
):  # pylint: disable=too-few-public-methods
    """
    Out expander factory to predict the expansion for an article based on doc2vec models per acronym
    """

    def __init__(self, *args, **kwargs):
        pass

    def get_expander(
        self, train_data_manager: TrainOutDataManager, execution_time_observer=None
    ):
        return _Doc2VecPerAcronym(train_data_manager.get_preprocessed_articles_db())


class _Doc2VecPerAcronym(OutExpander):
    def __init__(self, articlesDB):
        self.articlesDB = articlesDB
        self.cacheModelsPerAcronym = {}

    def _get_label(self, mostSimilarVec, X_train_vec, y_train_labels):
        most_similar_article_id = mostSimilarVec[0]

        for i, train_instance in enumerate(X_train_vec):
            if str(train_instance.article_id) == str(most_similar_article_id):
                return y_train_labels[i], mostSimilarVec[1]

        return None, None

    def _get_context(self, context_ix, text, max_length):
        side_max_length = max_length // 2

        if context_ix == -1:
            return None
        if (side_max_length - context_ix) >= 0:
            start = 0
        else:
            start = context_ix - side_max_length
        if (side_max_length + context_ix) <= len(text):
            end = context_ix + side_max_length
        else:
            end = len(text)
        return text[start:end]

    def _pre_process(self, instance):

        article_id = instance.article_id
        article_text = instance.getText(self.articlesDB)

        if isinstance(instance, TrainInstance):
            token = get_expansion_without_spaces(instance.expansion)
        else:
            token = instance.acronym

        context_ix = article_text.lower().find(token.lower())

        if context_ix < 0:
            raise Exception(
                "Token "
                + token.lower()
                + " not found in article "
                + article_id
                + " : "
                + article_text.lower()
            )
        para = self._get_context(context_ix, article_text, 5000)
        if para is None:
            return None

        tokens = preProcessText(para)

        td = TaggedDocument(tokens, [str(article_id)])
        return td

    def _compute_model(self, X_train_vec):
        taggeddocs = []
        for x in X_train_vec:
            taggeddocs = taggeddocs + [self._pre_process(x)]

        model = gensim.models.Doc2Vec(
            taggeddocs,
            dm=1,
            alpha=0.025,
            vector_size=500,
            min_alpha=0.025,
            min_count=0,
            workers=1,
        )
        for epoch in range(15):
            model.train(
                taggeddocs, total_examples=model.corpus_count, epochs=model.iter
            )
            model.min_alpha = model.alpha

        return model

    @cached(
        cache=LRUCache(maxsize=10),
        key=lambda _, acronym, X_train_vec: keys.hashkey(acronym),
    )
    def _get_model_with_cache(self, acronym, X_train_vec):
        return self._compute_model(X_train_vec)

    def _get_model(self, acronym, X_train_vec, cache=True):
        if cache:
            return self._get_model_with_cache(acronym, X_train_vec)
        else:
            return self._compute_model(X_train_vec)

    def _get_most_frequent_label(self, labels, confidences):
        if len(labels) == 1:
            return labels[0], confidences[0]

        frequencies = {}
        acc_confidences = {}
        for (label, confidence) in zip(labels, confidences):
            if label not in frequencies:
                frequencies[label] = 0
            frequencies[label] = frequencies[label] + 1

            if label not in acc_confidences:
                acc_confidences[label] = 0
            acc_confidences[label] = acc_confidences[label] + confidence

        label = max(frequencies, key=lambda key: frequencies[key])
        # We average for now
        confidence = acc_confidences[label] / frequencies[label]

        return label, confidence

    def predict(self, X_train_vec, y_train_labels, X_test, acronym):

        model = self._get_model(acronym, X_train_vec, cache=False)

        labels = []
        confidences = []
        for test_instance in X_test:

            taggedDoc = self._pre_process(test_instance)

            document_words = taggedDoc.words

            label = None
            confidence = None

            try:
                test_vect = model.infer_vector(document_words)
                mostSimilarVec = model.docvecs.most_similar([test_vect], topn=1)[0]
                [label, confidence] = self._get_label(
                    mostSimilarVec, X_train_vec, y_train_labels
                )

            except TypeError as err:
                logger.error(
                    "article_id="
                    + str(test_instance.article_id)
                    + " acronym="
                    + str(acronym),
                    err,
                )
                logger.error(str(y_train_labels))
            except KeyError as err:
                logger.error(
                    "article_id="
                    + str(test_instance.article_id)
                    + " acronym="
                    + str(acronym),
                    err,
                )
                logger.error(str(y_train_labels))

            if label == None:
                logger.error("Label = None")

            labels.append(label)
            confidences.append(confidence)

        return labels, confidences

    def process_article(self, out_expander_input: OutExpanderArticleInput):

        predicted_expansions = []

        x_train_list = out_expander_input.get_train_instances_list()

        y_train_list = out_expander_input.train_instances_expansions_list
        acronyms_list = out_expander_input.acronyms_list

        for acronym, x_train, y_train in zip(acronyms_list, x_train_list, y_train_list):

            test_instance = TestInstance(
                out_expander_input.test_article_id,
                out_expander_input.article.get_preprocessed_text(),
                acronym,
            )

            results, confidences = self.predict(
                x_train, y_train, [test_instance], acronym
            )
            result = results[0]
            confidence = confidences[0]

            predicted_expansions.append((result, confidence))
        return predicted_expansions
