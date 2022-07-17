"""
In expander based on the paper:
Singh, Aadarsh, and Priyanshu Kumar. 
"SciDr at SDU-2020: IDEAS-Identifying and Disambiguating Everyday Acronyms for Scientific Domain."

Original code can be found in this repository:
https://github.com/aadarshsingh191198/AAAI-21-SDU-shared-task-1-AI

This in expander is an ensemble of CRFs that use the predictions of several SciBERT models and a rule based model as features for training.

Additional changes had to be perfomed to the code:
- Generalize to other datasets and external data sources
- Code refactoring
- Since the original work was proposed to a dataset containing sentences, we split the article text
 into sentences both for training and running the models.
"""
import logging
import pickle
from os.path import exists
import copy

import pandas as pd
import sklearn_crfsuite
import spacy
from DatasetParsers.process_tokens_and_bio_tags import create_diction
from helper import (
    getDatasetGeneratedFilesPath,
)
from sklearn.model_selection import KFold
from inputters import TrainInDataManager

from AcroExpExtractors.AcroExpExtractor import (
    AcroExpExtractorFactoryMl,
    AcroExpExtractorMl,
)
from AcroExpExtractors.AcroExpExtractor_Character_Match_AAAI import (
    AcroExpExtractor_Character_Match_AAAI,
)
from AcroExpExtractors.AcroExpExtractor_Scibert_Allennlp import (
    AcroExpExtractor_Scibert_Allennlp_Factory,
)
from AcroExpExtractors.AcroExpExtractor_Scibert_Sklearn import (
    AcroExpExtractor_Scibert_Sklearn_Factory,
)

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 30000000


def remove_control_characters(text):
    new_text = ""
    for c in text:
        if len(c.strip()) > 0:
            new_text += c
        else:
            new_text += " "
    return new_text


def create_features(x):
    return [word2features(x, i) for i in range(len(x["tokens"]))]


def word2features(x, i):
    word = x["tokens"][i]
    features = {
        "bias": 1.0,
        "word": word,
        "word[-3:]": word[-3:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
    }

    for j in range(len(x) - 1):
        features["model_" + str(j)] = x["model_" + str(j)][i]

    if i > 0:
        word1 = x["tokens"][i - 1]
        features.update(
            {
                "-1:word": word1,
                "-1:word.istitle()": word1.istitle(),
                "-1:word.isupper()": word1.isupper(),
            }
        )
    else:
        features["BOS"] = True

    if i < len(x["tokens"]) - 1:
        word1 = x["tokens"][i + 1]
        features.update(
            {
                "+1:word": word1,
                "+1:word.istitle()": word1.istitle(),
                "+1:word.isupper()": word1.isupper(),
            }
        )
    else:
        features["EOS"] = True

    return features


class AcroExpExtractor_Sci_Dr_Factory(AcroExpExtractorFactoryMl):
    def __init__(self, train_data_manager_base: TrainInDataManager, cuda: bool = False):
        super().__init__(train_data_manager_base)
        self.cuda = cuda

    def _split_training_data(self):

        self.train_data_manager_for_base_models = TrainInDataManager(
            self.training_data_name, "memory"
        )

        tmp_article_text = list(self.articles_raw_db.items())
        self.articles_raw_db = dict(tmp_article_text[len(tmp_article_text) // 2 :])
        self.train_data_manager_for_base_models.articles_raw_db = dict(
            tmp_article_text[: len(tmp_article_text) // 2]
        )

        for (
            article_id
        ) in self.train_data_manager_for_base_models.articles_raw_db.keys():
            self.train_data_manager_for_base_models.article_acronym_db[
                article_id
            ] = self.article_acronym_db.pop(article_id, {})

    def _get_base_models(self):

        self._split_training_data()

        base_models = []

        base_models.append(
            AcroExpExtractor_Scibert_Sklearn_Factory(
                copy.deepcopy(self.train_data_manager_for_base_models), 3, self.cuda
            ).get_in_expander()
        )

        base_models.append(
            AcroExpExtractor_Scibert_Sklearn_Factory(
                copy.deepcopy(self.train_data_manager_for_base_models), 4, self.cuda
            ).get_in_expander()
        )

        base_models.append(
            AcroExpExtractor_Scibert_Allennlp_Factory(
                copy.deepcopy(self.train_data_manager_for_base_models),
                10,
                False,
                False,
                self.cuda,
            ).get_in_expander()
        )

        base_models.append(
            AcroExpExtractor_Scibert_Allennlp_Factory(
                copy.deepcopy(self.train_data_manager_for_base_models),
                10,
                True,
                False,
                self.cuda,
            ).get_in_expander()
        )

        base_models.append(
            AcroExpExtractor_Scibert_Allennlp_Factory(
                copy.deepcopy(self.train_data_manager_for_base_models),
                20,
                True,
                True,
                self.cuda,
            ).get_in_expander()
        )

        base_models.append(AcroExpExtractor_Character_Match_AAAI())

        return base_models

    def _get_training_data(self, base_models):

        if exists(
            getDatasetGeneratedFilesPath(self.training_data_name)
            + "sci_dr_training_data.pickle"
        ):
            return pickle.load(
                open(
                    getDatasetGeneratedFilesPath(self.training_data_name)
                    + "sci_dr_training_data.pickle",
                    "rb",
                )
            )

        else:

            article_text = self.articles_raw_db.copy()

            acro_exp_dict = self.article_acronym_db.copy()

            tokens = []
            gold = []
            labels_base_models = []

            for i in range(len(base_models)):
                labels_base_models.append([])

            for id, article in article_text.items():

                doc = nlp(remove_control_characters(article))

                tokenized_article = []
                sentences = []

                for sentence in doc.sents:
                    tokenized_sentence = []
                    for token in sentence:
                        if len(token.text.strip()) > 0:
                            tokenized_sentence.append(token.text)
                            tokenized_article.append(token.text.strip().lower())
                    sentences.append(tokenized_sentence)

                acros = []
                exps = []

                for model in base_models:
                    model_predictions = model.get_all_acronym_expansion(article)
                    acros.append(list(model_predictions.keys()))
                    exps.append(list(filter(None, model_predictions.values())))

                acros.append(list(acro_exp_dict.get(id, {}).keys()))
                exps.append(list(filter(None, acro_exp_dict.get(id, {}).values())))

                for acro_list in acros:
                    for i in range(len(acro_list)):
                        acro_list[i] = [
                            t.text
                            for t in nlp(acro_list[i].strip().lower())
                            if len(t.text.strip()) > 0
                        ]

                for exp_list in exps:
                    for i in range(len(exp_list)):
                        exp_list[i] = [
                            t.text
                            for t in nlp(exp_list[i].strip().lower())
                            if len(t.text.strip()) > 0
                        ]

                labeled_article = []
                for j in range(len(base_models) + 1):
                    labeled_article.append(["O"] * len(tokenized_article))

                for j in range(len(base_models) + 1):
                    i = 0

                    while i < len(tokenized_article):
                        tagged = False

                        for acr in acros[j]:
                            if acr == tokenized_article[i : i + len(acr)]:
                                labeled_article[j][i : i + len(acr)] = ["B-short"] + [
                                    "I-short"
                                ] * (len(acr) - 1)

                                i = i + len(acr)
                                tagged = True
                                break

                        if not tagged:
                            for exp in exps[j]:
                                if exp == tokenized_article[i : i + len(exp)]:
                                    labeled_article[j][i : i + len(exp)] = [
                                        "B-long"
                                    ] + ["I-long"] * (len(exp) - 1)

                                    i = i + len(exp)
                                    tagged = True
                                    break
                        if not tagged:
                            i += 1

                index = 0
                for sentence in sentences:
                    tokens.append(sentence)

                    for i in range(len(labeled_article) - 1):
                        labels_base_models[i].append(
                            labeled_article[i][index : index + len(sentence)]
                        )

                    gold.append(
                        labeled_article[len(labeled_article) - 1][
                            index : index + len(sentence)
                        ]
                    )
                    index = index + len(sentence)

            data_dict = {"tokens": tokens, "gold": gold}

            for model_number, model_labels in enumerate(labels_base_models):
                data_dict["model_" + str(model_number)] = model_labels

            data = pd.DataFrame(data=data_dict)

            pickle.dump(
                data,
                open(
                    getDatasetGeneratedFilesPath(self.training_data_name)
                    + "sci_dr_training_data.pickle",
                    "wb",
                ),
                protocol=2,
            )

            return data

    def _train_model(self, data):

        crf_models = []

        kf = KFold(n_splits=5)
        for i, (train_index, test_index) in enumerate(kf.split(data)):

            train, val = data.iloc[train_index, :], data.iloc[test_index, :]

            train_X = train.drop("gold", axis=1).apply(create_features, axis=1)

            train_y = train["gold"].tolist()

            crf = sklearn_crfsuite.CRF(
                algorithm="lbfgs",
                c1=0.1,
                c2=0.1,
                max_iterations=500,
                all_possible_transitions=True,
            )
            crf.fit(train_X, train_y)

            crf_models.append(crf)

        pickle.dump(
            crf_models,
            open(
                getDatasetGeneratedFilesPath(self.training_data_name)
                + "sci_dr_crfs.pickle",
                "wb",
            ),
            protocol=2,
        )

        return crf_models

    def _check_model_exists(self):

        return exists(
            getDatasetGeneratedFilesPath(self.training_data_name) + "sci_dr_crfs.pickle"
        )

    def _get_trained_model(self):
        return pickle.load(
            open(
                getDatasetGeneratedFilesPath(self.training_data_name)
                + "sci_dr_crfs.pickle",
                "rb",
            )
        )

    def get_in_expander(self):
        if self._check_model_exists():
            crf_models = self._get_trained_model()
            base_models = self._get_base_models()
            return AcroExpExtractor_Sci_Dr(crf_models, base_models)
        else:
            base_models = self._get_base_models()
            data = self._get_training_data(base_models)
            crf_models = self._train_model(data)
            return AcroExpExtractor_Sci_Dr(crf_models, base_models)


class AcroExpExtractor_Sci_Dr(AcroExpExtractorMl):
    def __init__(
        self,
        crf_models,
        base_models,
    ):
        self.crf_models = crf_models
        self.base_models = base_models

    def _process_raw_input(self, text):
        doc = nlp(text)

        tokens = []
        labels_base_models = []
        tokenized_article = []
        sentences = []

        for i in range(len(self.base_models)):
            labels_base_models.append([])

        for sentence in doc.sents:
            tokenized_sentence = []
            for token in sentence:
                if len(token.text.strip()) > 0:
                    tokenized_sentence.append(token.text)
                    tokenized_article.append(token.text.strip().lower())
            sentences.append(tokenized_sentence)

        acros = []
        exps = []

        for model in self.base_models:
            model_predictions = model.get_all_acronym_expansion(text)
            acros.append(list(model_predictions.keys()))
            exps.append(list(filter(None, model_predictions.values())))

        for acro_list in acros:
            for i in range(len(acro_list)):
                acro_list[i] = [
                    t.text
                    for t in nlp(acro_list[i].strip().lower())
                    if len(t.text.strip()) > 0
                ]

        for exp_list in exps:
            for i in range(len(exp_list)):
                exp_list[i] = [
                    t.text
                    for t in nlp(exp_list[i].strip().lower())
                    if len(t.text.strip()) > 0
                ]

        labeled_article = []
        for j in range(len(self.base_models)):
            labeled_article.append(["O"] * len(tokenized_article))

        for j in range(len(self.base_models)):
            i = 0

            while i < len(tokenized_article):
                tagged = False

                for acr in acros[j]:
                    if acr == tokenized_article[i : i + len(acr)]:
                        labeled_article[j][i : i + len(acr)] = ["B-short"] + [
                            "I-short"
                        ] * (len(acr) - 1)

                        i = i + len(acr)
                        tagged = True
                        break

                if not tagged:
                    for exp in exps[j]:
                        if exp == tokenized_article[i : i + len(exp)]:
                            labeled_article[j][i : i + len(exp)] = ["B-long"] + [
                                "I-long"
                            ] * (len(exp) - 1)

                            i = i + len(exp)
                            tagged = True
                            break
                if not tagged:
                    i += 1

        index = 0
        for sentence in sentences:
            tokens.append(sentence)

            for i in range(len(labeled_article)):
                labels_base_models[i].append(
                    labeled_article[i][index : index + len(sentence)]
                )

            index = index + len(sentence)

        data_dict = {"tokens": tokens}

        for model_number, model_labels in enumerate(labels_base_models):
            data_dict["model_" + str(model_number)] = model_labels

        data = pd.DataFrame(data=data_dict)

        return data

    def _moder(self, x):
        master_l = pd.DataFrame([x[f"predictions_model_{i}"] for i in range(len(x))])
        return master_l.mode(axis=0).T[0].tolist()

    def _predict(self, data):

        predictions = []

        test_X = data.apply(create_features, axis=1)

        for crf in self.crf_models:
            predictions.append(crf.predict(test_X))

        data_dict = {}

        for i, model_prediction in enumerate(predictions):
            data_dict["predictions_model_" + str(i)] = model_prediction

        predictions_df = pd.DataFrame(data=data_dict)

        predictions_df["predictions_final"] = predictions_df.apply(self._moder, axis=1)

        full_input = []
        full_predictions = []

        for sentence in data.tokens.tolist():
            full_input.extend(sentence)

        for sentence in predictions_df.predictions_final.tolist():
            full_predictions.extend(sentence)

        return full_input, full_predictions

    def get_all_acronym_expansion(self, text):
        data_frame_input = self._process_raw_input(remove_control_characters(text))
        tokenized_input, bio_predictions = self._predict(data_frame_input)
        pairs = create_diction(tokenized_input, bio_predictions)
        return pairs

    def get_acronym_expansion_pairs(self, text):
        data_frame_input = self._process_raw_input(remove_control_characters(text))
        tokenized_input, bio_predictions = self._predict(data_frame_input)
        pairs = create_diction(tokenized_input, bio_predictions, all_acronyms=False)
        return pairs
