"""
In expander based on the paper:
Singh, Aadarsh, and Priyanshu Kumar. 
"SciDr at SDU-2020: IDEAS-Identifying and Disambiguating Everyday Acronyms for Scientific Domain."

Original code can be found in this repository:
https://github.com/aadarshsingh191198/AAAI-21-SDU-shared-task-1-AI

This in expander is one of the base models that SciDr (AcroExpExtractor_Sci_Dr) uses to get features for the training of its CRF ensemble.

Additional changes had to be perfomed to the code:
- Generalize to other datasets and external data sources
- Code refactoring
- Since the original work was proposed to a dataset containing sentences, we split the article text
 into sentences both for training and running the models.
"""

import pickle
from helper import (
    getDatasetGeneratedFilesPath,
)
from inputters import TrainInDataManager

from DatasetParsers.process_tokens_and_bio_tags import create_diction, bioless_to_bio

import logging

from AcroExpExtractors.AcroExpExtractor import (
    AcroExpExtractorFactoryMl,
    AcroExpExtractorMl,
)
import pandas as pd
import spacy
from os.path import exists
from bert_sklearn import BertTokenClassifier, load_model
import torch
import gc

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


class AcroExpExtractor_Scibert_Sklearn_Factory(AcroExpExtractorFactoryMl):
    def __init__(
        self, train_data_manager_base: TrainInDataManager, epochs: int = 15, cuda: bool = False
    ):
        super().__init__(train_data_manager_base)
        self.epochs = epochs
        self.cuda = cuda

    def _get_training_data(self):

        if exists(
            getDatasetGeneratedFilesPath(self.training_data_name)
            + "scibert_sklearn_training_data.pickle"
        ):
            return pickle.load(
                open(
                    getDatasetGeneratedFilesPath(self.training_data_name)
                    + "scibert_sklearn_training_data.pickle",
                    "rb",
                )
            )

        else:

            article_text = self.articles_raw_db.copy()

            acro_exp_dict = self.article_acronym_db.copy()

            tokens = []
            labels = []

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

                acros = list(acro_exp_dict.get(id, {}).keys())
                exps = list(filter(None, acro_exp_dict.get(id, {}).values()))

                for i in range(len(acros)):
                    acros[i] = [
                        t.text
                        for t in nlp(acros[i].strip().lower())
                        if len(t.text.strip()) > 0
                    ]

                for i in range(len(exps)):
                    exps[i] = [
                        t.text
                        for t in nlp(exps[i].strip().lower())
                        if len(t.text.strip()) > 0
                    ]

                labeled_article = ["O"] * len(tokenized_article)

                i = 0

                while i < len(tokenized_article):
                    tagged = False

                    for acr in acros:
                        if acr == tokenized_article[i : i + len(acr)]:
                            labeled_article[i : i + len(acr)] = ["short"] * len(acr)

                            i = i + len(acr)
                            tagged = True
                            break

                    if not tagged:
                        for exp in exps:
                            if exp == tokenized_article[i : i + len(exp)]:
                                labeled_article[i : i + len(exp)] = ["long"] * len(exp)

                                i = i + len(exp)
                                tagged = True
                                break
                    if not tagged:
                        i += 1

                index = 0
                for sentence in sentences:
                    tokens.append(sentence)
                    labels.append(labeled_article[index : index + len(sentence)])
                    index = index + len(sentence)

            data = pd.DataFrame(data={"tokens": tokens, "labels": labels})

            pickle.dump(
                data,
                open(
                    getDatasetGeneratedFilesPath(self.training_data_name)
                    + "scibert_sklearn_training_data.pickle",
                    "wb",
                ),
                protocol=2,
            )

            return data

    def _train_model(self, data):

        model = BertTokenClassifier(
            bert_model="scibert-scivocab-cased",
            epochs=self.epochs,
            learning_rate=1e-4,
            train_batch_size=8,
            eval_batch_size=8,
            validation_fraction=0.1,
            label_list=["O", "long", "short"],
            ignore_label=["O"],
            max_seq_length=272,
            gradient_accumulation_steps=2,
            use_cuda=self.cuda,
        )

        model.fit(data.tokens, data.labels)

        model_path = (
            getDatasetGeneratedFilesPath(self.training_data_name)
            + f"scibert_sklearn_{self.epochs}_epochs.bin"
        )

        model.save(model_path)

        model = None

        return model_path

    def _check_model_exists(self):

        return exists(
            getDatasetGeneratedFilesPath(self.training_data_name)
            + f"scibert_sklearn_{self.epochs}_epochs.bin"
        )

    def _get_trained_model_path(self):
        model_path = (
            getDatasetGeneratedFilesPath(self.training_data_name)
            + f"scibert_sklearn_{self.epochs}_epochs.bin"
        )

        return model_path

    def get_in_expander(self):
        if self._check_model_exists():
            model_path = self._get_trained_model_path()
            return AcroExpExtractor_Scibert_Sklearn(model_path)
        else:
            data = self._get_training_data()
            model_path = self._train_model(data)
            if self.cuda:
                gc.collect()
                torch.cuda.empty_cache()
            return AcroExpExtractor_Scibert_Sklearn(model_path)


class AcroExpExtractor_Scibert_Sklearn(AcroExpExtractorMl):
    def __init__(self, model_path):
        self.model_path = model_path

    def _process_raw_input(self, text):
        doc = nlp(text)

        tokenized_input = []

        for sentence in doc.sents:
            tokenized_sentence = []
            for token in sentence:
                if len(token.text.strip()) > 0:
                    tokenized_sentence.append(token.text)

            tokenized_input.append(tokenized_sentence)

        return pd.DataFrame(data={"tokens": tokenized_input})

    def _process_input_and_predictions(self, input, predictions):
        input = input.tokens.tolist()

        full_predictions = []
        full_input = []

        for sentence in input:
            full_input.extend(sentence)

        for i, pred in enumerate(predictions):
            for j, tag in enumerate(pred):
                if tag == None:
                    predictions[i][j] = "O"
            full_predictions.extend(pred)

        full_predictions = bioless_to_bio(full_predictions)

        return full_input, full_predictions

    def get_all_acronym_expansion(self, text):
        data_frame_input = self._process_raw_input(remove_control_characters(text))
        self.model = load_model(self.model_path)
        data_frame_predictions = self.model.predict(data_frame_input.tokens)
        tokenized_input, tokenized_predictions = self._process_input_and_predictions(
            data_frame_input, data_frame_predictions
        )
        pairs = create_diction(tokenized_input, tokenized_predictions)
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        return pairs

    def get_acronym_expansion_pairs(self, text):
        data_frame_input = self._process_raw_input(remove_control_characters(text))
        self.model = load_model(self.model_path)
        data_frame_predictions = self.model.predict(data_frame_input.tokens)
        tokenized_input, tokenized_predictions = self._process_input_and_predictions(
            data_frame_input, data_frame_predictions
        )
        pairs = create_diction(
            tokenized_input, tokenized_predictions, all_acronyms=False
        )
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        return pairs
