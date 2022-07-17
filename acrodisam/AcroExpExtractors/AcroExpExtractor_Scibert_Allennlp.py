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

import json
import logging
import shutil
import tempfile
from argparse import Namespace
from os.path import exists

import spacy
import torch
from allennlp.commands.predict import _predict
from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.common.util import import_submodules
from DatasetParsers.process_tokens_and_bio_tags import (
    bioul_to_bio,
    biouless_to_bio,
    create_diction,
)
from helper import (
    getDatasetGeneratedFilesPath,
)
from inputters import TrainInDataManager
from string_constants import FOLDER_SCIBERT_CASED

from AcroExpExtractors.AcroExpExtractor import (
    AcroExpExtractorFactoryMl,
    AcroExpExtractorMl,
)
import gc

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 30000000

import_submodules("scibert")


def remove_control_characters(text):
    new_text = ""
    for c in text:
        if len(c.strip()) > 0:
            new_text += c
        else:
            new_text += " "
    return new_text


class AcroExpExtractor_Scibert_Allennlp_Factory(AcroExpExtractorFactoryMl):
    def __init__(
        self,
        train_data_manager_base: TrainInDataManager,
        epochs: int = 10,
        reduced_tags: bool = False,
        fine_tune: bool = False,
        cuda: bool = False,
    ):
        super().__init__(train_data_manager_base)
        self.epochs = epochs
        self.reduced_tags = reduced_tags
        self.fine_tune = fine_tune
        self.cuda = cuda

    def _get_training_data(self):

        train_path = (
            getDatasetGeneratedFilesPath(self.training_data_name)
            + f"scibert_allennlp_{self.reduced_tags}_bioless_training_data.txt"
        )
        validation_path = (
            getDatasetGeneratedFilesPath(self.training_data_name)
            + f"scibert_allennlp_{self.reduced_tags}_bioless_validation_data.txt"
        )

        if exists(train_path) and exists(validation_path):
            return train_path, validation_path

        else:

            article_text = self.articles_raw_db.copy()

            acro_exp_dict = self.article_acronym_db.copy()

            tokens = []
            labels = []
            pos_tags = []

            for id, article in article_text.items():

                doc = nlp(remove_control_characters(article))

                tokenized_article = []
                sentences = []

                for sentence in doc.sents:
                    tokenized_sentence = []
                    sentence_pos_tags = []
                    for token in sentence:
                        if len(token.text.strip()) > 0:
                            tokenized_sentence.append(token.text)
                            sentence_pos_tags.append(token.tag_)
                            tokenized_article.append(token.text.strip().lower())
                    sentences.append(tokenized_sentence)
                    pos_tags.append(sentence_pos_tags)

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

                            if self.reduced_tags:
                                labeled_article[i : i + len(acr)] = ["B-short"] * len(
                                    acr
                                )

                            else:
                                labeled_article[i : i + len(acr)] = ["B-short"] + [
                                    "I-short"
                                ] * (len(acr) - 1)

                            i = i + len(acr)
                            tagged = True
                            break

                    if not tagged:
                        for exp in exps:
                            if exp == tokenized_article[i : i + len(exp)]:

                                if self.reduced_tags:
                                    labeled_article[i : i + len(exp)] = [
                                        "B-long"
                                    ] * len(exp)

                                else:
                                    labeled_article[i : i + len(exp)] = ["B-long"] + [
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
                    labels.append(labeled_article[index : index + len(sentence)])
                    index = index + len(sentence)

            train_index = round(len(tokens) * 0.8)
            validation_index = len(tokens)

            with open(train_path, "w") as f:
                for i in range(train_index):
                    for j in range(len(tokens[i])):
                        f.write(f"{tokens[i][j]} {pos_tags[i][j]} O {labels[i][j]}")
                        f.write("\n")
                    f.write("\n")

            with open(validation_path, "w") as f:
                for i in range(train_index, validation_index):
                    for j in range(len(tokens[i])):
                        f.write(f"{tokens[i][j]} {pos_tags[i][j]} O {labels[i][j]}")
                        f.write("\n")
                    f.write("\n")

            return train_path, validation_path

    def _get_config_dict(self, train_path, valid_path):

        config_dict = {}

        cuda_device = -1

        if self.cuda and torch.cuda.is_available():
            cuda_device = 0

        if self.fine_tune:
            config_dict = {
                "random_seed": 13270,
                "pytorch_seed": 1327,
                "numpy_seed": 1327,
                "dataset_reader": {
                    "type": "conll2003",
                    "tag_label": "ner",
                    "coding_scheme": "BIOUL",
                    "token_indexers": {
                        "bert": {
                            "type": "bert-pretrained",
                            "pretrained_model": FOLDER_SCIBERT_CASED + "vocab.txt",
                            "do_lowercase": False,
                            "use_starting_offsets": True,
                        }
                    },
                },
                "train_data_path": train_path,
                "validation_data_path": valid_path,
                "test_data_path": valid_path,
                "evaluate_on_test": True,
                "model": {
                    "type": "bert_crf_tagger",
                    "label_encoding": "BIOUL",
                    "constrain_crf_decoding": True,
                    "calculate_span_f1": True,
                    "dropout": 0.1,
                    "include_start_end_transitions": False,
                    "text_field_embedder": {
                        "allow_unmatched_keys": True,
                        "embedder_to_indexer_map": {"bert": ["bert", "bert-offsets"]},
                        "token_embedders": {
                            "bert": {
                                "type": "bert-pretrained",
                                "pretrained_model": FOLDER_SCIBERT_CASED
                                + "weights.tar.gz",
                                "requires_grad": "all",
                                "top_layer_only": True,
                            }
                        },
                    },
                },
                "iterator": {
                    "type": "bucket",
                    "sorting_keys": [["tokens", "num_tokens"]],
                    "batch_size": 8,
                    "cache_instances": True,
                },
                "trainer": {
                    "optimizer": {
                        "type": "bert_adam",
                        "lr": 0.001,
                        "parameter_groups": [
                            [
                                [
                                    "bias",
                                    "LayerNorm.bias",
                                    "LayerNorm.weight",
                                    "layer_norm.weight",
                                ],
                                {"weight_decay": 0.0},
                            ]
                        ],
                    },
                    "validation_metric": "+f1-measure-overall",
                    "num_serialized_models_to_keep": 3,
                    "num_epochs": self.epochs,
                    "should_log_learning_rate": True,
                    "learning_rate_scheduler": {
                        "type": "slanted_triangular",
                        "num_epochs": self.epochs,
                        "num_steps_per_epoch": 1191,
                    },
                    "gradient_accumulation_batch_size": 16,
                    "cuda_device": cuda_device,
                },
            }

        else:
            config_dict = {
                "random_seed": 13270,
                "pytorch_seed": 1327,
                "numpy_seed": 1327,
                "dataset_reader": {
                    "type": "conll2003",
                    "tag_label": "ner",
                    "coding_scheme": "BIOUL",
                    "token_indexers": {
                        "bert": {
                            "type": "bert-pretrained",
                            "pretrained_model": FOLDER_SCIBERT_CASED + "vocab.txt",
                            "do_lowercase": False,
                            "use_starting_offsets": True,
                        }
                    },
                },
                "train_data_path": train_path,
                "validation_data_path": valid_path,
                "test_data_path": valid_path,
                "evaluate_on_test": True,
                "model": {
                    "type": "crf_tagger",
                    "label_encoding": "BIOUL",
                    "constrain_crf_decoding": True,
                    "calculate_span_f1": True,
                    "dropout": 0.5,
                    "include_start_end_transitions": False,
                    "text_field_embedder": {
                        "allow_unmatched_keys": True,
                        "embedder_to_indexer_map": {"bert": ["bert", "bert-offsets"]},
                        "token_embedders": {
                            "bert": {
                                "type": "bert-pretrained",
                                "pretrained_model": FOLDER_SCIBERT_CASED
                                + "weights.tar.gz",
                            }
                        },
                    },
                    "encoder": {
                        "type": "lstm",
                        "input_size": 768,
                        "hidden_size": 200,
                        "num_layers": 2,
                        "dropout": 0.5,
                        "bidirectional": True,
                    },
                },
                "iterator": {
                    "type": "bucket",
                    "sorting_keys": [["tokens", "num_tokens"]],
                    "batch_size": 8,
                    "cache_instances": True,
                },
                "trainer": {
                    "optimizer": {"type": "bert_adam", "lr": 0.001},
                    "validation_metric": "+f1-measure-overall",
                    "num_serialized_models_to_keep": 3,
                    "num_epochs": self.epochs,
                    "should_log_learning_rate": True,
                    "gradient_accumulation_batch_size": 16,
                    "patience": 10,
                    "cuda_device": cuda_device,
                },
            }

        return config_dict

    def _train_model(self, train_data_path, validation_data_path):

        config_dict = self._get_config_dict(train_data_path, validation_data_path)

        parameters = Params(config_dict)

        model_path = (
            getDatasetGeneratedFilesPath(self.training_data_name)
            + f"scibert_allennlp_{self.reduced_tags}_bioless_{self.fine_tune}_finetuned_{self.epochs}_epochs/"
        )

        if self.cuda and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        train_model(
            params=parameters,
            serialization_dir=model_path,
            force=True,
        )

        return model_path

    def _check_model_exists(self):

        return exists(
            getDatasetGeneratedFilesPath(self.training_data_name)
            + f"scibert_allennlp_{self.reduced_tags}_bioless_{self.fine_tune}_finetuned_{self.epochs}_epochs/"
        )

    def _get_trained_model_path(self):
        model_path = (
            getDatasetGeneratedFilesPath(self.training_data_name)
            + f"scibert_allennlp_{self.reduced_tags}_bioless_{self.fine_tune}_finetuned_{self.epochs}_epochs/"
        )

        return model_path

    def get_in_expander(self):
        if self._check_model_exists():
            model_path = self._get_trained_model_path()
            return AcroExpExtractor_Scibert_Allennlp(
                model_path, self.reduced_tags, self.fine_tune, self.cuda
            )
        else:
            train_data_path, validation_data_path = self._get_training_data()
            model_path = self._train_model(train_data_path, validation_data_path)
            if self.cuda and torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

            return AcroExpExtractor_Scibert_Allennlp(
                model_path, self.reduced_tags, self.fine_tune, self.cuda
            )


class AcroExpExtractor_Scibert_Allennlp(AcroExpExtractorMl):
    def __init__(self, model_path, reduced_tags, fine_tuned, cuda):
        self.model_path = model_path
        self.reduced_tags = reduced_tags
        self.fine_tuned = fine_tuned
        self.cuda = cuda

    def _create_tmp_dir_for_input_output(self):
        return tempfile.mkdtemp() + "/"

    def _delete_tmp_dir_for_input_output(self, tmp_dir):
        shutil.rmtree(tmp_dir)

    def _process_raw_input(self, text, tmp_dir_path):
        doc = nlp(text)

        tokenized_input = []
        pos_tags = []

        for sentence in doc.sents:
            tokenized_sentence = []
            sentence_pos_tags = []
            for token in sentence:
                if len(token.text.strip()) > 0:
                    tokenized_sentence.append(token.text)
                    sentence_pos_tags.append(token.tag_)

            tokenized_input.append(tokenized_sentence)
            pos_tags.append(sentence_pos_tags)

        input_path = tmp_dir_path + "input.txt"

        with open(input_path, "w") as f:
            for i in range(len(tokenized_input)):
                for j in range(len(tokenized_input[i])):
                    f.write(f"{tokenized_input[i][j]} {pos_tags[i][j]} O O")
                    f.write("\n")
                f.write("\n")

        return input_path, tokenized_input

    def _process_input_and_predictions(self, tokenized_input, output_path):
        full_input = []

        for sentence in tokenized_input:
            full_input.extend(sentence)

        full_tags = []

        f = open(output_path)
        predictions = f.readlines()
        f.close()

        for prediction in predictions:
            prediction = json.loads(prediction)

            if self.reduced_tags:
                full_tags.extend(biouless_to_bio(prediction["tags"]))
            else:
                full_tags.extend(bioul_to_bio(prediction["tags"]))

        return full_input, full_tags

    def _get_config_dict(self, input_path, output_path):
        config_dict = {}

        cuda_device = -1

        if self.cuda and torch.cuda.is_available():
            cuda_device = 0

        if self.fine_tuned:
            config_dict = {
                "archive_file": self.model_path,
                "input_file": input_path,
                "use_dataset_reader": True,
                "output_file": output_path,
                "batch_size": 8,
                "cuda_device": cuda_device,
                "weights_file": "",
                "silent": True,
                "dataset_reader_choice": "train",
                "overrides": "",
                "predictor": "sentence-tagger",
            }

        else:
            config_dict = {
                "archive_file": self.model_path,
                "input_file": input_path,
                "use_dataset_reader": True,
                "output_file": output_path,
                "batch_size": 8,
                "cuda_device": cuda_device,
                "weights_file": "",
                "silent": True,
                "dataset_reader_choice": "train",
                "overrides": "",
                "predictor": "",
            }
        return config_dict

    def _predict(self, input_path, output_path):
        config_dict = self._get_config_dict(input_path, output_path)
        _predict(Namespace(**config_dict))

    def get_all_acronym_expansion(self, text):
        tmp_dir_path = self._create_tmp_dir_for_input_output()
        input_path, tokenized_sentences_input = self._process_raw_input(
            remove_control_characters(text), tmp_dir_path
        )
        output_path = tmp_dir_path + "output.txt"
        self._predict(input_path, output_path)
        tokenized_input, bio_predictions = self._process_input_and_predictions(
            tokenized_sentences_input, output_path
        )
        pairs = create_diction(tokenized_input, bio_predictions)
        self._delete_tmp_dir_for_input_output(tmp_dir_path)
        if self.cuda and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        return pairs

    def get_acronym_expansion_pairs(self, text):
        tmp_dir_path = self._create_tmp_dir_for_input_output()
        input_path, tokenized_sentences_input = self._process_raw_input(
            remove_control_characters(text), tmp_dir_path
        )
        output_path = tmp_dir_path + "output.txt"
        self._predict(input_path, output_path)
        tokenized_input, bio_predictions = self._process_input_and_predictions(
            tokenized_sentences_input, output_path
        )
        pairs = create_diction(tokenized_input, bio_predictions, all_acronyms=False)
        self._delete_tmp_dir_for_input_output(tmp_dir_path)
        if self.cuda and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        return pairs
