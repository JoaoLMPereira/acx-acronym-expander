"""
MadDog out-expander
We had to override some methods to use non-hard coded file paths
 and support training from scratch using any data source

Setup:
run  pip install -e git+https://github.com/amirveyseh/MadDog.git#egg=prototype

Created on Apr 30, 2021

@author: jpereira
"""

import json
from prototype.acronym.zeroshot.model.utils import torch_utils
from prototype.acronym.zeroshot.model.utils.vocab import Vocab
import prototype.acronym.zeroshot.zeroshot
import random
from typing import Optional

from scipy.special import softmax
import torch
from tqdm import tqdm

from DataCreators.maddog_data import nlp, DataLoader
import DataCreators.maddog_models
from Logger import logging
from helper import getDatasetGeneratedFilesPath, ExecutionTimeObserver
from inputters import TrainOutDataManager
from out_expanders._base import OutExpanderArticleInput
from out_expanders.impl.sci_dr import Devices
from run_config import RunConfig
from string_constants import FOLDER_MADDOG_PRE_TRAINED_MODELS

from .._base import OutExpanderFactory, OutExpander
from DataCreators.maddog_models import load_torch_models


logger = logging.getLogger(__name__)


# Function taken from ...
def deep_strip(text):
    new_text = ""
    for c in text:
        if len(c.strip()) > 0:
            new_text += c
        else:
            new_text += " "
    new_text = new_text.replace('"', "'")
    return new_text


class MadDogFactory(OutExpanderFactory):  # pylint: disable=too-few-public-methods
    """
    Out expander factory to predict the expansion for an article based on doc2vec models per acronym
    """

    def __init__(
        self,
        use_original_models: bool = False,
        device: Devices = "auto",
        run_config: Optional[RunConfig] = RunConfig(),
        **kwargs,
    ):
        self.use_original_models = use_original_models
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available else "cpu"
        else:
            self.device = device.lower()
        self.run_name = run_config.name

    def get_expander(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: ExecutionTimeObserver = None,
    ):
        if (
            train_data_manager.get_fold() is not None
            and train_data_manager.get_fold() != "TrainData"
        ):
            raise NotImplementedError(
                "This out-expander uses its own cross-validation. Not accepting fold: %s"
                % train_data_manager.get_fold()
            )

        if self.use_original_models:
            return _MadDog(FOLDER_MADDOG_PRE_TRAINED_MODELS, chunck_num=483)

        models_path = getDatasetGeneratedFilesPath(self.run_name)

        execution_time_observer.start()
        chunck_num = DataCreators.maddog_models.create_models(
            models_path, train_data_manager, self.device
        )
        execution_time_observer.stop()

        return _MadDog(models_path, chunck_num=chunck_num)


def last_index(lst, value):
    for i in range(len(lst) - 1, -1, -1):
        if value == lst[i]:
            return i

    return None


class ModelLoader(prototype.acronym.zeroshot.load_models.ModelLoader):
    def __init__(self, dir_path, chuncks_num):
        torch.manual_seed(1234)
        random.seed(1234)
        self.trainers = {}
        self.vocabs = {}
        self.opts = {}
        self.labels = {}

        for i in tqdm(range(chuncks_num)):
            i = str(i)
            try:
                # load labels
                with open(dir_path + "/100k_" + str(i) + "/labels.json") as file:
                    self.labels[i] = json.load(file)

                # load opt
                model_file = dir_path + "/100k_" + str(i) + "/best_model.pt"
                logger.info("Loading model from {}".format(model_file))
                opt = load_torch_models(model_file)['config']

                opt["cuda"] = False
                opt["cpu"] = True
                trainer = DataCreators.maddog_models.GCNTrainer(opt)
                trainer.load(model_file)

                self.opts[i] = opt
                self.trainers[i] = trainer

                # load vocab
                vocab_file = dir_path + "/100k_" + str(i) + "/vocab.pkl"
                vocab = Vocab(vocab_file, load=True)

                self.vocabs[i] = vocab
            except Exception as e:
                logger.exception(e)
                pass

    def predict(self, id, data, valid_labels):
        if id in self.trainers:
            trainer = self.trainers[id]
            opt = self.opts[id]
            vocab = self.vocabs[id]
            label2id = self.labels[id]
            batch = DataLoader(
                data, opt["batch_size"], opt, vocab, evaluation=True, label2id=label2id
            )

            id2label = dict([(v, k) for k, v in label2id.items()])
            label_mask = []
            for k in label2id:
                if k in valid_labels:
                    label_mask += [1]
                else:
                    label_mask += [0]
            for b in batch:
                probs, _ = trainer.predict(b, label_mask)
                break

            scores = []
            predictions = []
            for i, lm in enumerate(label_mask):
                if lm == 1:
                    predictions += [id2label[i]]
                    scores += [probs[0][i]]

            scores = softmax(scores).tolist()

            return predictions, scores
        else:
            return "", 0


class ZeroShotExtractor(prototype.acronym.zeroshot.zeroshot.ZeroShotExtractor):
    def __init__(self, dir_path, chuncks_num):
        with open(dir_path + "/diction.json") as file:
            self.diction = json.load(file)
        with open(dir_path + "/addresses.json") as file:
            self.addresses = json.load(file)
        self.model_loader = ModelLoader(dir_path + "/saved_models100", chuncks_num)
        self.dir_path = dir_path

    def predict(self, sentence, ind):
        acronym_pos = [0] * len(sentence)
        acronym_pos[ind] = 1
        data = {
            "tokens": sentence,
            "acronym_pos": acronym_pos,
        }
        data_file = self.dir_path + "/data.json"
        with open(data_file, "w") as file:
            json.dump([data] * 2, file)
        ids = self.addresses[sentence[ind]]
        predictions = []
        probablities = []
        for id in ids:
            preds, probs = self.model_loader.predict(
                id, data_file, set(self.diction[sentence[ind]])
            )
            if preds != "":
                predictions += preds
                probablities += probs
        if len(predictions) < 1:
            return "", [("", 0)]
        pairs = list(zip(predictions, probablities))
        pairs = sorted(pairs, key=lambda p: p[1], reverse=True)
        prediction = pairs[0][0]
        return prediction, pairs


class _MadDog(OutExpander):
    def __init__(self, models_path, chunck_num):
        self.models_path = models_path
        self.zeroshot_extractor = ZeroShotExtractor(models_path, chunck_num)

    def extract_expansions(self, article_text, acronyms):
        predicted_expansions = []
        for acronym in acronyms:
            sentence = self.tokenize_sentence(article_text, acronym)
            indx = last_index(sentence, acronym)
            if acronym in self.zeroshot_extractor.diction:
                try:
                    lf, scores = self.zeroshot_extractor.predict(sentence, indx)
                    predicted_expansions.append((lf, scores[0][1]))
                except Exception as e:
                    logger.exception(e)
                    predicted_expansions.append(("", 0))
            else:
                predicted_expansions.append(("", 0))

        return predicted_expansions

    def tokenize_sentence(self, article_text, acronym):
        # does not break acronym into tokens
        text_chunks = article_text.split(acronym)
        first = True
        sentence = []
        for text in text_chunks:
            sentence_chunk = deep_strip(text)

            tokens = [t.text for t in nlp(sentence_chunk) if len(t.text.strip()) > 0]

            if first:
                first = False
            else:
                sentence.append(acronym)
            sentence.extend(tokens)

        return sentence

    def process_article(self, out_expander_input: OutExpanderArticleInput):
        acronyms_list = out_expander_input.acronyms_list

        article_text = out_expander_input.article.get_raw_text()

        predicted_expansions = self.extract_expansions(article_text, acronyms_list)

        return predicted_expansions
