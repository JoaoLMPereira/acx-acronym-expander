"""
Out expander based on the paper:
Singh, Aadarsh, and Priyanshu Kumar. 
"SciDr at SDU-2020: IDEAS-Identifying and Disambiguating Everyday Acronyms for Scientific Domain."

Original code can be found in this repository:
https://github.com/aadarshsingh191198/AAAI-21-SDU-shared-task-2-AD/

Additional changes had to be perfomed to the code:
- Generalize to other datasets and external data sources
- Code refactoring
- Since the original work was proposed to a dataset containing sentences, we split the article text
 into sentences, for training each sentence is a sample. For prediction we sum the start and end 
 index predicted output like it was being done when merging output from multiple models.
"""
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
import itertools
import os
from typing import Optional
import functools

from sklearn import model_selection
from sklearn.metrics import f1_score
import spacy
import tokenizers
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW
import transformers
import wikipedia

from Logger import logging
from helper import getDatasetGeneratedFilesPath, ExecutionTimeObserver
from inputters import TrainOutDataManager
import numpy as np
from out_expanders._base import OutExpanderArticleInput
import pandas as pd
from run_config import RunConfig
from string_constants import FOLDER_SCIBERT_UNCASED
from text_preparation import get_expansion_without_spaces


from .._base import OutExpanderFactory, OutExpander
from pydantic import validate_arguments
from multiprocessing.managers import BaseManager
from multiprocessing import Manager


nlp = spacy.load("en_core_web_sm")

logger = logging.getLogger(__name__)


def _get_trained_scibert_uncased_model_path(datset_name, fold):
    return (
        getDatasetGeneratedFilesPath(datset_name)
        + f"sci_dr_scibert_uncased_model_{fold}.bin"
    )


def _get_trained_phase1_model_path(datset_name, fold):
    return (
        getDatasetGeneratedFilesPath(datset_name)
        + f"sci_dr_scibert_uncased_phase1_model_{fold}.bin"
    )


def _get_trained_phase2_model_path(datset_name, fold):
    return (
        getDatasetGeneratedFilesPath(datset_name)
        + f"sci_dr_scibert_uncased_phase2_model_{fold}.bin"
    )


def _get_trained_model_path(datset_name, fold, stage):
    if stage == 0:
        model_path = _get_trained_scibert_uncased_model_path(datset_name, fold)
    elif stage == 1:
        model_path = _get_trained_phase1_model_path(datset_name, fold)
    elif stage == 2:
        model_path = _get_trained_phase2_model_path(datset_name, fold)

    return model_path


def extract_wiki_content(args):
    exp = args[0]
    acronym = args[1]
    try:
        summary = wikipedia.page(exp).content
    except:
        try:
            suggest_term = wikipedia.suggest(exp)
            return acronym, exp, wikipedia.page(suggest_term).content
        except:
            return None
    return acronym, exp, summary


def get_wiki_data_source(dist_exp_per_acro, dataset_name):
    wiki_data_source_path = (
        getDatasetGeneratedFilesPath(dataset_name) + "sci_dr_wiki_external_data.csv"
    )
    if os.path.exists(wiki_data_source_path):
        return pd.read_csv(wiki_data_source_path)

    # convert acronym db values to have only distinct expansions
    values_iterable = [
        {(exp, acronym) for exp in exps} for acronym, exps in dist_exp_per_acro.items()
        ]
    # flat list of sets to a single list
    expansions = itertools.chain.from_iterable(values_iterable)

    my_data = []
    with ProcessPoolExecutor(max_workers=8) as executer:
        gen = executer.map(extract_wiki_content, expansions)

        my_data = [x for x in gen if x is not None]
    df = pd.DataFrame(my_data, columns=["acronym_", "expansion", "text"])

    df.to_csv(wiki_data_source_path)
    return df


def seed_all(seed=42):
    """
    Fix seed for reproducibility
    """
    # python RNG
    import random

    random.seed(seed)

    # pytorch RNGs
    import torch

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # numpy RNG
    import numpy as np

    np.random.seed(seed)


class AverageMeter:
    """
    Computes and stores the average and current value
    Source : https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch/
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stopping utility
    Source : https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch/
    """

    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            logger.info(
                "Validation score improved ({} --> {}). Saving model!".format(
                    self.val_score, epoch_score
                )
            )
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


def sample_text(text, acronym, expansion, config, max_len):
    if expansion:
        expansion_without_spaces = get_expansion_without_spaces(expansion)
        text = text.replace(expansion_without_spaces, acronym)

    # sentences = sent_tokenize(text)
    sentences = []
    while len(text) > 1000000:
        sentences.extend( nlp(text[:1000000]).sents)
        text = text[1000000:]
    sentences.extend( nlp(text).sents)
    for sent in sentences:
        split_text = sent.text.split(acronym,1)
        if len(split_text) < 2:
            continue #  acronym not found in this sent
        
        left_text = split_text[0]
        right_text = split_text[1]

        sampled_text_left = [
            item[0]
            for item in config.TOKENIZER.pre_tokenizer.pre_tokenize_str(left_text)
        ]

        sampled_text_right = [
            item[0]
            for item in config.TOKENIZER.pre_tokenizer.pre_tokenize_str(right_text)
        ]

        if len(sampled_text_left) + len(sampled_text_right) > max_len:
            left_idx = max(0, len(sampled_text_left) - max_len // 2)
            sampled_text_left = sampled_text_left[left_idx:]
            right_idx = min(len(sampled_text_right), max_len // 2)
            sampled_text_right = sampled_text_right[:right_idx]

        sampled_text = sampled_text_left + [acronym] + sampled_text_right
        yield " ".join(sampled_text)

def _distinct_expansions(exp_articles_list):
    return {item[0] for item in exp_articles_list}


def process_data(text, acronym, expansion, tokenizer, max_len, candidate_expansions):
    text = str(text)
    if expansion is not None:
        expansion = str(expansion)
    acronym = str(acronym)

    answers = acronym + " " + " ".join(candidate_expansions)

    if expansion is not None:
        start = answers.find(expansion)
        end = start + len(expansion)

        char_mask = [0] * len(answers)
        for i in range(start, end):
            char_mask[i] = 1
    
    
    tok_answer = tokenizer.encode(answers)
    answer_ids = tok_answer.ids
    answer_offsets = tok_answer.offsets

    answer_ids = answer_ids[1:-1]
    answer_offsets = answer_offsets[1:-1]

    if expansion is not None:
        target_idx = []
        for i, (off1, off2) in enumerate(answer_offsets):
            if sum(char_mask[off1:off2]) > 0:
                target_idx.append(i)

        start = target_idx[0] + 1
        end = target_idx[-1] + 1

    text_ids = tokenizer.encode(text).ids[1:-1]
    
    # fix to prevent expansions from occuping the whole input
    if len(answer_ids) > max_len - 12:
        answer_ids = answer_ids[0:max_len-12]
        answer_offsets = answer_offsets[0:max_len-12]
        
        if expansion is not None and end > max_len - 12:
            start = 0
            end = 0


    token_ids = [101] + answer_ids + [102] + text_ids + [102]
    offsets = [(0, 0)] + answer_offsets + [(0, 0)] * (len(text_ids) + 2)
    mask = [1] * len(token_ids)
    token_type = [0] * (len(answer_ids) + 1) + [1] * (2 + len(text_ids))

    text = answers + text

    padding = max_len - len(token_ids)

    if padding >= 0:
        token_ids = token_ids + ([0] * padding)
        token_type = token_type + [1] * padding
        mask = mask + ([0] * padding)
        offsets = offsets + ([(0, 0)] * padding)
    else:
        token_ids = token_ids[0:max_len]
        token_type = token_type[0:max_len]
        mask = mask[0:max_len]
        offsets = offsets[0:max_len]

    assert len(token_ids) == max_len
    assert len(mask) == max_len
    assert len(offsets) == max_len
    assert len(token_type) == max_len

    if expansion is not None:
        return {
            "ids": token_ids,
            "mask": mask,
            "token_type": token_type,
            "offset": offsets,
            "start": start,
            "end": end,
            "text": text,
            "expansion": expansion,
            "acronym": acronym,
        }

    return {
        "ids": token_ids,
        "mask": mask,
        "token_type": token_type,
        "offset": offsets,
        "text": text,
        "acronym": acronym,
    }


class Dataset:
    def __init__(self, text, acronym, expansion, config, dist_exp_per_acro):
        self.text = text
        self.acronym = acronym
        self.expansion = expansion
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.dist_exp_per_acro = dist_exp_per_acro

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        data = process_data(
            self.text[item],
            self.acronym[item],
            self.expansion[item],
            self.tokenizer,
            self.max_len,
            self.dist_exp_per_acro[self.acronym[item]],
        )

        return {
            "ids": torch.tensor(data["ids"], dtype=torch.long),
            "mask": torch.tensor(data["mask"], dtype=torch.long),
            "token_type": torch.tensor(data["token_type"], dtype=torch.long),
            "offset": torch.tensor(data["offset"], dtype=torch.long),
            "start": torch.tensor(data["start"], dtype=torch.long),
            "end": torch.tensor(data["end"], dtype=torch.long),
            "text": data["text"],
            "expansion": data["expansion"],
            "acronym": data["acronym"],
        }


class TestDataset:
    def __init__(self, texts, acronym, config, candidate_expansions):
        self.texts = texts
        self.acronym = acronym
        self.candidate_expansions = candidate_expansions
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        data = process_data(
            self.texts[item],
            self.acronym,
            None,
            self.tokenizer,
            self.max_len,
            self.candidate_expansions,
        )

        return {
            "ids": torch.tensor(data["ids"], dtype=torch.long),
            "mask": torch.tensor(data["mask"], dtype=torch.long),
            "token_type": torch.tensor(data["token_type"], dtype=torch.long),
            "offset": torch.tensor(data["offset"], dtype=torch.long),
            "text": data["text"],
            "acronym": data["acronym"],
        }


def get_loss(start, start_logits, end, end_logits):
    loss_fn = nn.CrossEntropyLoss()
    start_loss = loss_fn(start_logits, start)
    end_loss = loss_fn(end_logits, end)
    loss = start_loss + end_loss
    return loss


class BertAD(nn.Module):
    def __init__(self, stage_2, config):
        super().__init__()
        self.stage_2 = stage_2

        self.bert = transformers.BertModel.from_pretrained(
            config.MODEL, output_hidden_states=True
        )
        self.layer = nn.Linear(768, 2)
        if self.stage_2:
            self.drop_out = nn.Dropout(0.1)

    def forward(self, ids, mask, token_type, start=None, end=None):
        output = self.bert(
            input_ids=ids, attention_mask=mask, token_type_ids=token_type
        )

        if self.stage_2:
            out = self.drop_out(output[0])
            logits = self.layer(out)
        else:
            logits = self.layer(output[0])
        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start is not None and end is not None:
            loss = get_loss(start, start_logits, end, end_logits)
        else:
            loss = 0

        return loss, start_logits, end_logits


def train_fn(data_loader, model, optimizer, device):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    logger.info("Training for %d samples.", len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        mask = d["mask"]
        token_type = d["token_type"]
        start = d["start"]
        end = d["end"]

        ids = ids.to(device, dtype=torch.long)
        token_type = token_type.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        start = start.to(device, dtype=torch.long)
        end = end.to(device, dtype=torch.long)

        model.zero_grad()
        loss, start_logits, end_logits = model(ids, mask, token_type, start, end)

        loss.backward()
        optimizer.step()
        # xm.optimizer_step(optimizer, barrier=True)

        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def evaluate_jaccard(text, acronym, offsets, idx_start, idx_end, candidates):
    filtered_output = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += text[offsets[ix][0] : offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            filtered_output += " "

    # candidates = config.DICTIONARY[acronym]
    candidate_jaccards = [
        jaccard(w.strip(), filtered_output.strip()) for w in candidates
    ]
    idx = np.argmax(candidate_jaccards)

    return candidate_jaccards[idx], candidates[idx]


def eval_fn(data_loader, model, device, dist_exp_per_acro, exp_to_id):
    model.eval()
    losses = AverageMeter()
    jac = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    pred_expansion_ = []
    true_expansion_ = []

    for bi, d in enumerate(tk0):
        ids = d["ids"]
        mask = d["mask"]
        token_type = d["token_type"]
        start = d["start"]
        end = d["end"]

        text = d["text"]
        expansion = d["expansion"]
        offset = d["offset"]
        acronym = d["acronym"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type = token_type.to(device, dtype=torch.long)
        start = start.to(device, dtype=torch.long)
        end = end.to(device, dtype=torch.long)

        with torch.no_grad():
            loss, start_logits, end_logits = model(ids, mask, token_type, start, end)

        start_prob = torch.softmax(start_logits, dim=1).detach().cpu().numpy()
        end_prob = torch.softmax(end_logits, dim=1).detach().cpu().numpy()

        jac_ = []

        for px, s in enumerate(text):
            start_idx = np.argmax(start_prob[px, :])
            end_idx = np.argmax(end_prob[px, :])

            candidates = dist_exp_per_acro[acronym[px]]

            js, exp = evaluate_jaccard(
                s,  # expansion[px],
                acronym[px],
                offset[px],
                start_idx,
                end_idx,
                candidates,
            )
            jac_.append(js)
            pred_expansion_.append(exp)
            true_expansion_.append(expansion[px])

        jac.update(np.mean(jac_), len(jac_))
        losses.update(loss.item(), ids.size(0))

        tk0.set_postfix(loss=losses.avg, jaccard=jac.avg)

    pred_expansion_ = [exp_to_id[w] for w in pred_expansion_]
    true_expansion_ = [exp_to_id[w] for w in true_expansion_]

    f1 = f1_score(true_expansion_, pred_expansion_, average="macro")

    logger.info("Average Jaccard : %f", jac.avg)
    logger.info("Macro F1 : %f", f1)

    return f1


# stage 0 for base
def run(df_train, df_val, fold, dist_exp_per_acro, config, stage=0, datset_name="", device=None):
    
    logger.info("Creating train and valid datasets")
    train_dataset = Dataset(
        text=df_train.text.values,
        acronym=df_train.acronym_.values,
        expansion=df_train.expansion.values,
        config=config,
        dist_exp_per_acro=dist_exp_per_acro,
    )

    valid_dataset = Dataset(
        text=df_val.text.values,
        acronym=df_val.acronym_.values,
        expansion=df_val.expansion.values,
        config=config,
        dist_exp_per_acro=dist_exp_per_acro,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=2
    )

    logger.info("Creating BertAD")
    model = BertAD(stage_2=stage == 2, config=config)
    #device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    
    if device != torch.device("cpu") and torch.cuda.device_count() > 1:
        logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    if stage == 2:
        logger.info("Loading phase 1 model weights")
        model.load_state_dict(
            torch.load(os.path.join(_get_trained_phase1_model_path(datset_name, fold)), map_location=self.device)
        )
        logger.info("Loaded phase 1 model weights")
    model.to(device)

    lr = 2e-5
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    es = EarlyStopping(patience=2, mode="max")

    exp_to_id = {}
    for k, v in dist_exp_per_acro.items():
        for w in v:
            exp_to_id[w] = len(exp_to_id)

    logger.info("Starting training....")
    for epoch in range(config.EPOCHS):
        train_fn(train_data_loader, model, optimizer, device)
        valid_loss = eval_fn(valid_data_loader, model, device, dist_exp_per_acro, exp_to_id)
        logger.info(
            f"Fold {fold} | Epoch :{epoch + 1} | Validation Score :{valid_loss}"
        )
        model_path = _get_trained_model_path(datset_name, fold, stage)
        es(
            valid_loss,
            model,
            model_path=os.path.join(model_path),
        )

    return model


def run_k_fold(fold_id, train, dist_exp_per_acro, config, stage, dataset_name, device):
    """
    Perform k-fold cross-validation
    """
    seed_all()

    # dividing folds
    kf = model_selection.StratifiedKFold(
        n_splits=config.KFOLD, shuffle=True, random_state=config.SEED
    )
    for fold, (train_idx, val_idx) in enumerate(
        kf.split(X=train, y=train.acronym_.values)
    ):
        train.loc[val_idx, "kfold"] = fold

    logger.info("Stage: %s", stage)
    logger.info(
        f"################################################ Fold {fold_id} #################################################"
    )
    logger.info("Splitting data into train and val")
    df_train = train[train.kfold != fold_id]
    df_val = train[train.kfold == fold_id]
    return run(df_train, df_val, fold_id, dist_exp_per_acro, config, stage, dataset_name, device)

class ModelsType(str, Enum):
    base="base"
    stages="stages"
    both="both"

class Devices(str, Enum):
    auto="auto"
    gpu="gpu"
    cpu="cpu"
    
# Turns a dictionary into a class
class Dict2Class(object):
    def __init__(self, my_dict):

        for key in my_dict:
            setattr(self, key, my_dict[key])


class SciDrFactory(OutExpanderFactory):  # pylint: disable=too-few-public-methods
    """
    Out expander factory to predict the expansion for an article based on SciDR
    """
    @validate_arguments
    def __init__(
        self,
        models_type: ModelsType = "both",
        batch_size: int = 3,
        device:  Devices = "auto",
        run_config: Optional[RunConfig] = RunConfig(),
    ):
        if ModelsType.base == models_type:
            self.base_models = True
            self.stage_models = False
        elif ModelsType.stages.name == models_type:
            self.base_models = False
            self.stage_models = True
        else:
            self.base_models = True
            self.stage_models = True

        self.run_name = run_config.name
        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        elif device.casefold() == "GPU".casefold():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            
        self.persist_data = True if run_config.persistent_articles else False
        
        self.config = Dict2Class(
            {
                "SEED": 42,
                "KFOLD": 5,
                "MAX_LEN": 192,
                "MODEL": FOLDER_SCIBERT_UNCASED,
                "TOKENIZER": tokenizers.BertWordPieceTokenizer(
                    f"{FOLDER_SCIBERT_UNCASED}/vocab.txt", lowercase=True
                ),
                "EPOCHS": 5,
                "TRAIN_BATCH_SIZE": batch_size,
                "VALID_BATCH_SIZE": batch_size,
            }
        )
        
        self.sample_text_max_len = 120

    def _generate_models(self, train, dist_exp_per_acro, stage=0):
        
        for fold in range(self.config.KFOLD):
            model_path = _get_trained_model_path(self.run_name, fold, stage)
            if not os.path.exists(model_path):
                run_k_fold(
                    fold, train, dist_exp_per_acro, self.config, stage, self.run_name, self.device
                )
            
    def _load_models(self, stage=0):
        models = []
        for fold in range(self.config.KFOLD):
            model_path = _get_trained_model_path(self.run_name, fold, stage)
            model = BertAD(stage_2=stage == 2, config=self.config)
            model.to(self.device)
    
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            models.append(model)
        return models
    
    def _process_article_for_samples(self, arg_tuple):
        acro_exp = arg_tuple[0] 
        article_text = arg_tuple[1]

        data = []
        for acronym, expansion in acro_exp.items():
            for text in sample_text(
                article_text,
                acronym,
                expansion,
                self.config,
                self.sample_text_max_len,
            ):
                data.append([acronym, expansion, text])
        return data
    
    def _get_train_data(self, train_data_manager):
        logger.info("Creating training data samples.")
        
        if self.persist_data:
            train_data_path = getDatasetGeneratedFilesPath(self.run_name) + "sci_dr_train_data_pd.pickle"
            if os.path.exists(train_data_path):
                return pd.read_pickle(train_data_path)
        
        data = []
        article_acronym_db = train_data_manager.get_article_acronym_db()
        raw_articles_db = train_data_manager.get_raw_articles_db()

        with ProcessPoolExecutor() as executer:
            results = list(tqdm(executer.map(self._process_article_for_samples, [(article_acronym_db[k], v) for k, v in raw_articles_db.items()]), total=len(raw_articles_db)))
            
            for data_article in results:
                if data_article:
                    data.extend(data_article)

        train = pd.DataFrame(data, columns=["acronym_", "expansion", "text"])
        
        if self.persist_data:
            train.to_pickle(train_data_path)
            
        return train


    def _get_distinct_expansions_per_acronym(self, train_data_manager):
        logger.info("Creating distinct expansions per acronym dict.")
        acronym_db = train_data_manager.get_acronym_db()
        return {acro:list(_distinct_expansions(expansions)) for acro, expansions in acronym_db.items()}

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

        models = []
        train_data = None
        dist_exp_per_acro = None

        execution_time_observer.start()
        if self.stage_models:
            path_last_model = _get_trained_phase2_model_path(
                self.run_name, self.config.KFOLD - 1
            )
            if not os.path.exists(path_last_model):
                # check 1st stage models
                dist_exp_per_acro = self._get_distinct_expansions_per_acronym(train_data_manager)

                path_last_model_stage1 = _get_trained_phase1_model_path(
                    self.run_name, self.config.KFOLD - 1
                )
                if not os.path.exists(path_last_model_stage1):
                    # load/create external data
                    external_data_wiki = get_wiki_data_source(dist_exp_per_acro, self.run_name)
                    external_data_sentences = []
                    logger.info("Transforming external wiki data.")
                    for row in external_data_wiki.iterrows():
                        acronym = row[1]["acronym_"]
                        expansion = row[1]["expansion"]
                        doc_text = row[1]["text"]
                        try:
                            for text in sample_text(
                                row[1]["text"],
                                acronym,
                                expansion,
                                self.config,
                                self.sample_text_max_len,
                            ):
                                external_data_sentences.append([acronym, expansion, text])
                        except AttributeError:
                            #for cases where text is invalid, we skip
                            logger.debug("Invalid text in %s", str(row))
                            continue

                    train_ext_data = pd.DataFrame(
                        external_data_sentences,
                        columns=["acronym_", "expansion", "text"],
                    )

                    self._generate_models(train_ext_data, dist_exp_per_acro, stage=1)
                    
                # create stage 2 models
                train_data = self._get_train_data(train_data_manager)
            self._generate_models(train_data, dist_exp_per_acro, stage=2)

        if self.base_models:
            path_last_model = _get_trained_scibert_uncased_model_path(
                self.run_name, self.config.KFOLD - 1
            )

            if not os.path.exists(path_last_model):
                if train_data is None:
                    train_data = self._get_train_data(train_data_manager)

                if not dist_exp_per_acro:
                    dist_exp_per_acro = self._get_distinct_expansions_per_acronym(train_data_manager)
                    
            self._generate_models(train_data, dist_exp_per_acro, stage=0)
        execution_time_observer.stop()

        models = []
        if self.base_models:
            models.extend(self._load_models(stage=0))

        if self.stage_models:
            models.extend(self._load_models(stage=2))
        return _SciDr(models, self.config, self.sample_text_max_len, self.device)


class _SciDr(OutExpander):
    def __init__(self, models, config, sample_text_max_len, device):
        self.models = models
        self.config = config
        self.sample_text_max_len = sample_text_max_len
        self.device = device

    def models_infer(self, texts, acronym, candidate_expansions):
        # final_output = []

        test_dataset = TestDataset(
            texts=texts,
            acronym=acronym,
            config=self.config,
            candidate_expansions=candidate_expansions,
        )

        data_loader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=self.config.VALID_BATCH_SIZE,
            num_workers=1,
        )

        with torch.no_grad():
            outputs_start = None
            outputs_end = None

            for bi, d in enumerate(data_loader):

                ids = d["ids"]
                mask = d["mask"]
                token_type = d["token_type"]
                text = d["text"]
                acronym = d["acronym"]
                ids = ids.to(self.device, dtype=torch.long)
                mask = mask.to(self.device, dtype=torch.long)
                token_type = token_type.to(self.device, dtype=torch.long)


                offsets = d["offset"].numpy()

                for model in self.models:
                    _, outputs_start1, outputs_end1 = model(
                        ids=ids, mask=mask, token_type=token_type
                    )
                    outputs_start = (
                        (outputs_start + outputs_start1.sum(dim=0))
                        if outputs_start is not None
                        else outputs_start1.sum(dim=0)
                    )
                    outputs_end = (
                        (outputs_end + outputs_end1.sum(dim=0))
                        if outputs_end is not None
                        else outputs_end1.sum(dim=0)
                    )

            num_predictions = len(self.models) * len(texts)
            outputs_start /= num_predictions
            outputs_end /= num_predictions

            outputs_start = torch.softmax(outputs_start, dim=0).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=0).cpu().detach().numpy()

            start_idx = np.argmax(outputs_start)
            end_idx = np.argmax(outputs_end)
            js, exp = evaluate_jaccard(
                text[0],
                acronym[0],
                offsets[0],
                start_idx,
                end_idx,
                list(candidate_expansions),
            )
            return js, exp

    def process_article(self, out_expander_input: OutExpanderArticleInput):

        predicted_expansions = []

        acronyms_list = out_expander_input.acronyms_list
        distinct_expansions_list = out_expander_input.distinct_expansions_list

        article_text = out_expander_input.article.get_raw_text()
        for acronym, distinct_expansions in zip(
            acronyms_list, distinct_expansions_list
        ):
            texts = list(
                sample_text(
                    article_text, acronym, None, self.config, self.sample_text_max_len
                )
            )

            jaccard_score, predct_exp = self.models_infer(
                texts, acronym, distinct_expansions
            )

            result = predct_exp
            confidence = jaccard_score

            predicted_expansions.append((result, confidence))
        return predicted_expansions
