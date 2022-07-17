"""
Created on May 12, 2021

@author: jpereira
"""
from collections import Counter
import json
import os
import pickle
import prototype.acronym.zeroshot.model.data.loader
from prototype.acronym.zeroshot.model.utils import vocab, constant
import random

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import spacy
from tqdm import tqdm

from Logger import logging
from inputters import TrainOutDataManager
import numpy as np
from out_expanders.impl.sci_dr import _distinct_expansions
from string_constants import FILE_GLOVE_EMBEDDINGS
from text_preparation import get_expansion_without_spaces


logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

chunks_size = 10000


class DataLoader(prototype.acronym.zeroshot.model.data.loader.DataLoader):
    """
    Load data from json files, preprocess and prepare batches.
    """

    def __init__(
        self,
        filename,
        batch_size,
        opt,
        vocab,
        evaluation=False,
        label2id={},
        dev_data=False,
    ):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.dev_data = dev_data

        if len(label2id) == 0:
            with open(
                "/".join(filename.split("/")[:-1]) + "/long_forms.pkl", "rb"
            ) as file:
                long_forms = pickle.load(file)
            labels = {}
            for l in long_forms:
                if l not in labels:
                    labels[l] = len(labels)
            # self.label2id = constant.LABEL_TO_ID
            self.label2id = labels
        else:
            self.label2id = label2id

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data
        data = self.preprocess(data, vocab, opt, evaluation)
        logger.debug(filename + " : " + str(len(data)))

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v, k) for k, v in self.label2id.items()])
        if not dev_data:
            self.labels = [self.id2label[d[-1]] for d in data]
        else:
            self.labels = [d["long_form"] for d in self.raw_data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        logger.debug("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt, evaluation):
        """Preprocess the data and convert to ids."""
        processed = []
        iterator = data
        # Use tqdm only for training and dev testing, avoids tqdm for predictions
        if not evaluation or self.dev_data:
            iterator = tqdm(iterator)

        for d in iterator:
            # tokens = [t.text for t in nlp(d['paragraph'])]
            tokens = list(d["tokens"])
            # if len(tokens) > 200:
            #     continue
            tokens = prototype.acronym.zeroshot.model.data.loader.map_to_ids(
                tokens, vocab.word2id
            )
            # acronym = get_positions(d['acronym'], d['acronym'], len(tokens))
            acronym = d["acronym_pos"]
            # acronym = [1 if t == d['short_form'] else 0 for t in tokens]
            if not evaluation:
                expansion = self.label2id[d["long_form"]]
            else:
                expansion = 0
            processed += [(tokens, acronym, expansion)]
        return processed


def oversampling(x, y):
    counts = Counter(y)
    for e, c in counts.items():
        if c < 3:
            indx = y.index(e)
            y.append(e)
            x.append(x[indx])

        if c < 2:
            indx = y.index(e)
            y.append(e)
            x.append(x[indx])
    return x, y

def _make_splits(x, y):
    # split train 80% and test 10% based on expansion
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=0.1, train_size=0.9, random_state=0
    )
    try:
        return sss.split(x, y).__next__()
    except ValueError as value_error:
        message = repr(value_error)
        if "should be greater or equal to the number of classes" in message:
            try:
                unique_classes = np.unique(y)
                class_counts = len(unique_classes)
                if "The test_size" in message:
                    sss = StratifiedShuffleSplit(
                        n_splits=1, test_size=class_counts, train_size=None, random_state=0
                    )
                else:
                    sss = StratifiedShuffleSplit(
                        n_splits=1, test_size=None, train_size=class_counts, random_state=0
                    )
                return sss.split(x, y).__next__()
            except ValueError as new_value_error:
                logger.warning(repr(new_value_error))
                
        sss = ShuffleSplit(
            n_splits=1, test_size=0.1, train_size=0.9, random_state=0
        )
        return sss.split(x, y).__next__()
    
def flush_data_samples(models_path, samples, chunck_id):
    folder = models_path + "saved_models100/100k_" + str(chunck_id)
    os.makedirs(folder, exist_ok=True)
    x, y = oversampling(samples, [sample["long_form"] for sample in samples])
    x = np.asarray(x)
    y = np.asarray(y)
    # x = np.asarray(samples)
    # y = np.array([sample[2] for sample in samples])

    labels = {}
    i = 0
    for expansion in y:
        if expansion not in labels:
            labels[expansion] = i
            i += 1

    with open(folder + "/labels.json", "w") as outfile:
        json.dump(labels, outfile)

    with open(folder + "/long_forms.pkl", "wb") as outfile:
        pickle.dump(labels, outfile)

    train_index, dev_index = _make_splits(x,y)
    X_train, X_dev = x[train_index], x[dev_index]

    with open(folder + "/train.json", "w") as outfile:
        json.dump(X_train.tolist(), outfile)

    with open(folder + "/dev.json", "w") as outfile:
        json.dump(X_dev.tolist(), outfile)


def _create_data_chuncks(models_path, article_db, acronym_db):
    addresses = {}


    sorted_acronyms = set(acronym_db.keys())
    samples_count = 0
    samples = []
    acronyms_list = []
    for acronym in tqdm(sorted_acronyms, total=len(sorted_acronyms)):
        expansion_articles = acronym_db[acronym]
        for (exp, article_id) in expansion_articles:
            article_text = article_db[article_id]
            expansion_without_spaces = get_expansion_without_spaces(exp)
            text = article_text.replace(expansion_without_spaces, acronym)
            if len(text) > 1000000:
                text = text[:1000000]
            for sentence in nlp(text).sents:
                tokens = [t.text for t in nlp(sentence.text) if len(t.text.strip()) > 0]
                if acronym in tokens:
                    acronym_pos = [1 if t == acronym else 0 for t in tokens]

                    sample = {
                        "tokens": tokens,
                        "acronym_pos": acronym_pos,
                        "long_form": exp,
                    }
                    samples.append(sample)
                    samples_count += 1

                    if not acronyms_list or acronym != acronyms_list[-1]:
                        acronyms_list.append(acronym)

                    if samples_count % chunks_size == 0:
                        chunck_id = (samples_count // chunks_size) - 1
                        flush_data_samples(models_path, samples, chunck_id)
                        samples = []
                        _add_acronyms_to_adresses(acronyms_list, chunck_id, addresses)
                        acronyms_list = []
                        
                        return addresses, samples_count #TODO remove

    if samples_count % chunks_size != 0:
        chunck_id = samples_count // chunks_size
        flush_data_samples(models_path, samples, chunck_id)
        _add_acronyms_to_adresses(acronyms_list, chunck_id, addresses)

    return addresses, samples_count
def _add_acronyms_to_adresses(acronyms_list, chunck_id, addresses):
    for acronym in acronyms_list:
        addresses.setdefault(acronym, []).append(str(chunck_id))


def create_data(models_path, train_data_manager: TrainOutDataManager):
    acronym_db = train_data_manager.get_acronym_db()
    article_db = train_data_manager.get_raw_articles_db()
    addresses, samples_count = _create_data_chuncks(models_path, article_db, acronym_db)

    with open(models_path + "/addresses.json", "w") as outfile:
        json.dump(addresses, outfile)

    # create diction.json
    diction = {
        acronym: list(_distinct_expansions(expansion_articles))
        for acronym, expansion_articles in acronym_db.items()
    }
    with open(models_path + "/diction.json", "w") as outfile:
        json.dump(diction, outfile)

    return samples_count


def create_vocab(data_dir, lower=True, min_freq=0):
    # args = parse_args()

    # input files
    train_file = data_dir + "/train.json"
    dev_file = data_dir + "/dev.json"
    wv_file = FILE_GLOVE_EMBEDDINGS
    wv_dim = 300

    # output files
    # helper.ensure_dir(data_dir)
    vocab_file = data_dir + "/vocab.pkl"
    emb_file = data_dir + "/embedding.npy"

    # load files
    logger.info("loading files...")
    train_tokens = load_tokens(train_file)
    dev_tokens = load_tokens(dev_file)
    if lower:
        train_tokens, dev_tokens = [
            [t.lower() for t in tokens] for tokens in (train_tokens, dev_tokens)
        ]

    # load glove
    logger.info("loading glove...")
    glove_vocab = vocab.load_glove_vocab(wv_file, wv_dim)
    logger.info("{} words loaded from glove.".format(len(glove_vocab)))

    logger.info("building vocab...")
    v = build_vocab(train_tokens + dev_tokens, glove_vocab, min_freq)

    logger.info("calculating oov...")
    datasets = {"train": train_tokens, "dev": dev_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
        logger.info(
            "{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov * 100.0 / total)
        )

    logger.info("building embeddings...")
    embedding = vocab.build_embedding(wv_file, v, wv_dim)
    logger.info("embedding size: {} x {}".format(*embedding.shape))

    logger.info("dumping to files...")
    with open(vocab_file, "wb") as outfile:
        pickle.dump(v, outfile)
    np.save(emb_file, embedding)
    logger.info("all done.")


def load_tokens(filename):
    with open(filename) as infile:
        data = json.load(infile)
        tokens = []
        for d in data:
            ts = d["tokens"]
            tokens += list(filter(lambda t: t != "<PAD>", ts))

    logger.info(
        "{} tokens from {} examples loaded from {}.".format(
            len(tokens), len(data), filename
        )
    )
    return tokens


def build_vocab(tokens, glove_vocab, min_freq):
    """build vocab from tokens and glove words."""
    counter = Counter(t for t in tokens)
    # if min_freq > 0, use min_freq, otherwise keep all glove words
    if min_freq > 0:
        v = sorted(
            [t for t in counter if counter.get(t) >= min_freq],
            key=counter.get,
            reverse=True,
        )
    else:
        v = sorted(
            [t for t in counter if t in glove_vocab], key=counter.get, reverse=True
        )
    # add special tokens and entity mask tokens
    v = constant.VOCAB_PREFIX + v
    logger.info("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v


def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total - matched
