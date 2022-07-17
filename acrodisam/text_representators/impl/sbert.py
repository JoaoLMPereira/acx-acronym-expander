"""

"""
import os
import pickle
from typing import Optional

from pydantic import validate_arguments, PositiveInt
from typing_extensions import Literal
import numpy as np
from Logger import logging
from run_config import RunConfig
from inputters import TrainOutDataManager, InputArticle

from .._base import TextRepresentatorAcronymIndependent, TextRepresentatorFactory
from helper import ExecutionTimeObserver, TrainInstance, \
    getDatasetGeneratedFilesPath

from tqdm.autonotebook import trange

from sentence_transformers import SentenceTransformer
import spacy

# replace typing_extensions by typing in python 3.7+
logger = logging.getLogger(__name__)

nlp = None


def get_sentences(text):
    if not nlp:
        nlp = spacy.load("en_core_web_sm")
    
    article_sentences = []
    while len(text) > 1000000:
        article_sentences.extend(nlp(text[:1000000]).sents)
        text = text[1000000:]
    article_sentences.extend(nlp(text).sents)
    # next filter avoids IndexError: [E201] Span index out of range when running the tokinzer of the sentembedding
    return [sent for sent in article_sentences if len(sent) > 1] 


def get_splits_max_model(text, sbertmodel: SentenceTransformer):
    # strip
    texts = [text]
    to_tokenize = [texts]
    to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]
    all_tokens = sbertmodel.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=sbertmodel.max_seq_length, return_overflowing_tokens=True)["input_ids"]
    if all_tokens.shape[0] < 2:
        return [text]
    
    text_splits = []
    for i in range(0, all_tokens.shape[0]):
        text = sbertmodel.tokenizer.decode(all_tokens[i], True, False)
        text_splits.append(text)
        
    return text_splits


def get_mean_sent_embedding(sentences, bertmodel):
        sent_embdings = bertmodel.encode(sentences, show_progress_bar=False)
        doc_emb = np.mean(sent_embdings, axis=0)
        return doc_emb


class FactorySBert(
    TextRepresentatorFactory
):  # pylint: disable=too-few-public-methods
    """
    Text representator factory to create SBERT embeddings
    """

    @validate_arguments
    def __init__(# pylint: disable=too-many-arguments
        self,
        model_name_or_path: Optional[str]='bert-base-uncased',
        split_policy: Literal[None, "simple", "sentence"]=None,
        device: Optional[str]=None,
        run_config: Optional[RunConfig]=RunConfig(),
    ):  
        """

        :param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from Huggingface models repository with that name.
        :param split_policy: If None the input text is truncated if exceeds the model limit, if 'simple' the excess is kept as a new sample and embeddings are merged with mean, if 'sentence' then spacy sentence segmentation is applied and each sentence is independently passed to the model.
        :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if a GPU can be used.
        :param run_config: general run configurations
        """

        self.model_name_or_path = model_name_or_path
        self.split_policy = split_policy
        self.device = device
        self.run_config = run_config


    def get_embeddings_interator(self, bertmodel, sentences):
        batch_size=32
        for start_index in trange(0, len(sentences), batch_size, desc="Batches"):
            sentences_batch = sentences[start_index:start_index+batch_size]
            train_sents_emb = bertmodel.encode(sentences_batch, convert_to_numpy=True, show_progress_bar=False)
            for emb in train_sents_emb:
                yield emb

    def get_embeddings_train_articles(self, bertmodel, train_data_manager, split_func, execution_time_observer):
        execution_time_observer.start()
        if not split_func:
            article_ids = []
            texts = []
            for article_id, text in train_data_manager.get_raw_articles_db().items():
                article_ids.append(article_id)
                texts.append(text)
            
            train_docs_emb = bertmodel.encode(texts)
            
            article_emb_dict = dict(zip(article_ids, train_docs_emb))
            execution_time_observer.stop()
    
        else:
            sentences = []
            article_ids = []
            for article_id, text in train_data_manager.get_raw_articles_db().items():
                article_sentences = split_func(text)
                sentences.extend(article_sentences)
                article_ids.extend([article_id] * len(article_sentences))
        
            #train_sents_emb = bertmodel.encode(sentences, convert_to_numpy=False)
            train_sents_emb = self.get_embeddings_interator(bertmodel, sentences)
            
            article_emb_dict = {}
            prev_article_id = -1#article_ids[0]
            #acc_embedding = train_sents_emb[0]
            acc_embedding = None
            acc_emb_num = 0
            for article_id, embedding in zip(article_ids, train_sents_emb):
                if article_id != prev_article_id:
                    if prev_article_id != -1:
                        article_emb_dict[prev_article_id] = acc_embedding / acc_emb_num
                    
                    # reset vars
                    acc_emb_num = 1
                    prev_article_id = article_id
                    acc_embedding = embedding
                else:
                    acc_emb_num += 1
                    acc_embedding += embedding
                    #all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
            article_emb_dict[prev_article_id] = acc_embedding / acc_emb_num
        execution_time_observer.stop()
        return  article_emb_dict              

    def get_text_representator(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: Optional[ExecutionTimeObserver]=None,
    ):

        bertmodel = SentenceTransformer(self.model_name_or_path)

        if not self.split_policy:
            split_func = None
            emb_func = lambda text: bertmodel.encode(text, show_progress_bar=False)
        else:
            if self.split_policy.casefold() == "simple":
                split_func = lambda text: get_splits_max_model(text, bertmodel)
            else:
                split_func = get_sentences
            emb_func = lambda text: get_mean_sent_embedding(split_func(text), bertmodel)

        if self.run_config.save_and_load:
            dataset_name = self.run_config.name
            generated_files_folder = getDatasetGeneratedFilesPath(dataset_name)
            if dataset_name.endswith("_confidences"):
                    dataset_name = dataset_name.replace("_confidences", "")
            fold = train_data_manager.get_fold()
            sbert_path = generated_files_folder + "sbert_embeddings_" + str(fold) + ("_" + self.split_policy) if self.split_policy else ""
            sbert_path += ".pickle"
            if os.path.isfile(sbert_path):
                with open(sbert_path, "rb") as f:
                    article_emb_dict = pickle.load(f)
                
                return _RepresentatorSBERT(emb_func, article_emb_dict)
        
        article_emb_dict = self.get_embeddings_train_articles(bertmodel, train_data_manager, split_func, execution_time_observer)
        
        if self.run_config.save_and_load:
            with open(sbert_path, "wb") as f:
                pickle.dump(article_emb_dict, f, pickle.HIGHEST_PROTOCOL)
        
        return _RepresentatorSBERT(emb_func, article_emb_dict)

            
class _RepresentatorSBERT(TextRepresentatorAcronymIndependent):

    def __init__(self, encode_func, train_articles_embeddings):
        super().__init__()
        self.train_articles_embeddings = train_articles_embeddings
        self.encode_func = encode_func

    def _transform_input_text(self, article: InputArticle):
        return self.encode_func(article.get_raw_text())

    def _transform_train_instance(self, train_instance: TrainInstance):
        return self.train_articles_embeddings[train_instance.article_id]

