"""
Download the repro at https://github.com/studio-ousia/luke and add it to the PythonPath
Dowload the pre trained model luke_large_ed.tar.gz from https://drive.google.com/file/d/1BTf9XM83tWrq9VOXqj9fXlGm2mP5DNRF/view?usp=sharing
 and place it in data/PreTrainedModels/LUKE folder
"""

from luke.utils.model_utils import ModelArchive
from transformers import RobertaTokenizer
from examples.entity_disambiguation.main import EntityDisambiguationTrainer
from examples.entity_disambiguation.utils import InputFeatures

import re
import math

import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import WEIGHTS_NAME

from luke.utils.entity_vocab import MASK_TOKEN, PAD_TOKEN

from examples.entity_disambiguation.model import LukeForEntityDisambiguation

import os
from typing import Optional
import functools

import torch

from Logger import logging
from helper import getDatasetGeneratedFilesPath, ExecutionTimeObserver
from inputters import TrainOutDataManager
import numpy as np
from out_expanders._base import OutExpanderArticleInput
from run_config import RunConfig
from text_preparation import get_expansion_without_spaces

from .._base import OutExpanderFactory, OutExpander
from pydantic import validate_arguments

from string_constants import FILE_LUKE_PRETRAINED_MODEL

logger = logging.getLogger(__name__)


def _distinct_expansions(exp_articles_list):
    return {item[0] for item in exp_articles_list}


def get_expansions(train_data_manager: TrainOutDataManager):
    acronym_db = train_data_manager.get_acronym_db()
    return frozenset(expansion for expansions in acronym_db.values() for expansion in _distinct_expansions(expansions))

    
def load_model(entity_titles, args):
    # entity_titles = frozenset(entity_titles)

    entity_vocab = {PAD_TOKEN: 0, MASK_TOKEN: 1}
    for n, title in enumerate(sorted(entity_titles), 2):
        entity_vocab[title] = n

    model_config = args.model_config
    model_config.entity_vocab_size = len(entity_vocab)

    model_weights = args.model_weights
    orig_entity_vocab = args.entity_vocab
    orig_entity_emb = model_weights["entity_embeddings.entity_embeddings.weight"]
    if orig_entity_emb.size(0) != len(entity_vocab):  # detect whether the model is fine-tuned
        entity_emb = orig_entity_emb.new_zeros((len(entity_titles) + 2, model_config.hidden_size))
        orig_entity_bias = model_weights["entity_predictions.bias"]
        entity_bias = orig_entity_bias.new_zeros(len(entity_titles) + 2)
        for title, index in entity_vocab.items():
            if title in orig_entity_vocab:
                orig_index = orig_entity_vocab[title]
                entity_emb[index] = orig_entity_emb[orig_index]
                entity_bias[index] = orig_entity_bias[orig_index]
        model_weights["entity_embeddings.entity_embeddings.weight"] = entity_emb
        model_weights["entity_embeddings.mask_embedding"] = entity_emb[1].view(1, -1)
        model_weights["entity_predictions.decoder.weight"] = entity_emb
        model_weights["entity_predictions.bias"] = entity_bias
        del orig_entity_bias, entity_emb, entity_bias
    del orig_entity_emb

    model = LukeForEntityDisambiguation(model_config)
    model.load_state_dict(model_weights, strict=False)
    model.to(args.device)
    
    return model, entity_vocab


class _mention():
    title = ""
    candidates = []

    def __init__(self, title, candidates):
        self.title = title
        self.candidates = candidates


def convert_document_to_features(
    article_id,
    text,
    mentions,
    tokenizer,
    entity_vocab,
    mode,
    document_split_mode,
    max_seq_length,
    max_candidate_length,
    max_mention_length,
):
    max_num_tokens = max_seq_length - 2

    def generate_feature_dict(tokens, mentions, doc_start, doc_end):
        all_tokens = [tokenizer.cls_token] + tokens[doc_start:doc_end] + [tokenizer.sep_token]
        word_ids = np.array(tokenizer.convert_tokens_to_ids(all_tokens), dtype=np.int)
        word_attention_mask = np.ones(len(all_tokens), dtype=np.int)
        word_segment_ids = np.zeros(len(all_tokens), dtype=np.int)

        target_mention_data = []
        for start, end, mention in mentions:
            if start >= doc_start and end <= doc_end:
                candidates = [c for c in mention.candidates[:max_candidate_length]]  # TODO os candidates aqui, sao as expansoes possibeis para acronimos, 
                if mode == "train" and mention.title not in candidates:
                    continue
                target_mention_data.append((start - doc_start, end - doc_start, mention, candidates))

        entity_ids = np.empty(len(target_mention_data), dtype=np.int)
        entity_attention_mask = np.ones(len(target_mention_data), dtype=np.int)
        entity_segment_ids = np.zeros(len(target_mention_data), dtype=np.int)
        entity_position_ids = np.full((len(target_mention_data), max_mention_length), -1, dtype=np.int)
        entity_candidate_ids = np.zeros((len(target_mention_data), max_candidate_length), dtype=np.int)

        for index, (start, end, mention, candidates) in enumerate(target_mention_data):
            if mode == "train":
                entity_ids[index] = entity_vocab[mention.title]
            else:
                entity_ids[index] = entity_vocab[candidates[0]]
            entity_position_ids[index][: end - start] = range(start + 1, end + 1)  # +1 for [CLS]
            entity_candidate_ids[index,: len(candidates)] = [entity_vocab[cand] for cand in candidates]

        output_mentions = [mention for _, _, mention, _ in target_mention_data]

        return (
            output_mentions,
            dict(
                word_ids=word_ids,
                word_segment_ids=word_segment_ids,
                word_attention_mask=word_attention_mask,
                entity_ids=entity_ids,
                entity_position_ids=entity_position_ids,
                entity_segment_ids=entity_segment_ids,
                entity_attention_mask=entity_attention_mask,
                entity_candidate_ids=entity_candidate_ids,
            ),
        )

    ret = []
    tokens = []
    mention_data = []
    cur = 0
    for (start, concept, mention) in mentions: 
        tokens += tokenizer.tokenize(text[cur: start])
        mention_tokens = tokenizer.tokenize(mention.title)
        mention_data.append((len(tokens), len(tokens) + len(mention_tokens), mention))
        tokens += mention_tokens
        cur = start + len(concept)
    tokens += tokenizer.tokenize(text[cur:])

    if len(tokens) > max_num_tokens:
        if document_split_mode == "simple":
            in_mention_flag = [False] * len(tokens)
            for n, obj in enumerate(mention_data):
                in_mention_flag[obj[0]: obj[1]] = [n] * (obj[1] - obj[0])

            num_splits = math.ceil(len(tokens) / max_num_tokens)
            tokens_per_batch = math.ceil(len(tokens) / num_splits)
            doc_start = 0
            while True:
                doc_end = min(len(tokens), doc_start + tokens_per_batch)
                if mode != "train":
                    while True:
                        if (
                            doc_end == len(tokens)
                            or not in_mention_flag[doc_end - 1]
                            or (in_mention_flag[doc_end - 1] != in_mention_flag[doc_end])
                        ):
                            break
                        doc_end -= 1
                output_mentions, feature_dict = generate_feature_dict(tokens, mention_data, doc_start, doc_end)
                if output_mentions:
                    ret.append(
                        InputFeatures(
                            document=article_id,
                            mentions=output_mentions,
                            target_mention_indices=range(len(output_mentions)),
                            **feature_dict
                        )
                    )
                if doc_end == len(tokens):
                    break
                doc_start = doc_end

        else:
            for mention_index, (start, end, mention) in enumerate(mention_data):
                left_token_length = start
                right_token_length = len(tokens) - end
                mention_length = end - start
                half_context_size = int((max_num_tokens - mention_length) / 2)
                if left_token_length < right_token_length:
                    left_cxt_length = min(left_token_length, half_context_size)
                    right_cxt_length = min(right_token_length, max_num_tokens - left_cxt_length - mention_length)
                else:
                    right_cxt_length = min(right_token_length, half_context_size)
                    left_cxt_length = min(left_token_length, max_num_tokens - right_cxt_length - mention_length)
                input_mentions = (
                    [mention_data[mention_index]] + mention_data[:mention_index] + mention_data[mention_index + 1:]
                )
                output_mentions, feature_dict = generate_feature_dict(
                    tokens, input_mentions, start - left_cxt_length, end + right_cxt_length
                )
                ret.append(
                    InputFeatures(
                        document=article_id, mentions=output_mentions, target_mention_indices=[0], **feature_dict
                    )
                )
    else:
        output_mentions, feature_dict = generate_feature_dict(tokens, mention_data, 0, len(tokens))
        ret.append(
            InputFeatures(
                document=article_id,
                mentions=output_mentions,
                target_mention_indices=range(len(output_mentions)),
                **feature_dict
            )
        )

    return ret


def convert_train_documents_to_features(
    articles_db,
    articles_acronym_db,
    acronym_db,
    tokenizer,
    entity_vocab,
    mode,
    document_split_mode,
    max_seq_length,
    max_candidate_length,
    max_mention_length,
):
    ret = []
    for article_id, text in articles_db.items():

        mentions = []
        for item in articles_acronym_db.get(article_id, []).items():
            acronym = item[0]
            expansion = item[1]
            exp_without_spaces = get_expansion_without_spaces(expansion)
            candidates = list({item[0] for item in acronym_db.get(acronym, [])})  # TODO sort? in case of prunning, maybe most freq? Should keep priors?
            if len(candidates) < 2:
                continue
            res = [(i.start(), exp_without_spaces, _mention(expansion, candidates)) for i in re.finditer(exp_without_spaces, text)]
            mentions.extend(res)
        
        if len(mentions) < 1:
            continue
        
        mentions.sort(key=lambda k: k[0])
        ret.extend(convert_document_to_features(
            article_id,
            text,
            mentions,
            tokenizer,
            entity_vocab,
            mode,
            document_split_mode,
            max_seq_length,
            max_candidate_length,
            max_mention_length))
    return ret

def convert_test_document_to_features(
    article_id,
    article_text,
    acronyms_list,
    distinct_expansions_list,
    tokenizer,
    entity_vocab,
    mode,
    document_split_mode,
    max_seq_length,
    max_candidate_length,
    max_mention_length,
):

    mentions = []
    for acronym, distinct_expansions in zip(
            acronyms_list, distinct_expansions_list
        ):
        candidates = list(distinct_expansions)

        res = [(i.start(), acronym, _mention(acronym, candidates)) for i in re.finditer(acronym, article_text)]
        mentions.extend(res)
    
    mentions.sort(key=lambda k: k[0])
    return convert_document_to_features(
        article_id,
        article_text,
        mentions,
        tokenizer,
        entity_vocab,
        mode,
        document_split_mode,
        max_seq_length,
        max_candidate_length,
        max_mention_length)


def train_model(model, entity_vocab, args, collate_fn, articles_db, articles_acronym_db, acronym_db):

    train_data = convert_train_documents_to_features(
        articles_db,
        articles_acronym_db,
        acronym_db,
        args.tokenizer,
        entity_vocab,
        "train",
        "simple",
        args.max_seq_length,
        args.max_candidate_length,
        args.max_mention_length,
    )
    train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, collate_fn=collate_fn, shuffle=True)

    logger.info("Fix entity embeddings during training: %s", args.fix_entity_emb)
    if args.fix_entity_emb:
        model.entity_embeddings.entity_embeddings.weight.requires_grad = False
    logger.info("Fix entity bias during training: %s", args.fix_entity_bias)
    if args.fix_entity_bias:
        model.entity_predictions.bias.requires_grad = False

    num_train_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    trainer = EntityDisambiguationTrainer(args, model, train_dataloader, num_train_steps)
    trainer.train()

    if args.output_dir:
        logger.info("Saving model to %s", args.output_dir)
        torch.save(model.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    
    return model

def evaluate(args, eval_dataloader, model, reverse_entity_vocab):
    acro_exp_dict = dict()

    for item in eval_dataloader:
        inputs = {
            k: v.to(args.device) for k, v in item.items() if k not in ("document", "mentions", "target_mention_indices")
        }
        entity_ids = inputs.pop("entity_ids")
        entity_attention_mask = inputs.pop("entity_attention_mask")
        input_entity_ids = entity_ids.new_full(entity_ids.size(), 1)  # [MASK]
        entity_length = entity_ids.size(1)
        with torch.no_grad():
            if args.use_context_entities:
                result = torch.zeros(entity_length, dtype=torch.long)
                prediction_order = torch.zeros(entity_length, dtype=torch.long)
                for n in range(entity_length):
                    logits = model(entity_ids=input_entity_ids, entity_attention_mask=entity_attention_mask, **inputs)[
                        0
                    ]
                    probs = F.softmax(logits, dim=2) * (input_entity_ids == 1).unsqueeze(-1).type_as(logits)
                    max_probs, max_indices = torch.max(probs.squeeze(0), dim=1)
                    if args.context_entity_selection_order == "highest_prob":
                        target_index = torch.argmax(max_probs, dim=0)
                    elif args.context_entity_selection_order == "random":
                        target_index = random.choice((input_entity_ids == 1).squeeze(0).nonzero().view(-1).tolist())
                    elif args.context_entity_selection_order == "natural":
                        target_index = (input_entity_ids == 1).squeeze(0).nonzero().view(-1)[0]
                    input_entity_ids[0, target_index] = max_indices[target_index]
                    result[target_index] = max_indices[target_index]
                    prediction_order[target_index] = n
            else:
                logits = model(entity_ids=input_entity_ids, entity_attention_mask=entity_attention_mask, **inputs)[0]
                result = torch.argmax(logits, dim=2).squeeze(0)
    
        for index in item["target_mention_indices"][0]:
            prediction = result[index].item()
            expansion = reverse_entity_vocab[prediction]
            acronym = item["mentions"][0][index].title
            acro_exp_dict[acronym] = expansion
            
    return acro_exp_dict


# Turns a dictionary into a class
class Dict2Class(object):

    def __init__(self, my_dict):

        for key in my_dict:
            setattr(self, key, my_dict[key])

          
class LukeFactory(OutExpanderFactory):  # pylint: disable=too-few-public-methods
    """
    Out expander factory to predict the expansion for an article based on Luke
    """

    @validate_arguments
    def __init__(
        self,
        num_gpus: int=1,
        do_train: bool=False,
        batch_size: int=16,
        gradient_accumulation_steps: int = 8,
        epochs: int=2,
        fix_entities: bool=True,
        run_config: Optional[RunConfig]=RunConfig(),
    ):
        self.run_name = run_config.name
        args_dict = {
            "num_gpus": num_gpus,
            "do_train": do_train,
            "model_file": FILE_LUKE_PRETRAINED_MODEL,
            "num_train_epochs": epochs,
            "train_batch_size": batch_size,
            "output_dir": getDatasetGeneratedFilesPath(run_config.name),
            "max_seq_length": 512,
            "max_candidate_length": 30,
            "masked_entity_prob": 0.9,
            "use_context_entities": True,
            "context_entity_selection_order": "highest_prob",  # type=click.Choice(["natural", "random", "highest_prob"])
            "document_split_mode": "per_mention",  # type=click.Choice(["simple", "per_mention"]))
            "fix_entity_emb": fix_entities, # default True
            "fix_entity_bias": fix_entities, # default True
            "seed": 1,
            
            "local_rank":-1,
            "lr_schedule": "warmup_linear",
            "weight_decay": 0.01,
            "adam_b1": 0.9,
            "adam_eps": 1e-06,
            "adam_correct_bias": True,
            "fp16": False,  # TODO default True install apex
            "fp16_opt_level": "O2",
            "fp16_min_loss_scale": 1,
            "fp16_max_loss_scale": 4,
            "save_steps": 0,
            
            "learning_rate":2e-5,
            "adam_b2":0.999,
            "max_grad_norm": 1.0,
            "warmup_proportion": 0.1,
            "gradient_accumulation_steps":gradient_accumulation_steps,  # TODO default 8
        }
        
        if num_gpus == 0:
            args_dict["device"] = torch.device("cpu")
        else:
            args_dict["device"] = torch.device("cuda")
        """
        elif args.local_rank == -1:
            args_dict["device"] = torch.device("cuda")
        else:
            torch.cuda.set_device(args.local_rank)
            args_dict["device"] = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
        """

        # if args.model_file:
        model_archive = ModelArchive.load(args_dict["model_file"])
        args_dict["entity_vocab"] = model_archive.entity_vocab
        args_dict["bert_model_name"] = model_archive.bert_model_name
        if model_archive.bert_model_name.startswith("roberta"):
            # the current example code does not support the fast tokenizer
            args_dict["tokenizer"] = RobertaTokenizer.from_pretrained(model_archive.bert_model_name)
        else:
            args_dict["tokenizer"] = model_archive.tokenizer
        args_dict["model_config"] = model_archive.config
        args_dict["max_mention_length"] = model_archive.max_mention_length
        args_dict["model_weights"] = model_archive.state_dict
        
        self.args = Dict2Class(args_dict)
         
    def collate_fn(self, batch, is_eval=False):

        def create_padded_sequence(attr_name, padding_value):
            tensors = [torch.tensor(getattr(o, attr_name), dtype=torch.long) for o in batch]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)
    
        ret = dict(
            word_ids=create_padded_sequence("word_ids", self.args.tokenizer.pad_token_id),
            word_segment_ids=create_padded_sequence("word_segment_ids", 0),
            word_attention_mask=create_padded_sequence("word_attention_mask", 0),
            entity_ids=create_padded_sequence("entity_ids", 0),
            entity_position_ids=create_padded_sequence("entity_position_ids", -1),
            entity_segment_ids=create_padded_sequence("entity_segment_ids", 0),
            entity_attention_mask=create_padded_sequence("entity_attention_mask", 0),
        )
        ret["entity_candidate_ids"] = create_padded_sequence("entity_candidate_ids", 0)
    
        if is_eval:
            # ret["document"] = [o.document for o in batch]
            ret["mentions"] = [o.mentions for o in batch]
            ret["target_mention_indices"] = [o.target_mention_indices for o in batch]
    
        return ret       
        
    def get_expander(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: ExecutionTimeObserver=None,
    ):
        if (
            train_data_manager.get_fold() is not None
            and train_data_manager.get_fold() != "TrainData"
        ):
            raise NotImplementedError(
                "This out-expander uses its own cross-validation. Not accepting fold: %s"
                % train_data_manager.get_fold()
            )
            
        expansions = get_expansions(train_data_manager)
        execution_time_observer.start()
        model, entity_vocab = load_model(expansions, self.args)
        model.eval()
        if self.args.do_train:
            articles_db = train_data_manager.get_raw_articles_db()
            articles_acronym_db = train_data_manager.get_article_acronym_db()
            acronym_db = train_data_manager.get_acronym_db()
            model = train_model(model, entity_vocab, self.args, self.collate_fn, articles_db, articles_acronym_db, acronym_db)
        execution_time_observer.stop()
        reverse_entity_vocab = {v: k for k, v in entity_vocab.items()}
        return _Luke(model, self.args, expansions, entity_vocab, reverse_entity_vocab, collate_fn=functools.partial(self.collate_fn, is_eval=True))
    
    
class _Luke(OutExpander):

    def __init__(self, model, args, expansions, entity_vocab, reverse_entity_vocab, collate_fn):
        self.model = model
        self.args = args
        self.expansions = expansions
        self.entity_vocab = entity_vocab
        self.reverse_entity_vocab = reverse_entity_vocab
        self.collate_fn = collate_fn
    
    def process_article(self, out_expander_input: OutExpanderArticleInput):

        predicted_expansions = []

        acronyms_list = out_expander_input.acronyms_list
        distinct_expansions_list = out_expander_input.distinct_expansions_list

        article_text = out_expander_input.article.get_raw_text()
        
        article_id = out_expander_input.test_article_id
        eval_data = convert_test_document_to_features(
            article_id,
            article_text,
            acronyms_list,
            distinct_expansions_list,
            self.args.tokenizer,
            self.entity_vocab,
            "eval",
            self.args.document_split_mode,
            self.args.max_seq_length,
            self.args.max_candidate_length,
            self.args.max_mention_length,
        )
        eval_dataloader = DataLoader(
            eval_data, batch_size=1, collate_fn=self.collate_fn
        )

        results = evaluate(self.args, eval_dataloader, self.model, self.reverse_entity_vocab)
        
        for acronym in acronyms_list:
            predct_exp = results[acronym]

            result = predct_exp
            confidence = 1

            predicted_expansions.append((result, confidence))
        return predicted_expansions
