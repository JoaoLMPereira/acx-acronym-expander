"""
Created on May 12, 2021

@author: jpereira
"""
import argparse
from datetime import datetime
import json
import os
import prototype.acronym.zeroshot.model.model.trainer
from prototype.acronym.zeroshot.model.utils import torch_utils, scorer, helper
from prototype.acronym.zeroshot.model.utils.vocab import Vocab
import random
from shutil import copyfile
import time

import torch

from DataCreators.maddog_data import DataLoader, create_data, chunks_size, create_vocab
from Logger import logging
from inputters import TrainOutDataManager
import numpy as np
import torch.nn.functional as F


logger = logging.getLogger(__name__)

def load_torch_models(filename):
    try:
        dump = torch.load(filename)
        return dump
    except RuntimeError:
        logger.debug("Failed to load model from {}".format(filename), exc_info=True)
    logger.debug("Trying to load model with cpu device")
        
    dump = torch.load(filename, map_location=torch.device('cpu'))
    return dump

class GCNTrainer(prototype.acronym.zeroshot.model.model.trainer.GCNTrainer):
    
    def load(self, filename):
        try:
            checkpoint = load_torch_models(filename)
        except BaseException as e:
            logger.info("Cannot load model from {}".format(filename))
            raise e
        self.model.load_state_dict(checkpoint['model'])
        opt = checkpoint['config']
        opt['cuda'] = False
        opt['cpu'] = True
        self.opt = opt

    def predict_dev_no_mask(self, batch, unsort=True):
        inputs, labels = prototype.acronym.zeroshot.model.model.trainer.unpack_batch(
            batch, self.opt["cuda"]
        )
        orig_idx = batch[-1]
        # forward
        self.model.eval()
        logits = self.model(inputs)
        # label_mask = torch.Tensor([label_mask,label_mask])
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, 1)
        probs = probs.data.cpu().numpy().tolist()

        predictions = np.argmax(
            (F.softmax(logits, 1)).data.cpu().numpy(), axis=1
        ).tolist()
        if unsort:
            _, predictions, probs = [
                list(t) for t in zip(*sorted(zip(orig_idx, predictions, probs)))
            ]
        return predictions, probs, loss.item()

    def predict_dev(self, batch, get_label_mask, unsort=True):
        inputs, labels = prototype.acronym.zeroshot.model.model.trainer.unpack_batch(
            batch, self.opt["cuda"]
        )
        orig_idx = batch[-1]
        # forward
        self.model.eval()
        logits = self.model(inputs)
        # label_mask = torch.Tensor([label_mask,label_mask])
        loss = self.criterion(logits, labels)
        label_mask = get_label_mask(inputs)

        probs = F.softmax(F.softmax(logits, 1) * label_mask, 1)
        probs = probs.data.cpu().numpy().tolist()

        predictions = np.argmax(
            (F.softmax(logits, 1) * label_mask).data.cpu().numpy(), axis=1
        ).tolist()
        if unsort:
            _, predictions, probs = [
                list(t) for t in zip(*sorted(zip(orig_idx, predictions, probs)))
            ]
        return predictions, probs, loss.item()


def get_maddog_trainer_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset/0/")
    parser.add_argument("--vocab_dir", type=str, default="dataset/db/")
    parser.add_argument(
        "--emb_dim", type=int, default=300, help="Word embedding dimension."
    )
    parser.add_argument(
        "--ner_dim", type=int, default=30, help="NER embedding dimension."
    )
    parser.add_argument(
        "--pos_dim", type=int, default=30, help="POS embedding dimension."
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=200, help="RNN hidden state size."
    )
    parser.add_argument("--num_layers", type=int, default=2, help="Num of RNN layers.")
    parser.add_argument(
        "--input_dropout", type=float, default=0.5, help="Input dropout rate."
    )
    parser.add_argument(
        "--gcn_dropout", type=float, default=0.5, help="GCN layer dropout rate."
    )
    parser.add_argument(
        "--word_dropout",
        type=float,
        default=0.04,
        help="The rate at which randomly set a word to UNK.",
    )
    parser.add_argument(
        "--topn", type=int, default=1e10, help="Only finetune top N word embeddings."
    )
    parser.add_argument(
        "--lower", dest="lower", action="store_true", help="Lowercase all words."
    )
    parser.add_argument("--no-lower", dest="lower", action="store_false")
    parser.set_defaults(lower=False)

    parser.add_argument(
        "--prune_k",
        default=-1,
        type=int,
        help="Prune the dependency tree to <= K distance off the dependency path; set to -1 for no pruning.",
    )
    parser.add_argument(
        "--conv_l2", type=float, default=0, help="L2-weight decay on conv layers only."
    )
    parser.add_argument(
        "--pooling",
        choices=["max", "avg", "sum"],
        default="max",
        help="Pooling function type. Default max.",
    )
    parser.add_argument(
        "--pooling_l2", type=float, default=0, help="L2-penalty for all pooling output."
    )
    parser.add_argument(
        "--mlp_layers", type=int, default=2, help="Number of output mlp layers."
    )
    parser.add_argument(
        "--no_adj",
        dest="no_adj",
        action="store_true",
        help="Zero out adjacency matrix for ablation.",
    )

    parser.add_argument(
        "--no-rnn", dest="rnn", action="store_false", help="Do not use RNN layer."
    )
    parser.add_argument(
        "--rnn_hidden", type=int, default=200, help="RNN hidden state size."
    )
    parser.add_argument(
        "--rnn_layers", type=int, default=1, help="Number of RNN layers."
    )
    parser.add_argument(
        "--rnn_dropout", type=float, default=0.5, help="RNN dropout rate."
    )

    parser.add_argument(
        "--lr", type=float, default=1.0, help="Applies to sgd and adagrad."
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.9, help="Learning rate decay rate."
    )
    parser.add_argument(
        "--decay_epoch",
        type=int,
        default=5,
        help="Decay learning rate after this epoch.",
    )
    parser.add_argument(
        "--optim",
        choices=["sgd", "adagrad", "adam", "adamax"],
        default="sgd",
        help="Optimizer: sgd, adagrad, adam or adamax.",
    )
    parser.add_argument(
        "--num_epoch", type=int, default=100, help="Number of total training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=50, help="Training batch size."
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=5.0, help="Gradient clipping."
    )
    parser.add_argument(
        "--log_step", type=int, default=20, help="Print log every k steps."
    )
    parser.add_argument(
        "--log", type=str, default="logs.txt", help="Write training log to file."
    )
    parser.add_argument(
        "--save_epoch",
        type=int,
        default=100,
        help="Save model checkpoints every k epochs.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./saved_models",
        help="Root dir for saving models.",
    )
    parser.add_argument(
        "--id", type=int, default=0, help="Model ID under which to save models."
    )
    parser.add_argument(
        "--info", type=str, default="", help="Optional info for the experiment."
    )

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available())
    parser.add_argument("--cpu", action="store_true", help="Ignore CUDA.")

    parser.add_argument(
        "--load", dest="load", action="store_true", help="Load pretrained model."
    )
    parser.add_argument(
        "--model_file", type=str, help="Filename of the pretrained model."
    )
    return parser


""" GCC model transforms the input masks with .squeeze, when the batch size is one this has an unexcpected behaviour
To prevent this issue, whenever we have a batch with size one, we concat the batch by itself creating a batch whoose size is 2.
"""


def _check_batch(batch):
    if len(batch[4]) > 1:
        return batch

    words = torch.cat((batch[0], batch[0]))
    masks = torch.cat((batch[1], batch[1]))
    acronym = torch.cat((batch[2], batch[2]))
    exps = torch.cat((batch[3], batch[3]))
    orig_idx = batch[4] + batch[4]

    return (words, masks, acronym, exps, orig_idx)


def create_model(func_args, diction):
    args = get_maddog_trainer_parser().parse_args(func_args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(1234)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)
    init_time = time.time()

    # make opt
    opt = vars(args)
    # label2id = constant.LABEL_TO_ID
    # opt['num_class'] = len(label2id)
    # opt['num_class'] = 1199036

    # load vocab
    vocab_file = opt["vocab_dir"] + "/vocab.pkl"
    vocab = Vocab(vocab_file, load=True)
    opt["vocab_size"] = vocab.size
    emb_file = opt["vocab_dir"] + "/embedding.npy"
    emb_matrix = np.load(emb_file)
    assert emb_matrix.shape[0] == vocab.size
    assert emb_matrix.shape[1] == opt["emb_dim"]

    # load data
    logger.info(
        "Loading data from {} with batch size {}...".format(
            opt["data_dir"], opt["batch_size"]
        )
    )
    train_batch = DataLoader(
        opt["data_dir"] + "/train.json", opt["batch_size"], opt, vocab, evaluation=False
    )
    dev_batch = DataLoader(
        opt["data_dir"] + "/dev.json",
        opt["batch_size"],
        opt,
        vocab,
        evaluation=True,
        dev_data=True,
    )

    label2id = train_batch.label2id
    opt["num_class"] = len(label2id)

    model_id = opt["id"]  # if len(opt['id']) > 1 else '0' + opt['id']
    model_save_dir = opt["save_dir"] + "/100k_" + str(model_id)
    opt["model_save_dir"] = model_save_dir
    helper.ensure_dir(model_save_dir, verbose=True)

    # save config
    helper.save_config(opt, model_save_dir + "/config.json", verbose=True)

    # No need to save again
    # vocab.save(model_save_dir + '/vocab.pkl')
    file_logger = helper.FileLogger(
        model_save_dir + "/" + opt["log"],
        header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score",
    )

    # print model info
    helper.print_config(opt)

    # model
    if not opt["load"]:
        trainer = GCNTrainer(opt, emb_matrix=emb_matrix)
    else:
        # load pretrained model
        model_file = opt["model_file"]
        logger.info("Loading model from {}".format(model_file))
        model_opt = torch_utils.load_config(model_file)
        model_opt["optim"] = opt["optim"]
        trainer = GCNTrainer(model_opt)
        trainer.load(model_file)

    id2label = dict([(v, k) for k, v in label2id.items()])
    dev_score_history = []
    current_lr = opt["lr"]

    def get_label_mask(inputs):
        label_mask_list = []
        for words_ids, _, acronym_pos in zip(*inputs):
            acronym_indx = acronym_pos.argmax()
            acronym_id = words_ids[acronym_indx]
            acronym = vocab.id2word[acronym_id]
            if acronym == "<UNK>":
                valid_labels = label2id.keys()
                # logger.info(" ".join([vocab.id2word[i] for i in words_ids]))
            else:
                valid_labels = diction[acronym]
            label_mask = []
            for k in label2id:
                if k in valid_labels:
                    label_mask += [1]
                else:
                    label_mask += [0]
            label_mask_list.append(label_mask)
        tensor_label_mask = torch.Tensor(label_mask_list)
        if args.cuda:
            tensor_label_mask = tensor_label_mask.cuda()
        return tensor_label_mask

    global_step = 0
    global_start_time = time.time()
    format_str = (
        "{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}"
    )
    max_steps = len(train_batch) * opt["num_epoch"]

    # start training
    patience = 0
    for epoch in range(1, opt["num_epoch"] + 1):
        train_loss = 0
        for i, batch in enumerate(train_batch):
            try:
                batch = _check_batch(batch)
                start_time = time.time()
                global_step += 1
                loss = trainer.update(batch)
                train_loss += loss
                if global_step % opt["log_step"] == 0:
                    duration = time.time() - start_time
                    logger.info(
                        format_str.format(
                            datetime.now(),
                            global_step,
                            max_steps,
                            epoch,
                            opt["num_epoch"],
                            loss,
                            duration,
                            current_lr,
                        )
                    )
            except Exception as exp:
                logger.error(
                    "Error when training, batch id is %d, batch content: %s",
                    i,
                    str(batch),
                )
                raise exp
        # eval on dev
        logger.info("Evaluating on dev set...")
        predictions = []
        dev_loss = 0

        for i, batch in enumerate(dev_batch):
            try:
                batch = _check_batch(batch)
                preds, _, loss = trainer.predict_dev(batch, get_label_mask)
                predictions += preds
                dev_loss += loss
            except Exception as exp:
                logger.error(
                    "Error when validating on dev, batch id is %d, batch content: %s",
                    i,
                    str(batch),
                )
                raise exp
        predictions = [id2label[p] for p in predictions]
        train_loss = (
            train_loss / train_batch.num_examples * opt["batch_size"]
        )  # avg loss per batch
        dev_loss = dev_loss / dev_batch.num_examples * opt["batch_size"]

        dev_p, dev_r, dev_f1 = scorer.score_expansion(dev_batch.gold(), predictions)
        logger.info(
            "epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(
                epoch, train_loss, dev_loss, dev_f1
            )
        )
        dev_score = dev_f1
        file_logger.log(
            "{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}".format(
                epoch,
                train_loss,
                dev_loss,
                dev_score,
                max([dev_score] + dev_score_history),
            )
        )

        # save
        model_file = model_save_dir + "/checkpoint_epoch_{}.pt".format(epoch)
        trainer.save(model_file, epoch)
        if epoch == 1 or dev_score > max(dev_score_history):
            copyfile(model_file, model_save_dir + "/best_model.pt")
            logger.info("new best model saved.")
            file_logger.log(
                "new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}".format(
                    epoch, dev_p * 100, dev_r * 100, dev_score * 100
                )
            )
            patience = 0
        if epoch % opt["save_epoch"] != 0:
            os.remove(model_file)

        # lr schedule
        if (
            len(dev_score_history) > opt["decay_epoch"]
            and dev_score <= dev_score_history[-1]
            and opt["optim"] in ["sgd", "adagrad", "adadelta"]
        ):
            current_lr *= opt["lr_decay"]
            trainer.update_lr(current_lr)

        dev_score_history += [dev_score]
        logger.info("")
        patience += 1
        if patience == 10:
            break

    logger.info("Training ended with {} epochs.".format(epoch))


def _get_train_args(chunck_id, model_path):
    return f"--id {chunck_id} --seed 1234 --prune_k -1 --optim adamax --lr 0.3 --rnn_hidden 200 --num_epoch 30 --pooling max --mlp_layers 2 --pooling_l2 0.003 --data_dir {model_path}/100k_{chunck_id}/  --vocab_dir {model_path}/100k_{chunck_id}/ --save_dir {model_path}".split()


def create_models(models_path, train_data_manager: TrainOutDataManager, device):
    samples_count_file_path = models_path + "maddog_samples_count.txt"
    if not os.path.exists(samples_count_file_path):
        samples_count = create_data(models_path, train_data_manager)
        logger.info("Data created")
        with open(samples_count_file_path, "w") as f:
            f.write(str(samples_count))
    else:
        with open(samples_count_file_path, "r") as f:
            samples_count = int(f.read())

    chunck_num = samples_count // chunks_size
    if samples_count % chunks_size != 0:
        chunck_num += 1

    for chunck_id in range(chunck_num):
        if not os.path.exists(
            models_path + "saved_models100/100k_%d/vocab.pkl" % chunck_id
        ):
            create_vocab(
                models_path + "saved_models100/100k_%d" % chunck_id, lower=False
            )
            logger.info("Vocab %d created", chunck_id)

    with open(models_path + "/diction.json", "r") as infile:
        diction = json.load(infile)
    for chunck_id in range(chunck_num):
        if not os.path.exists(
            models_path + "saved_models100/100k_%d/best_model.pt" % chunck_id
        ):
            train_args = _get_train_args(chunck_id, models_path + "saved_models100/")
            if device == "cpu":
                train_args.append("--cpu")
            create_model(train_args, diction)
            logger.info("Model %d created", chunck_id)

    return chunck_num
