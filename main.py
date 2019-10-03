# coding: utf-8

import os
import sys
import argparse
import logging
from datetime import datetime
import pickle
from math import sqrt

import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from torch.utils.data import DataLoader

from utils.log_print import loginfo_and_print

from train import run_epochs
from test import test
from dataset import DialDataset

from model.encdec import EncDec
from model.hred import HRED


os.makedirs("./log", exist_ok=True)
logfilename = "./log/main_{}.log".format(datetime.now().strftime("%Y%m%d%H%M%S"))
format = "%(message)s"
logger = logging.getLogger("logger")
handler = logging.FileHandler(filename=logfilename)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(format))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

encdec = "encdec"
hred = "hred"


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir",
        help="data directory",
        default="./data/"
    )
    parser.add_argument(
        "-m", "--model_pre",
        help="model prefix",
        default="./model_log/ncm"
    )
    parser.add_argument(
        "-c", "--checkpoint",
        help="checkpoint model path"
    )
    parser.add_argument(
        "-o", "--output",
        help="pickle output filepath of inference",
    )
    parser.add_argument(
        "--pretrained",
        help="the model is pretrained (then, start training from epoch 1)",
        action="store_true"
    )
    parser.add_argument(
        "--valid_every",
        help="validation interval (iteration)",
        type=int,
        default=100
    )
    parser.add_argument(
        "--save_every",
        help="model saving interval (epoch)",
        type=int,
        default=1
    )
    parser.add_argument(
        "--fix_embedding",
        help="fix embedding or not",
        action="store_true"
    )
    parser.add_argument(
        "--inference",
        help="inference mode",
        action="store_true"
    )

    hparams = parser.add_argument_group("hyper parameters",
        "These arguments are ignored if a checkpoint file is given and the model is not pretrained.")
    hparams.add_argument(
        "--model_arc",
        help="model architecture (encdec or hred)",
        default="hred"
    )
    hparams.add_argument(
        "--hidden_size",
        help="hidden size",
        type=int,
        default=256
    )
    hparams.add_argument(
        "--num_layers",
        help="number of layers",
        type=int,
        default=2
    )
    hparams.add_argument(
        "--batch_size",
        help="batch size",
        type=int,
        default=100
    )
    hparams.add_argument(
        "--max_epoch",
        help="maximum epoch number",
        type=int,
        default=40
    )
    hparams.add_argument(
        "--max_gradient",
        help="maximum gradient",
        type=float,
        default=50.0
    )
    hparams.add_argument(
        "--learning_rate",
        help="learning rate",
        type=float,
        default=1e-4
    )
    hparams.add_argument(
        "--decoder_learning_ratio",
        help="decoder learning ratio",
        type=float,
        default=5.0
    )
    hparams.add_argument(
        "--decay_step",
        help="decay step",
        type=int,
        default=6000
    )
    hparams.add_argument(
        "--lr_decay",
        help="learning rate decay",
        type=float,
        default=1/sqrt(3)
    )
    hparams.add_argument(
        "--dropout",
        help="dropout probability",
        type=float,
        default=0.1
    )
    hparams.add_argument(
        "--teacher_forcing_ratio",
        help="teacher_forcing_ratio",
        type=float,
        default=1.0
    )
    hparams.add_argument(
        "--tfd_lambda",
        help="tfd loss lambda",
        type=float,
        default=0.4
    )
    hparams.add_argument(
        "--l2_pooling",
        help="use l2_pooling or not",
        action="store_true"
    )
    hparams.add_argument(
        "--beam_width",
        help="beam width",
        type=int,
        default=20
    )
    hparams.add_argument(
        "--suppress_lambda",
        help="supress lambda",
        type=float,
        default=1.0
    )
    hparams.add_argument(
        "--len_alpha",
        help="length alpha",
        type=float,
        default=0.6
    )

    return parser


def parse_hparams(args):
    return {
        "PAD_id": 0,
        "SOS_id": 1,
        "EOS_id": 2,
        "UNK_id": 3,
        "MAX_DIAL_LEN": 4,
        "MAX_UTTR_LEN": 30,

        "model_arc": args.model_arc,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "max_epoch": args.max_epoch,
        "max_gradient": args.max_gradient,
        "learning_rate": args.learning_rate,
        "decoder_learning_ratio": args.decoder_learning_ratio,
        "decay_step": args.decay_step,
        "lr_decay": args.lr_decay,
        "dropout": args.dropout,
        "teacher_forcing_ratio": args.teacher_forcing_ratio,
        "tfd_lambda": args.tfd_lambda,
        "l2_pooling": args.l2_pooling,
    }


def update_hparams(hparams):
    hparams.update({
        "batch_size": 1,
        "beam_width": args.beam_width,
        "suppress_lambda": args.suppress_lambda,
        "len_alpha": args.len_alpha
    })
    return hparams


if __name__ == "__main__":
    args = get_argparse().parse_args()
    data_dir = args.data_dir
    vocab_path = os.path.join(data_dir, "vocab.pkl")
    model_pre = args.model_pre
    checkpoint_path = args.checkpoint
    pretrained = args.pretrained
    if pretrained:
        assert checkpoint_path is not None
    valid_every = args.valid_every
    save_every = args.save_every
    fix_embedding = args.fix_embedding
    train = not args.inference
    pickle_path = args.output
    if not train:
        assert checkpoint_path is not None
        assert pickle_path is not None
    logger.info("Data directory: {}".format(data_dir))
    logger.info("Vocabulary file: {}".format(vocab_path))
    logger.info("Model prefix: {}".format(model_pre))
    logger.info("Checkpoint path: {}".format(checkpoint_path))
    logger.info("Pretrained: {}".format(pretrained))
    logger.info("Fix embedding: {}".format(fix_embedding))
    logger.info("Training mode: {}".format(train))
    logger.info("Pickle output: {}".format(pickle_path))

    os.makedirs(os.path.dirname(model_pre), exist_ok=True)
    if pickle_path:
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    n_words = len(vocab["wtoi"])

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = None

    if checkpoint:
        hparams = checkpoint["hparams"]
    if checkpoint is None or pretrained:
        hparams = parse_hparams(args)
    if not train:
        hparams = update_hparams(hparams)
    for k, v in hparams.items():
        logger.info("{}: {}".format(k, v))

    if train:
        data_pre = data_dir+"/train"
        print("Loading the train dataset...")
        train_dataset = DialDataset(hparams, data_pre, vocab)
        tfdloss_weight = train_dataset.tfdloss_weight

        data_pre = data_dir+"/validation"
        print("Loading the valid dataset...")
        valid_dataset = DialDataset(hparams, data_pre, vocab, tfdloss_weight=tfdloss_weight)
    else:
        data_pre = data_dir+"/test"
        print("Loading the test dataset...")
        test_dataset = DialDataset(hparams, data_pre, vocab)
        tfdloss_weight = test_dataset.tfdloss_weight

    print("Building the model...")
    if hparams["model_arc"] == encdec:
        model = EncDec(hparams, n_words, tfdloss_weight, fix_embedding).cuda()
    elif hparams["model_arc"] == hred:
        model = HRED(hparams, n_words, tfdloss_weight, fix_embedding).cuda()
    else:
        raise ValueError("Unknown model architecture!")
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    print("Model built and ready to go!")

    if train:
        print("Training the model...")
        run_epochs(
            hparams, model, train_dataset, valid_dataset, model_pre,
            valid_every, save_every, checkpoint, pretrained
        )
        print("Done")
    else:
        print("Inference utterances...")
        test(hparams, model, test_dataset, pickle_path)
        print("Done")
