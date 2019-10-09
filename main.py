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

from dataset import DialDataset
from train import run_epochs
from test import test
from test import chat

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
        default="./pkl/ncm"
    )
    parser.add_argument(
        "-c", "--checkpoint",
        help="checkpoint model path"
    )
    parser.add_argument(
        "-i", "--inf_pkl",
        help="pickle inf_pkl filepath of inference",
        default="./pkl/inf"
    )
    parser.add_argument(
        "--pretrained",
        help="model is pretrained (then, start training from epoch 1)",
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
        "--mode",
        help="model mode (train, inference, or chat)",
        default="train"
    )

    hparams = parser.add_argument_group("hyper parameters",
        "These arguments are ignored if a checkpoint file is given and model is not pretrained.")
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
        "--loss",
        help="loss function (ce (Cross Entropy), mmi (Maximum Mutual Information), or itf (Inverse Token Frequency))",
        default="ce"
    )
    hparams.add_argument(
        "--mmi_lambda",
        help="mmi loss lambda",
        type=float,
        default=0.4
    )
    hparams.add_argument(
        "--mmi_gamma",
        help="mmi loss gamma",
        type=int,
        default=10
    )
    hparams.add_argument(
        "--itf_lambda",
        help="itf loss lambda",
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
        "loss": args.loss,
        "mmi_lambda": args.mmi_lambda,
        "mmi_gamma": args.mmi_gamma,
        "itf_lambda": args.itf_lambda,
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
    mode = args.mode
    inf_pkl = args.inf_pkl
    if mode == "inference":
        assert checkpoint_path is not None
    elif mode == "chat":
        assert checkpoint_path is not None
        assert inf_pkl is not None
    logger.info("Data directory: {}".format(data_dir))
    logger.info("Vocabulary file: {}".format(vocab_path))
    logger.info("Model prefix: {}".format(model_pre))
    logger.info("Checkpoint path: {}".format(checkpoint_path))
    logger.info("Pretrained: {}".format(pretrained))
    logger.info("Fix embedding: {}".format(fix_embedding))
    logger.info("Model mode: {}".format(mode))
    logger.info("Inference pickle path: {}".format(inf_pkl))

    os.makedirs(os.path.dirname(model_pre), exist_ok=True)
    if inf_pkl:
        os.makedirs(os.path.dirname(inf_pkl), exist_ok=True)

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
    if mode != "train":
        hparams = update_hparams(hparams)
    for k, v in hparams.items():
        logger.info("{}: {}".format(k, v))

    if mode == "train":
        data_pre = data_dir+"/validation"
        print("Loading valid dataset...")
        valid_dataset = DialDataset(hparams, data_pre, vocab)

        data_pre = data_dir+"/train"
    else:
        data_pre = data_dir+"/test"

    print("Loading dataset...")
    dataset = DialDataset(hparams, data_pre, vocab)
    if hparams["loss"] == "itf":
        itfloss_weight = dataset.itfloss_weight
    else:
        itfloss_weight = None

    print("Building model...")
    if hparams["model_arc"] == encdec:
        model = EncDec(hparams, n_words, itfloss_weight, fix_embedding).cuda()
    elif hparams["model_arc"] == hred:
        model = HRED(hparams, n_words, itfloss_weight, fix_embedding).cuda()
    else:
        raise ValueError("Unknown model architecture!")
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    print("Model built and ready to go!")

    if mode == "train":
        print("Training model...")
        run_epochs(
            hparams, model, dataset, valid_dataset, model_pre,
            valid_every, save_every, checkpoint, pretrained
        )
    elif mode == "inference":
        print("Inference utterances...")
        test(hparams, model, dataset, inf_pkl)
    elif mode == "chat":
        print("Chatting with bot...")
        chat(hparams, model, vocab)
    else:
        raise ValueError("Unknown mode!")
    print("Done")
