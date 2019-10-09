# coding: utf-8

import os
import sys
import argparse
import logging
from datetime import datetime
import pickle
from collections import OrderedDict

from utils.log_print import loginfo_and_print
from utils.bleu import get_bleu
from utils.dist import get_dist_1
from utils.dist import get_dist_2
from utils.pmi import get_pmi


os.makedirs("./log", exist_ok=True)
logfilename = "./log/auto_eval_{}.log".format(datetime.now().strftime("%Y%m%d%H%M%S"))
format = "%(message)s"
logger = logging.getLogger("logger")
handler = logging.FileHandler(filename=logfilename, mode="w", encoding="utf-8")
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(format))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def compare(vocab, inf_dic):
    # BLEU
    for filename, dial_list in inf_dic.items():
        loginfo_and_print(
            logger,
            "BLEU ({}): {:.2f}".format(filename, get_bleu(dial_list))
        )

    # dist-1
    for filename, dial_list in inf_dic.items():
        loginfo_and_print(
            logger,
            "dist-1 ({}): {:.2f}".format(filename, get_dist_1(dial_list))
        )

    # dist-2
    for filename, dial_list in inf_dic.items():
        loginfo_and_print(
            logger,
            "dist-2 ({}): {:.2f}".format(filename, get_dist_2(dial_list))
        )

    # PMI
    for filename, dial_list in inf_dic.items():
        loginfo_and_print(
            logger,
            "PMI ({}): {:.2f}".format(filename, get_pmi(vocab, dial_list))
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--vocab_path",
        help="vocabulary file path",
        default="./data/vocab.pkl"
    )
    parser.add_argument(
        "-i", "--inf_pkl_list",
        help="inference pickle file list",
        nargs="+",
        required=True
    )
    args = parser.parse_args()
    vocab_path = args.vocab_path
    inf_pkl_list = args.inf_pkl_list
    logger.info("Vocabulary file: {}".format(vocab_path))
    logger.info("Inference pickle file list:")
    for path in inf_pkl_list:
        logger.info(path)

    with open(vocab_path, mode="rb") as f:
        vocab = pickle.load(f)
    inf_dic = OrderedDict()
    for path in inf_pkl_list:
        with open(path, mode="rb") as f:
            inf_dic[os.path.basename(path)] = pickle.load(f)

    print("Compare model inferences...")
    compare(vocab, inf_dic)
    print("Done")
