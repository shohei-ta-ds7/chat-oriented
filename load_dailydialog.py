# coding: utf-8

# This script is based on "https://github.com/Sanghoon94/DailyDialogue-Parser/blob/master/parser.py" by Sanghoon Kang.

import os
import sys
import argparse
import logging
from datetime import datetime

from utils.log_print import loginfo_and_print
from utils.build_vocab import build_vocab


os.makedirs("./log", exist_ok=True)
logfilename = "./log/load_dailydialog_{}.log".format(datetime.now().strftime("%Y%m%d%H%M%S"))
format = "%(message)s"
logger = logging.getLogger("logger")
handler = logging.FileHandler(filename=logfilename)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(format))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def load_data(in_dir):
    dirname = os.path.dirname(in_dir)
    if dirname.endswith("train"):
        dial_path = os.path.join(in_dir, "dialogues_train.txt")
        prefix = "train"
    elif dirname.endswith("validation"):
        dial_path = os.path.join(in_dir, "dialogues_validation.txt")
        prefix = "validation"
    elif dirname.endswith("test"):
        dial_path = os.path.join(in_dir, "dialogues_test.txt")
        prefix = "test"
    else:
        raise ValueError("Cannot find directory")

    dial_list = []
    with open(dial_path, "r") as f:
        for line_dial in f:
            dial = line_dial.split("__eou__")[:-1]
            tmp = []
            for uttr in dial:
                if uttr[0] == " ":
                    uttr = uttr[1:]
                if uttr[-1] == " ":
                    uttr = uttr[:-1]
                tmp.append(uttr)
            dial_list.append(tmp)

    return dial_list, prefix


def extend_dial(dial_list):
    extended_dial_list = []
    for d in dial_list:
        for i in range(2, len(d)+1):
            extended_dial_list += [d[:i]]
    return extended_dial_list


def save_data(out_dir, prefix, dial_list):
    out_dial_path = os.path.join(out_dir, prefix+"_text.txt")
    with open(out_dial_path, "w") as f:
        for dial in dial_list:
            f.write("<dial>\n")
            for uttr in dial:
                f.write(uttr+"\n")
            f.write("</dial>\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--in_dir",
        help="input directory",
        required=True
    )
    parser.add_argument(
        "-o", "--out_dir",
        help="output directory",
        required=True
    )
    args = parser.parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    dial_list, prefix = load_data(in_dir)
    loginfo_and_print(logger, "Load {} dialogues from {}".format(len(dial_list), in_dir))
    if prefix == "train":
        build_vocab(out_dir, dial_list)
    dial_list = extend_dial(dial_list)
    loginfo_and_print(logger, "Extend to {} dialogues".format(len(dial_list)))
    save_data(out_dir, prefix, dial_list)
