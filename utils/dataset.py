# coding: utf-8

import os
import sys
import logging
import collections
from collections import Counter
import tqdm
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset

from utils.log_print import loginfo_and_print


logger = logging.getLogger("logger").getChild("dataset")

np.random.seed(0)


class DialDataset(Dataset):
    def __init__(self, hparams, data_pre, vocab):
        super().__init__()
        self.hparams = hparams
        self.vocab = vocab
        self.data, self.grouped_data = self.read_dials(data_pre)
        self.itfloss_weight = self.calc_itfloss_weight(self.data)
        self.init_epoch(1)

    def read_dials(self, prefix):
        pkl_file = prefix+".pkl"
        if os.path.exists(pkl_file):
            print("Loading from pickle dump file...")
            with open(pkl_file, mode="rb") as f:
                data_dic = pickle.load(f)
            data = data_dic["data"]
            grouped_data = data_dic["grouped_data"]
            print("Done")
            return data, grouped_data

        MAX_DIAL_LEN = self.hparams["MAX_DIAL_LEN"]
        print("Reading files...")
        data = []

        with open(prefix+"_text.txt", "r") as dial_f:
            lines = [l.strip() for l in dial_f]

        dials = []
        for l in lines:
            if l == "<dial>":
                dial = []
            elif l == "</dial>":
                dials += [dial]
            else:
                dial += [l.split(" ")]

        dials = [d[-MAX_DIAL_LEN:] for d in dials]
        [data.append({"dial": d}) for d in dials]

        loginfo_and_print(logger, "Read {} dialogues".format(len(data)))

        data = [d for d in data if self.filter(d)]
        data = [self.trim(d) for d in data]
        loginfo_and_print(logger, "Trimmed to {} dialogues".format(len(data)))

        loginfo_and_print(logger, "Converting to indexes...")
        idx_data = []
        pbar = tqdm.tqdm(data, total=len(data))
        for d in pbar:
            idx_data += [self.indexes_from_data(d)]
        data = idx_data

        # Group by dial length
        loginfo_and_print(logger, "Grouping by dialogue length...")
        grouped_data = {str(l): [] for l in range(2, MAX_DIAL_LEN+1)}
        for d in data:
            grouped_data[str(len(d["dial"]))] += [
                {"src": d["dial"][:-1], "tgt": d["dial"][-1]}
            ]
        for l in range(2, MAX_DIAL_LEN+1):
            loginfo_and_print(logger, "Group {} contains {} dialogues".format(l, len(grouped_data[str(l)])))

        grouped_data = self.sort(grouped_data)

        print("Dump to pickle file...")
        data_dic = {
            "data": data,
            "grouped_data": grouped_data
        }
        with open(pkl_file, mode="wb") as f:
            pickle.dump(data_dic, f)
        print("Done")

        return data, grouped_data

    def filter(self, data):
        # Filter short dialogue
        if len(data["dial"]) < 2:
            return False

        # Filter dialogue contains too long utterance
        MAX_UTTR_LEN = self.hparams["MAX_UTTR_LEN"]
        if len(data["dial"][-2]) > MAX_UTTR_LEN or len(data["dial"][-1]) > MAX_UTTR_LEN\
            or len(data["dial"][-2]) < 1 or len(data["dial"][-1]) < 1:
            return False

        return True

    def trim(self, data):
        # Trim too long utterance
        MAX_UTTR_LEN = self.hparams["MAX_UTTR_LEN"]
        for i in range(0, len(data["dial"])-2):
            if len(data["dial"][i]) > MAX_UTTR_LEN or len(data["dial"][i]) < 1:
                data["dial"] = data["dial"][i+1:]
        return data

    def indexes_from_data(self, data):
        return {k: self.indexes_from_uttrs(k, v) for k, v in data.items()}

    def indexes_from_uttrs(self, key, uttrs):
        if key == "dial":
            return [self.indexes_from_uttr(uttr) for uttr in uttrs]
        else:
            raise ValueError("unknown key!")

    def indexes_from_uttr(self, uttr):
        return [
            self.vocab["wtoi"].get(w.lower(), self.hparams["UNK_id"])
            for w in uttr
        ]+[self.hparams["EOS_id"]]

    def sort(self, grouped_data):
        MAX_DIAL_LEN = self.hparams["MAX_DIAL_LEN"]
        for l in range(2, MAX_DIAL_LEN+1):
            grouped_data[str(l)] = sorted(
                grouped_data[str(l)],
                key=lambda x: len(x["tgt"]),
                reverse=True
            )
        return grouped_data

    def calc_itfloss_weight(self, data):
        PAD_id = self.hparams["PAD_id"]
        SOS_id = self.hparams["SOS_id"]
        EOS_id = self.hparams["EOS_id"]
        UNK_id = self.hparams["UNK_id"]
        itf_lmd = self.hparams["itf_lambda"]
        counter = Counter([
            id for dial in data
            for uttr in dial["dial"]
            for id in uttr
        ])
        uttr_num = len([
            uttr for dial in data
            for uttr in dial["dial"]
        ])
        counter[PAD_id] = uttr_num
        counter[SOS_id] = uttr_num
        counter[EOS_id] = uttr_num
        counter[UNK_id] = uttr_num
        return [1/((counter[i]+1) ** itf_lmd) for i in range(len(self.vocab["wtoi"]))]

    def init_epoch(self, epoch):
        print("Initializing data...")
        MAX_DIAL_LEN = self.hparams["MAX_DIAL_LEN"]
        batch_size = self.hparams["batch_size"]
        grouped_data = dict(
            {l: [
                    {key: [w for w in dial] for key, dial in data.items()}
                    for data in v
                ] for l, v in self.grouped_data.items()
            }
        )
        # Drop surplus data
        for l in range(2, MAX_DIAL_LEN+1):
            l_len = len(grouped_data[str(l)])
            surplus = l_len % batch_size
            for _ in range(surplus):
                pop_idx = np.random.randint(len(grouped_data[str(l)]))
                grouped_data[str(l)].pop(pop_idx)
        # Concat all dialogue length data
        self.data = [data for l in range(MAX_DIAL_LEN, 1, -1) for data in grouped_data[str(l)]]
        print("Done")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {k: v for k, v in self.data[idx].items()}
