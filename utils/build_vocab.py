# coding: utf-8

import os
import sys
import logging
from collections import Counter
import pickle

import tqdm

import numpy as np

import scipy.sparse as sparse

from utils.log_print import loginfo_and_print


logger = logging.getLogger("logger").getChild("vocab")

MIN_COUNT = 5


def build_vocab(out_dir, data):
    data = [[w.lower() for l in dial for w in l.split(" ")] for dial in data]
    wtoi, itow, wcount = get_vocab(data)
    loginfo_and_print(logger, "vocab size: {}".format(len(wtoi)))

    id_data = [[wtoi[word] for word in line if wcount[word] >= MIN_COUNT] for line in data]
    wwcount, wwcount_matrix = get_wwcount_matrix(id_data)

    pmi_matrix, ppmi_matrix, spmi_matrix, sppmi_matrix = get_pmi_matrix(wwcount, wwcount_matrix)

    vocab_dic = {
        "wtoi": wtoi,
        "itow": itow,
        "wcount": wcount,
        "wwcount_matrix": wwcount_matrix,
        "pmi_matrix": pmi_matrix,
        "ppmi_matrix": ppmi_matrix,
        "spmi_matrix": spmi_matrix,
        "sppmi_matrix": sppmi_matrix
    }
    out_path = os.path.join(out_dir, "vocab.pkl")
    with open(out_path, mode="wb") as f:
        pickle.dump(vocab_dic, f)


def get_vocab(data):
    wcount = Counter([word for line in data for word in line])
    wtoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    pbar = tqdm.tqdm(data, total=len(data))
    for line in pbar:
        for word in line:
            if wtoi.get(word, -1) < 0 and wcount[word] >= MIN_COUNT:
                wtoi[word] = len(wtoi)
    itow = {v: k for k, v in wtoi.items()}
    return wtoi, itow, wcount


def get_wwcount_matrix(data):
    WINDOW = 5
    wwcount = Counter()
    pbar = tqdm.tqdm(data, total=len(data))
    for line in pbar:
        for idx, word in enumerate(line):
            for w in range(1, WINDOW+1):
                if idx-w >= 0:
                    wwcount[(word, line[idx-w])] += 1
                if idx+w < len(line):
                    wwcount[(word, line[idx+w])] += 1

    row_idx = []
    col_idx = []
    cnt_values = []
    pbar = tqdm.tqdm(wwcount.items(), total=len(wwcount))
    for (word1, word2), count in pbar:
        row_idx += [word1]
        col_idx += [word2]
        cnt_values += [count]
    wwcount_matrix = sparse.csr_matrix((cnt_values, (row_idx, col_idx)))

    return wwcount, wwcount_matrix


def get_pmi_matrix(wwcount, wwcount_matrix):
    row_idx = []
    col_idx = []

    pmi_values = []
    ppmi_values = []
    spmi_values = []
    sppmi_values = []

    # smoothing
    alpha = 0.75
    nw2a_denom = np.sum(np.array(wwcount_matrix.sum(axis=0)).flatten()**alpha)
    sum_over_word1 = np.array(wwcount_matrix.sum(axis=0)).flatten()
    sum_over_word1_alpha = sum_over_word1 ** alpha
    sum_over_word2 = np.array(wwcount_matrix.sum(axis=1)).flatten()
    sum_wwcount = wwcount_matrix.sum()

    pbar = tqdm.tqdm(wwcount.items(), total=len(wwcount))
    for (word1, word2), count in pbar:
        nww = count
        Pww = nww / sum_wwcount
        nw1 = sum_over_word2[word1]
        Pw1 = nw1 / sum_wwcount
        nw2 = sum_over_word1[word2]
        Pw2 = nw2 / sum_wwcount

        nw2a = sum_over_word1_alpha[word2]
        Pw2a = nw2a / nw2a_denom

        pmi = np.log2(Pww/(Pw1*Pw2))
        ppmi = max(pmi, 0)

        spmi = np.log2(Pww/(Pw1*Pw2a))
        sppmi = max(spmi, 0)

        row_idx += [word1]
        col_idx += [word2]
        pmi_values += [pmi]
        ppmi_values += [ppmi]
        spmi_values += [spmi]
        sppmi_values += [sppmi]

    pmi_matrix = sparse.csr_matrix((pmi_values, (row_idx, col_idx)))
    ppmi_matrix = sparse.csr_matrix((ppmi_values, (row_idx, col_idx)))
    spmi_matrix = sparse.csr_matrix((spmi_values, (row_idx, col_idx)))
    sppmi_matrix = sparse.csr_matrix((sppmi_values, (row_idx, col_idx)))

    return pmi_matrix, ppmi_matrix, spmi_matrix, sppmi_matrix
