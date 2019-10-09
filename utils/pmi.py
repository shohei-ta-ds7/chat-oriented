# coding: utf-8

import tqdm
import pickle

import scipy.sparse as sparse

import numpy as np


def get_pmi(pmi_matrix_dic, dial_list):
    # between context and response
    MIN_COUNT = 5
    wtoi = pmi_matrix_dic["wtoi"]
    wcount = pmi_matrix_dic["wcount"]
    pmi_matrix = pmi_matrix_dic["ppmi_matrix"]

    context_list = [
        [
            wtoi[word] for uttr in dial["src"]
            for word in uttr.split(" ")
            if wcount[word] >= MIN_COUNT
        ] for dial in dial_list
    ]
    response_list = [
        [
            wtoi[word] for word in dial["inf"][0].split(" ")
            if wcount[word] >= MIN_COUNT
        ] for dial in dial_list
    ]

    pmi_list = []
    pbar = tqdm.tqdm(zip(context_list, response_list), total=len(dial_list))
    for context, response in pbar:
        if not context or not response:
            pmi_list += [0.0]
            continue

        wc_pmi_matrix = pmi_matrix[response][:, context].toarray()
        pmi_list += [np.average(np.max(wc_pmi_matrix, axis=1))]

    return np.average(pmi_list)
