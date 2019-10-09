# coding: utf-8


import numpy as np

from nltk.translate.bleu_score import sentence_bleu


def get_bleu(dial_list):
    bleu_list = []
    for dial in dial_list:
        bleu_list += [sentence_bleu(
            [dial["tgt"].split(" ")],
            dial["inf"][0].split(" "),
            weights=(0.5, 0.5, 0, 0),
        )]
    return np.mean(bleu_list) * 100
