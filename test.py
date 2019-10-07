# coding: utf-8

import os
import sys
import logging
import pickle

import tqdm

import torch

from utils.log_print import loginfo_and_print
from utils.batch_sampler import collate_fn


logger = logging.getLogger("logger").getChild("test")


def test(hparams, model, dataset, pickle_path):
    PAD_id = hparams["PAD_id"]
    EOS_id = hparams["EOS_id"]
    vocab = dataset.vocab
    model.eval()

    dial_list = []
    pbar = tqdm.tqdm(enumerate(dataset), total=len(dataset))
    for idx, data in pbar:
        data = collate_fn([data])
        pbar.set_description("Dial {}".format(idx+1))
        inf_uttrs, decoded_uttrs, likelihoods = inference(hparams, model, vocab, data)

        dial_list += [{
            "id": idx,
            "src": [
                " ".join([vocab["itow"].get(w, "<unk>") for w in s])
                for s in data["src"][0].numpy()
            ],
            "tgt": " ".join([
                vocab["itow"].get(w, "<unk>")
                for w in data["tgt"][0].numpy()
            ]),
            "tgt_id": [[id for id in data["tgt"][0].tolist() if id != PAD_id and id != EOS_id]],
            "inf_id": [[id for id in res if id != PAD_id and id != EOS_id] for res in inf_uttrs],
            "inf": decoded_uttrs,
            "likelihoods": likelihoods,
        }]

    with open(pickle_path, mode="wb") as f:
        pickle.dump(dial_list, f)


def inference(hparams, model, vocab, data):
    beam_width = hparams["beam_width"]
    src = data["src"]

    even = len(src[0]) % 2 == 0
    for i, u in enumerate(src[0]):
        if even:
            if i % 2 == 1:
                logger.info("User 1: {}".format(" ".join([vocab["itow"].get(w, "<unk>") for w in u])))
            else:
                logger.info("User 2: {}".format(" ".join([vocab["itow"].get(w, "<unk>") for w in u])))
        else:
            if i % 2 == 0:
                logger.info("User 1: {}".format(" ".join([vocab["itow"].get(w, "<unk>") for w in u])))
            else:
                logger.info("User 2: {}".format(" ".join([vocab["itow"].get(w, "<unk>") for w in u])))
    logger.info("User 2: {}".format(" ".join([vocab["itow"].get(w, "<unk>") for w in data["tgt"][0]])))

    with torch.no_grad():
        inf_uttrs, likelihoods = model(data, train=False)
    decoded_uttrs = [" ".join([vocab["itow"].get(w, "<unk>") for w in uttr]) for uttr in inf_uttrs]
    likelihoods = [l.item() for l in likelihoods]

    logger.info("inf:")
    for rank in range(beam_width):
        logger.info("{}[{:.2f}]: {} ".format(rank+1, likelihoods[rank], decoded_uttrs[rank]))

    return inf_uttrs, decoded_uttrs, likelihoods
