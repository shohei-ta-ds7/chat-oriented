# coding: utf-8

import os
import sys

import torch
from torch.nn.functional import pad


def pad_batch(batch, key=None):
    if key is not None:
        feat = [d[key] for d in batch]
    else:
        feat = batch

    def _pad_data(x, length):
        return pad(x, (0, length - x.shape[0]), mode="constant", value=0)

    if isinstance(feat[0][0], list):  # 3D tensor
        lengths = [[len(x) for x in matrix] for matrix in feat]
        max_len = max([u for d in lengths for u in d])
        padded = torch.stack(
            [torch.stack(
                [
                    _pad_data(x.clone().detach(), max_len)
                    if type(x) is torch.Tensor
                    else _pad_data(torch.tensor(x), max_len)
                    for x in matrix
                ]
            ) for matrix in feat]
        )
        return padded, torch.tensor(lengths)
    else:  # 2D matrix
        lengths = [len(x) for x in feat]
        max_len = max(lengths)
        return torch.stack([
            _pad_data(x.clone().detach(), max_len) if type(x) is torch.Tensor
            else _pad_data(torch.tensor(x), max_len)
            for x in feat
        ]), torch.tensor(lengths)
