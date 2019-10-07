# coding: utf-8

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logging
import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from torch.utils.data import DataLoader

from utils.log_print import loginfo_and_print
from utils.batch_sampler import collate_fn
from utils.batch_sampler import RandomBatchSampler


logger = logging.getLogger("logger").getChild("train")

encdec = "encdec"
hred = "hred"


def run_epochs(
        hparams, model, dataset, valid_dataset, model_pre,
        valid_every=1000, save_every=1, checkpoint=None, pretrained=False
    ):
    learning_rate = hparams["learning_rate"]
    decoder_learning_ratio = hparams["decoder_learning_ratio"]
    decay_step = hparams["decay_step"]
    lr_decay = hparams["lr_decay"]
    max_epoch = hparams["max_epoch"]
    batch_size = hparams["batch_size"]
    max_gradient = hparams["max_gradient"]

    model.train()

    encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=learning_rate)
    encoder_scheduler = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=decay_step, gamma=lr_decay)
    decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    decoder_scheduler = optim.lr_scheduler.StepLR(decoder_optimizer, step_size=decay_step, gamma=lr_decay)
    if hparams["model_arc"] == hred:
        context_optimizer = optim.Adam(model.context.parameters(), lr=learning_rate)
        context_scheduler = optim.lr_scheduler.StepLR(context_optimizer, step_size=decay_step, gamma=lr_decay)
    else:
        context_optimizer = None
        context_scheduler = None

    if checkpoint and not pretrained:
        encoder_optimizer.load_state_dict(checkpoint["en_opt"])
        encoder_scheduler.load_state_dict(checkpoint["en_sch"])
        decoder_optimizer.load_state_dict(checkpoint["de_opt"])
        decoder_scheduler.load_state_dict(checkpoint["de_sch"])
        if hparams["model_arc"] == hred:
            context_optimizer.load_state_dict(checkpoint["cn_opt"])
            context_scheduler.load_state_dict(checkpoint["cn_sch"])

    start_epoch = checkpoint["epoch"]+1 if checkpoint and not pretrained else 1
    loginfo_and_print(
        logger,
        "Valid (Epoch {}): {:.4f}".format(start_epoch-1, valid(model, valid_dataset, batch_size, hparams))
    )
    for epoch in range(start_epoch, max_epoch+1):
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn,
            drop_last=True, num_workers=2,
            sampler=RandomBatchSampler(dataset, batch_size))
        pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
        loss = 0
        for idx, data in pbar:
            loss += iteration(
                model, data, hparams,
                encoder_optimizer, context_optimizer,
                decoder_optimizer, max_gradient
            )
            pbar.set_description("Train (Epoch {}): {:.4f}".format(epoch, loss/(idx+1)))

            encoder_scheduler.step()
            decoder_scheduler.step()
            if hparams["model_arc"] == hred:
                context_scheduler.step()

            if (idx+1) % valid_every == 0:  # validation
                logger.info("Valid: {:.4f}".format(valid(model, valid_dataset, batch_size, hparams)))

        logger.info("Train (Epoch {}): {:.4f}".format(epoch, loss/(idx+1)))
        loginfo_and_print(
            logger,
            "Valid (Epoch {}): {:.4f}".format(epoch, valid(
                model, valid_dataset, batch_size, hparams
            ))
        )

        if epoch % save_every == 0:
            torch.save({
                "epoch": epoch,
                "hparams": hparams,
                "model": model.state_dict(),
                "en_opt": encoder_optimizer.state_dict(),
                "cn_opt": context_optimizer.state_dict() if context_optimizer else None,
                "de_opt": decoder_optimizer.state_dict(),
                "en_sch": encoder_scheduler.state_dict(),
                "cn_sch": context_scheduler.state_dict() if context_scheduler else None,
                "de_sch": decoder_scheduler.state_dict(),
            }, model_pre+"_{}.tar".format(epoch))

        dataset.init_epoch(epoch+1)


def valid(model, valid_dataset, batch_size, hparams):
    model.eval()
    dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, collate_fn=collate_fn,
        drop_last=True, num_workers=2,
        sampler=RandomBatchSampler(valid_dataset, batch_size))
    loss = 0
    for idx, data in enumerate(dataloader):
        loss += iteration(model, data, hparams)
    model.train()
    return loss/(idx+1)


def iteration(
        model, data, hparams,
        encoder_optimizer=None, context_optimizer=None,
        decoder_optimizer=None, max_gradient=None
    ):
    with torch.set_grad_enabled(model.training):
        if model.training:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            if hparams["model_arc"] == hred:
                context_optimizer.zero_grad()

        loss, print_loss = model(data)

        if model.training:
            loss.backward()

            _ = clip_grad_norm(model.encoder.parameters(), max_gradient)
            _ = clip_grad_norm(model.decoder.parameters(), max_gradient)
            if hparams["model_arc"] == hred:
                _ = clip_grad_norm(model.context.parameters(), max_gradient)

            encoder_optimizer.step()
            decoder_optimizer.step()
            if hparams["model_arc"] == hred:
                context_optimizer.step()

    return print_loss
