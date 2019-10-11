# coding: utf-8

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from model.modules import FFN
from model.modules import EncoderRNN
from model.modules import ContextRNN
from model.modules import HREDDecoderRNN
from model.modules import l2_pooling
from model.modules import sample_z


class VHCR(nn.Module):
    def __init__(self, hparams, n_words, itfloss_weight, fix_embedding=False):
        super().__init__()
        self.training = True
        self.hparams = hparams
        self.n_words = n_words
        self.embedding = nn.Embedding(
            n_words,
            hparams["hidden_size"],
            padding_idx=hparams["PAD_id"]
        )
        self.embedding.weight.requires_grad = not fix_embedding
        self.encoder = EncoderRNN(
            hparams["hidden_size"],
            self.embedding,
            hparams["num_layers"],
            dropout=hparams["dropout"]
        )
        self.context = ContextRNN(
            hparams["hidden_size"]*2,
            hparams["num_layers"],
            dropout=hparams["dropout"]
        )
        self.decoder = HREDDecoderRNN(
            self.embedding,
            hparams["hidden_size"],
            hparams["hidden_size"]*6,
            n_words,
            hparams["num_layers"],
            dropout=hparams["dropout"]
        )
        self.conv = ContextRNN(
            hparams["hidden_size"]*2,
            hparams["num_layers"],
            dropout=hparams["dropout"],
            bidirectional=True
        )
        self.context_init_ffn = FFN(
            hparams["hidden_size"]*2,
            hparams["hidden_size"]*2,
            dropout=hparams["dropout"],
            act="tanh"
        )
        self.conv_mean_ffn = FFN(
            hparams["hidden_size"]*2,
            hparams["hidden_size"]*2,
            dropout=hparams["dropout"],
            act="tanh"
        )
        self.conv_var_ffn = FFN(
            hparams["hidden_size"]*2,
            hparams["hidden_size"]*2,
            dropout=hparams["dropout"],
            act="relu"
        )
        self.uttr_mean_ffn = FFN(
            hparams["hidden_size"]*4,
            hparams["hidden_size"]*2,
            dropout=hparams["dropout"],
            act="tanh"
        )
        self.uttr_var_ffn = FFN(
            hparams["hidden_size"]*4,
            hparams["hidden_size"]*2,
            dropout=hparams["dropout"],
            act="relu"
        )
        self.criterion = nn.NLLLoss(
            weight=torch.tensor(itfloss_weight).cuda() if itfloss_weight else None,
            ignore_index=hparams["PAD_id"]
        )
        self.kldivloss = nn.KLDivLoss()

    def forward(self, data, train=True):
        src = data["src"]
        src_len = data["src_len"]

        batch_size = src.size(0)
        MAX_DIAL_LEN = src.size(1)
        MAX_UTTR_LEN = src.size(2)
        hidden_size = self.hparams["hidden_size"]
        num_layers = self.hparams["num_layers"]

        # Flatten
        src = src.view(batch_size * MAX_DIAL_LEN, -1)
        src_len = src_len.flatten()

        if len(src_len) > 1:
            src_len, perm_index = src_len.sort(0, descending=True)
            back_index = [(perm_index == i).nonzero().flatten().item() for i in range(perm_index.size(0))]
            src = src[perm_index]

        src = src.cuda()
        src_len = src_len.cuda()

        encoder_output, encoder_hidden = self.encoder(src, src_len)

        # Back from permutation
        if len(src_len) > 1:
            encoder_output = encoder_output[back_index]
            encoder_hidden = encoder_hidden[:, back_index]
            src_len = src_len[back_index]

        if train:
            post_encoder_output, post_encoder_hidden = self.encoder(
                data["tgt"].cuda(), data["tgt_len"].cuda()
            )

        # num_layers * num_directions, batch * MAX_DIAL_LEN, hidden_size
        # -> batch * MAX_DIAL_LEN, num_layers, num_directions * hidden_size
        encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(batch_size * MAX_DIAL_LEN, num_layers, -1)
        if train:
            # num_layers * num_directions, batch, hidden_size
            # -> batch, num_layers, num_directions * hidden_size
            post_encoder_hidden = post_encoder_hidden.transpose(1, 0).contiguous().view(batch_size, num_layers, -1)

        if self.hparams["l2_pooling"]:
            # Separate forward and backward hiddens
            encoder_output = encoder_output.view(batch_size * MAX_DIAL_LEN, MAX_UTTR_LEN, 2, -1)
            # L2 pooling
            forward = l2_pooling(encoder_output[:, :, 0], src_len)
            backward = l2_pooling(encoder_output[:, :, 1], src_len)
            encoder_hidden = torch.cat((forward, backward), dim=1).view(batch_size, MAX_DIAL_LEN, -1)
            if train:
                # Separate forward and backward hiddens
                post_encoder_output = post_encoder_output.view(batch_size, data["tgt"].size(1), 2, -1)
                # L2 pooling
                forward = l2_pooling(post_encoder_output[:, :, 0], data["tgt_len"])
                backward = l2_pooling(post_encoder_output[:, :, 1], data["tgt_len"])
                post_encoder_hidden = torch.cat((forward, backward), dim=1).view(batch_size, 1, -1)
        else:
            # Reshape to each uttr context (use only top hidden states)
            encoder_hidden = encoder_hidden[:, -1].view(batch_size, MAX_DIAL_LEN, -1)
            if train:
                post_encoder_hidden = post_encoder_hidden[:, -1].view(batch_size, 1, -1)

        # sample z_conv
        conv_output, _ = self.conv(encoder_hidden)
        conv_output = conv_output[:, -1]
        conv_output = conv_output[:, :hidden_size*2] + conv_output[:, hidden_size*2:]
        mean_conv = self.conv_mean_ffn(conv_output)
        var_conv = self.conv_var_ffn(conv_output)
        z_conv = sample_z(mean_conv, var_conv)
        context_hidden = self.context_init_ffn(z_conv).expand(
            num_layers, batch_size, hidden_size*2
        ).contiguous()

        # Input each dial context
        context_output, context_hidden = self.context(encoder_hidden, context_hidden)
        if train:
            post_context_output, _ = self.context(
                post_encoder_hidden, context_hidden
            )
        # sample z_uttr
        context_output = context_output[:, -1]
        context_z_conv = torch.cat((context_output, z_conv), dim=1)
        mean_uttr = self.uttr_mean_ffn(context_z_conv)
        var_uttr = self.uttr_var_ffn(context_z_conv)
        z_uttr = sample_z(mean_uttr, var_uttr)
        if train:
            post_context_output = post_context_output[:, 0]
            post_context_z_conv = torch.cat((post_context_output, z_conv), dim=1)
            mean_post_uttr = self.uttr_mean_ffn(post_context_z_conv)
            var_post_uttr = self.uttr_var_ffn(post_context_z_conv)
            post_z_uttr = sample_z(mean_post_uttr, var_post_uttr)

        # make context vector and initial decoder hidden
        context_output = torch.cat((context_output, z_uttr, z_conv), dim=1)
        decoder_hidden = (
            context_output[:, :hidden_size]
            + context_output[:, hidden_size:hidden_size*2]
            + context_output[:, hidden_size*2:hidden_size*3]
            + context_output[:, hidden_size*3:hidden_size*4]
            + context_output[:, hidden_size*4:hidden_size*5]
            + context_output[:, hidden_size*5:]
        )
        decoder_hidden = decoder_hidden.expand(num_layers, batch_size, hidden_size).contiguous()

        if train:
            return self.compute_loss(decoder_hidden, context_output, z_conv, z_uttr, post_z_uttr, data["tgt"])
        else:
            return self.beam_search(decoder_hidden[:, -1].unsqueeze(1).contiguous(), context_output[-1].unsqueeze(0))

    def compute_loss(self, initial_hidden, context_hidden, z_conv, z_uttr, post_z_uttr, tgt):
        PAD_id = self.hparams["PAD_id"]
        tgt = tgt.cuda()
        batch_size = tgt.size(0)
        MAX_TGT_LEN = tgt.size(1)
        teacher_forcing_ratio = self.hparams["teacher_forcing_ratio"]
        loss_name = self.hparams["loss"]
        mmi_lambda = self.hparams["mmi_lambda"]
        mmi_gamma = self.hparams["mmi_gamma"]

        loss = 0
        print_losses = []
        n_totals = 0

        decoder_input = torch.ones(batch_size, 1).type(torch.cuda.LongTensor)  # <sos>

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = initial_hidden
        # Zero tensor for U(T) of MMI
        ut_decoder_hidden = torch.zeros(decoder_hidden.size()).cuda()

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        # Forward batch of sequences through decoder one time step at a time
        for t in range(MAX_TGT_LEN):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, context_hidden
            )
            if loss_name == "mmi":
                ut_decoder_output, ut_decoder_hidden = self.decoder(
                    decoder_input, ut_decoder_hidden, context_hidden
                )

            # Calculate and accumulate loss
            mask_loss = self.criterion(F.log_softmax(decoder_output, dim=1), tgt[:, t])
            if loss_name == "mmi" and t+1 <= mmi_gamma:
                mask_loss -= mmi_lambda * self.criterion(F.log_softmax(ut_decoder_output, dim=1), tgt[:, t])
            n_total = (tgt[:, t] != PAD_id).sum().item()
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total

            if use_teacher_forcing:
                # Teacher forcing: next input is current target
                decoder_input = tgt[:, t].view(-1, 1)
            else:
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0]] for i in range(batch_size)]).cuda()

        conv_kldiv_loss = -self.kldivloss(
            F.log_softmax(z_conv, dim=1),
            F.softmax(torch.randn(z_conv.size()).cuda(), dim=1)
        )

        uttr_kldiv_loss = -self.kldivloss(
            F.log_softmax(z_uttr, dim=1),
            F.softmax(post_z_uttr, dim=1)
        )

        return (
            loss + conv_kldiv_loss + uttr_kldiv_loss,
            (sum(print_losses) + (conv_kldiv_loss.item() + uttr_kldiv_loss.item()) * batch_size) / n_totals
        )

    def beam_search(self, initial_hidden, context_hidden):
        n_words = self.n_words
        EOS_id = self.hparams["EOS_id"]
        batch_size = context_hidden.size(0)
        hidden_size = self.hparams["hidden_size"]
        num_layers = self.hparams["num_layers"]
        beam_width = self.hparams["beam_width"]
        len_alpha = self.hparams["len_alpha"]
        suppress_lmd = self.hparams["suppress_lambda"]
        MAX_UTTR_LEN = self.hparams["MAX_UTTR_LEN"]

        decoder_hidden = initial_hidden
        # Inference tgt
        decoder_input = torch.ones(batch_size, 1).type(torch.cuda.LongTensor)  # <sos>
        decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, context_hidden
        )
        topv, topi = F.log_softmax(decoder_output, dim=1).topk(beam_width)

        # beam width 1d tensor
        topv = topv.flatten()
        decoder_input = topi.t()
        decoder_hidden = decoder_hidden.expand(num_layers, beam_width, hidden_size).contiguous()
        inf_uttrs = [[id.item()] for id in decoder_input]
        repet_counts = torch.ones(beam_width, decoder_output.size(1)).type(torch.cuda.FloatTensor)

        # beam search
        decoder_output = decoder_output.expand(beam_width, n_words)
        for _ in range(MAX_UTTR_LEN-1):
            for b in range(beam_width):
                repet_counts[b, inf_uttrs[b][-1]] += 1
            eos_idx = [idx for idx, words in enumerate(inf_uttrs) if words[-1] == EOS_id]
            prev_output, prev_hidden = decoder_output, decoder_hidden

            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, context_hidden)

            # suppression of repetitive generation
            suppressor = torch.ones(repet_counts.size()).cuda() / repet_counts.pow(suppress_lmd)
            decoder_output = topv.unsqueeze(1) + F.log_softmax(decoder_output * suppressor, dim=1)

            # Don't update output, hidden if the last word is <eos>
            if len(eos_idx) > 0:
                decoder_output[eos_idx] = float("-inf")
                decoder_output[eos_idx, EOS_id] = prev_output[eos_idx, EOS_id]
                decoder_hidden[:, eos_idx] = prev_hidden[:, eos_idx]

            lp = torch.tensor([(5+len(inf_uttr)+1)**len_alpha / (5+1)**len_alpha for inf_uttr in inf_uttrs]).cuda()
            normalized_output = decoder_output / lp.unsqueeze(1)
            topv, topi = normalized_output.topk(beam_width)
            topv, topi = topv.flatten(), topi.flatten()
            topv, perm_index = topv.sort(0, descending=True)

            # Search Next inputs
            topv = topv[:beam_width]
            decoder_input = topi[perm_index[:beam_width]].view(-1, 1)
            former_index = perm_index[:beam_width] // beam_width
            decoder_output = decoder_output[former_index]
            decoder_hidden = decoder_hidden[:, former_index]
            repet_counts = repet_counts[former_index]
            inf_uttrs = [
                inf_uttrs[former]+[decoder_input[i].item()]
                if inf_uttrs[former][-1] != EOS_id
                else inf_uttrs[former]
                for i, former in enumerate(former_index)
            ]

            # If all last words are <eos>, break
            if sum([words[-1] == EOS_id for words in inf_uttrs]) == beam_width:
                break

        return inf_uttrs, topv
