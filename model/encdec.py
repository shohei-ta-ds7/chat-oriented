# coding: utf-8

from collections import Counter
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class EncDec(nn.Module):
    def __init__(self, hparams, n_words, tfdloss_weight, fix_embedding=False):
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
        self.decoder = LuongAttnDecoderRNN(
            "dot",
            self.embedding,
            hparams["hidden_size"],
            n_words,
            hparams["num_layers"],
            dropout=hparams["dropout"]
        )
        self.criterion = nn.NLLLoss(
            weight=torch.tensor(tfdloss_weight).cuda(),
            ignore_index=hparams["PAD_id"]
        )

    def forward(self, data, train=True):
        src = data["src"]
        src_len = data["src_len"]

        hidden_size = self.hparams["hidden_size"]
        num_layers = self.hparams["num_layers"]
        batch_size = src.size(0)
        src = src[:, -1]
        src_len = src_len[:, -1]
        if len(src_len) > 1:
            src_len, perm_index = src_len.sort(0, descending=True)
            src = src[perm_index]

        src = src.cuda()
        src_len = src_len.cuda()

        encoder_outputs, encoder_hidden = self.encoder(src, src_len)
        encoder_outputs = encoder_outputs[:, :, :hidden_size] + encoder_outputs[:, :, hidden_size:]
        decoder_hidden = (encoder_hidden[-1] + encoder_hidden[-2]).expand(num_layers, batch_size, hidden_size).contiguous()

        if train:
            tgt = data["tgt"][perm_index]
            return self.compute_loss(encoder_outputs, decoder_hidden, tgt)
        else:
            return self.inf_uttr(encoder_outputs, decoder_hidden)

    def compute_loss(self, encoder_outputs, initial_hidden, tgt):
        PAD_id = self.hparams["PAD_id"]
        tgt = tgt.cuda()
        batch_size = tgt.size(0)
        MAX_TGT_LEN = tgt.size(1)
        teacher_forcing_ratio = self.hparams["teacher_forcing_ratio"]

        loss = 0
        print_losses = []
        n_totals = 0

        decoder_input = torch.ones(batch_size, 1).type(torch.cuda.LongTensor)  # <s>

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = initial_hidden

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(MAX_TGT_LEN):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = tgt[:, t].view(-1, 1)
                # Calculate and accumulate loss
                mask_loss = self.criterion(F.log_softmax(decoder_output, dim=1), tgt[:, t])
                n_total = (tgt[:, t] != PAD_id).sum().item()
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_total)
                n_totals += n_total
        else:
            for t in range(MAX_TGT_LEN):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0]] for i in range(batch_size)]).cuda()
                # Calculate and accumulate loss
                mask_loss = self.criterion(F.log_softmax(decoder_output, dim=1), tgt[:, t])
                n_total = (tgt[:, t] != PAD_id).sum().item()
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_total)
                n_totals += n_total

        return loss, (sum(print_losses) / n_totals)

    def inf_uttr(self, encoder_outputs, initial_hidden):
        n_words = self.n_words
        EOS_id = self.hparams["EOS_id"]
        batch_size = encoder_outputs.size(0)
        hidden_size = self.hparams["hidden_size"]
        num_layers = self.hparams["num_layers"]
        beam_width = self.hparams["beam_width"]
        len_alpha = self.hparams["len_alpha"]
        # eos_gamma = self.hparams["eos_gamma"]
        suppress_lmd = self.hparams["suppress_lambda"]
        MAX_UTTR_LEN = self.hparams["MAX_UTTR_LEN"]

        decoder_hidden = initial_hidden
        # Inference tgt
        decoder_input = torch.ones(batch_size, 1).type(torch.cuda.LongTensor)  # <s>
        decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
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

            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            # suppression of repetitive generation
            suppressor = torch.ones(repet_counts.size()).cuda() / repet_counts.pow(suppress_lmd)
            decoder_output = topv.unsqueeze(1) + F.log_softmax(decoder_output * suppressor, dim=1)

            # Don't update output, hidden if the last word is </s>
            if len(eos_idx) > 0:
                decoder_output[eos_idx] = float("-inf")
                decoder_output[eos_idx, EOS_id] = prev_output[eos_idx, EOS_id]
                decoder_hidden[:, eos_idx] = prev_hidden[:, eos_idx]

            lp = torch.tensor([(5+len(inf_uttr)+1)**len_alpha / (5+1)**len_alpha for inf_uttr in inf_uttrs]).cuda()
            normalized_output = decoder_output / lp.unsqueeze(1)
            # normalized_output[:, EOS_id] -= eos_gamma * (MAX_UTTR_LEN / torch.tensor([len(uttr) for uttr in inf_uttrs]).cuda().float())
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

            # If all last words are </s>, break
            if sum([words[-1] == EOS_id for words in inf_uttrs]) == beam_width:
                break

        return inf_uttrs, topv


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, num_layers=1, dropout=0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers,
            dropout=(0 if num_layers == 1 else dropout),
            bidirectional=True, batch_first=True
        )

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.gru(packed, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output, hidden


# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ["dot", "general", "concat"]:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == "general":
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == "concat":
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == "general":
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == "concat":
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == "dot":
            attn_energies = self.dot_score(hidden, encoder_outputs)

        return F.softmax(attn_energies, dim=1).unsqueeze(2)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = embedding
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers,
            dropout=(0 if num_layers == 1 else dropout),
            batch_first=True
        )
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = F.dropout(embedded, p=self.dropout, training=self.training)
        rnn_output, hidden = self.gru(embedded, last_hidden)

        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = (attn_weights * encoder_outputs).sum(dim=1)
        rnn_output = rnn_output.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        return self.out(concat_output), hidden
