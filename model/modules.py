# coding: utf-8

from collections import Counter
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class FFN(nn.Module):
    def __init__(self, input_size, output_size, dropout=0, act="tanh"):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.act = act
        if self.act not in ["tanh", "relu"]:
            raise ValueError(self.act, "is not an appropriate activation function.")
        if self.act == "tanh":
            self.act = torch.tanh
        elif self.act == "relu":
            self.act = torch.relu

        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input_feat):
        output = F.dropout(
            self.linear(input_feat),
            p=self.dropout,
            training=self.training
        )
        return self.act(output)


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


class ContextRNN(nn.Module):
    def __init__(self, hidden_size, num_layers=1, dropout=0, bidirectional=False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers,
            dropout=(0 if num_layers == 1 else dropout),
            bidirectional=bidirectional, batch_first=True
        )

    def forward(self, input_seq, hidden=None):
        output, hidden = self.gru(input_seq, hidden)
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


class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x):
        assert x.size(-1) % self.pool_size == 0
        m, _ = x.view(*x.size()[:-1], x.size(-1) // self.pool_size, self.pool_size).max(-1)
        return m


class HREDDecoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size, context_hidden_size, output_size, num_layers=1, dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.context_hidden_size = context_hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = embedding
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers,
            dropout=(0 if num_layers == 1 else dropout),
            batch_first=True
        )

        self.embedding_linear = nn.Linear(hidden_size, hidden_size*2)
        self.hidden_linear = nn.Linear(hidden_size, hidden_size*2, bias=False)
        self.context_linear = nn.Linear(context_hidden_size, hidden_size*2, bias=False)
        # Maxout activation
        self.maxout = Maxout(2)

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_step, last_hidden, context_hidden):
        embedded = self.embedding(input_step)
        embedded = F.dropout(embedded, p=self.dropout, training=self.training)
        rnn_output, hidden = self.gru(embedded, last_hidden)

        pre_active = self.embedding_linear(embedded.squeeze(1)) \
                    + self.hidden_linear(rnn_output.squeeze(1)) \
                    + self.context_linear(context_hidden)
        pre_active = self.maxout(pre_active)

        return self.out(pre_active), hidden


def l2_pooling(hiddens, src_len):
    return torch.stack(
        [
            torch.sqrt(
                torch.sum(torch.pow(hiddens[b][:src_len[b]], 2), dim=0)
                /src_len[b].type(torch.FloatTensor).cuda()
            )
            for b in range(hiddens.size(0))
        ]
    )


def sample_z(mean, var):
    epsilon = torch.randn(mean.size()).cuda()
    return mean + torch.sqrt(var) * epsilon
