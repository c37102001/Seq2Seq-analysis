from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):                       # input(1), hidden(1,1,256)
        embedded = self.embedding(input).view(1, 1, -1)     # (1) -> (1,256) -> (1,1,256)
        output = embedded
        output, hidden = self.gru(output, hidden)         # output(1,1,256), hidden(1,1,256)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):                       # input(1,1), hidden(1,1,256)
        output = self.embedding(input).view(1, 1, -1)       # (1,1,256) -> (1,1,256)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)           # (1,1,256), (1,1,256)
        output = self.softmax(self.out(output[0]))          # (1, 10)
        return output, hidden                               # (1, 10), (1,1,256)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):      # input(1,1), hidden(1,1,256), encoder_outputs(maxl,256)
        embedded = self.embedding(input).view(1, 1, -1)     # (1,1) -> (1,1,256) -> (1,1,256)
        embedded = self.dropout(embedded)

        # --- attention part: from embedded to output ---
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)  # (1,256*2) -> (1,maxl)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))   # (1,1,maxl)x(1,maxl,256)
                                                                                            # = (1,1,256)
        output = torch.cat((embedded[0], attn_applied[0]), 1)   # (1,256) + (1,256) = (1,512)
        output = self.attn_combine(output).unsqueeze(0)         # (1,512) -> (1,256) -> (1,1,256)
        # --- attention part: from embedded to output ---

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)               # (1,1,256)
        output = F.log_softmax(self.out(output[0]), dim=1)      # (1, 10)
        return output, hidden, attn_weights                     # (1,10), (1,1,256), (1,maxl)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
