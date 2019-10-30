import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):                       # (batch), hidden(1,batch,256)
        embedded = self.embedding(input).unsqueeze(0)       # (batch) -> (batch,256) -> (1,batch,256)
        output, hidden = self.gru(embedded, hidden)         # output(1,batch,256), hidden(1,batch,256)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):                       # input(1,b), hidden(1,b,h)
        output = self.embedding(input)                      # (1,b,h)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)           # (1,b,h), (1,b,h)
        output = self.softmax(self.out(output))             # (1, b, voc_size)
        return output, hidden                               # (1, b, voc_size), (1,b,h)

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
