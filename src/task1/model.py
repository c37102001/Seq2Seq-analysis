import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(embedding.get_vocabulary_size(), embedding.get_dim())
        self.embedding.weight = nn.Parameter(embedding.vectors)
        self.gru = nn.GRU(embedding.get_dim(), hidden_size)

    def forward(self, input, hidden):                       # input(1), hidden(1,1,256)
        embedded = self.embedding(input).view(1, 1, -1)     # (1) -> (1,300) -> (1,1,300)
        output, hidden = self.gru(embedded, hidden)         # output(1,1,256), hidden(1,1,256)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(embedding.get_vocabulary_size(), embedding.get_dim())
        self.embedding.weight = nn.Parameter(embedding.vectors)
        self.gru = nn.GRU(embedding.get_dim(), hidden_size)
        self.out = nn.Linear(hidden_size, embedding.get_vocabulary_size())
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):                       # input(1,1), hidden(1,1,256)
        output = self.embedding(input).view(1, 1, -1)       # (1,1) -> (1,1,300)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)           # (1,1,256), (1,1,256)
        output = self.softmax(self.out(output[0]))          # (1, 10)
        return output, hidden                               # (1, 10), (1,1,256)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
