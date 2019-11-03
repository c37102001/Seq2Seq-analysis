import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)

    def forward(self, input, hidden):                       # (batch), hidden(1,batch,256)
        embedded = self.embedding(input).unsqueeze(0)       # (b) -> (b,e) -> (1,b,e)
        output, hidden = self.gru(embedded, hidden)         # output(1,b,h), hidden(1,b,h)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden):                       # input(1,b), hidden(1,b,h)
        output = self.embedding(input)                      # (1,b,e)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)           # (1,b,h), (1,b,h)
        output = self.out(output)                           # (1, b, voc_size)
        return output, hidden                               # (1, b, voc_size), (1,b,h)

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
