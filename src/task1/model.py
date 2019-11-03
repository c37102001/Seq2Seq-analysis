import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from ipdb import set_trace as pdb


class EncoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)

    def forward(self, input):                    # (b,l,e)
        output, hidden = self.gru(input)         # hidden(1,b,h)
        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden):                       # input(b,1,e), hidden(1,b,h)
        output = F.relu(input)                              # (b,1,e)
        output, hidden = self.gru(output, hidden)           # (b,1,h), (1,b,h)
        output = self.out(output)                           # (b,1,voc)
        return output, hidden                               # (b,1,voc), (b,1,h)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, device):
        super(Seq2Seq, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = EncoderRNN(embedding_size, hidden_size)
        self.decoder = DecoderRNN(embedding_size, hidden_size, vocab_size)
        self.device = device

    def forward(self, input_tensor, target_tensor, teacher_forcing_ratio):     # (batch, max_len)
        batch_size, max_len = target_tensor.shape
        outputs = torch.zeros(batch_size, max_len - 1, self.vocab_size).to(self.device)     # (b, l, v)

        # encoder
        embedded = self.embedding(input_tensor)         # (b, l, e)
        encoder_hidden = self.encoder(embedded)         # (1, b, h)
        decoder_hidden = encoder_hidden

        # decoder
        decoder_input = target_tensor[:, 0:1]           # (b, 1)
        teacher_force = random.random() < teacher_forcing_ratio

        if teacher_force:
            for t in range(max_len - 1):
                decoder_input = self.embedding(decoder_input)   # (b, 1, e)
                output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)    # (b, 1, v), (b,1,h)
                outputs[:, t:t + 1] = output
                decoder_input = target_tensor[:, t+1:t+2]   # (b,1)
        else:
            for t in range(max_len - 1):
                decoder_input = self.embedding(decoder_input)  # (b, 1, e)
                output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)  # (b, 1, v), (b,1,h)
                outputs[:, t:t + 1] = output
                decoder_input = output.argmax(2)            # (b,1)

        return outputs
