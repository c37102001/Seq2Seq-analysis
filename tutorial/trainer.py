from __future__ import unicode_literals, print_function, division
import random
import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from utils import SOS_token, EOS_token
import ipdb
from matplotlib import pyplot as plt


class Trainer:
    def __init__(self, device, encoder, decoder, input_lang, output_lang, pairs,
                 max_length, learning_rate, teacher_forcing_ratio=0.5, ckpt_path='./models/'):

        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.pairs = pairs
        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.ckpt_path = ckpt_path

        self.encoder_optim = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optim = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        self.criterion = nn.NLLLoss()

    def start(self, n_iters, plot_every=100, fig_name='lc'):
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        training_pairs = [tensorsFromPair(self.input_lang, self.output_lang, random.choice(self.pairs), self.device)
                          for _ in range(n_iters)]

        tqdm.write('[-] Start training!')
        bar = tqdm(range(1, n_iters + 1), desc='[Total progress]')
        for iter in bar:
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]                 # _(7,1)
            target_tensor = training_pair[1]                # _(10,1)

            loss = self.train(input_tensor, target_tensor)
            print_loss_total += loss
            plot_loss_total += loss

            bar.set_postfix(avg_loss=print_loss_total / iter)

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        self.save_models()
        plt.plot(range(len(plot_losses)), plot_losses)
        plt.savefig('%s.png' % fig_name)

    def train(self, input_tensor, target_tensor):       # _(7,1), _(10,1)
        loss = 0
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()

        input_length = input_tensor.size(0)         # 7
        target_length = target_tensor.size(0)       # 10
        encoder_hidden = self.encoder.initHidden().to(self.device)      # (1,1,256)
        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)    # (maxl, 256)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)  # o(1,1,256), h(1,1,256)
            encoder_outputs[ei] = encoder_output[0, 0]  # (256)

        decoder_input = torch.tensor([[SOS_token]], device=self.device)     # (1,1)
        decoder_hidden = encoder_hidden     # last encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions(topi) as the next input(decoder_input)
            for di in range(target_length):
                # (1,10), (1,1,256), (1,maxl)
                decoder_output, decoder_hidden, decoder_attention = \
                    self.decoder(decoder_input, decoder_hidden, encoder_outputs)     # (1,1), (1,1,256), (maxl,256)

                # ipdb.set_trace()
                topv, topi = decoder_output.topk(1)  # (1,1), (1,1)
                decoder_input = topi.squeeze().detach()  # detach from history as input (1)

                loss += self.criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

        loss.backward()

        self.encoder_optim.step()
        self.decoder_optim.step()

        return loss.item() / target_length

    def save_models(self):
        print('[-] Saving models')
        torch.save({
            'encoder': self.encoder.state_dict(),
            'encoder_optim': self.encoder_optim.state_dict(),
            'decoder': self.decoder.state_dict(),
            'decoder_optim': self.decoder_optim.state_dict()
        }, self.ckpt_path + 'models.ckpt')

        print('[-] Models saved')

    def load_models(self):
        print('[*] Loading model state')

        ckpt = torch.load(self.ckpt_path + 'models.ckpt')
        self.encoder.load_state_dict(ckpt['encoder'])
        self.encoder.to(self.device)
        self.decoder.load_state_dict(ckpt['decoder'])
        self.decoder.to(self.device)
        self.encoder_optim.load_state_dict(ckpt['encoder_optim'])
        self.decoder_optim.load_state_dict(ckpt['decoder_optim'])
        for state in self.encoder_optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=self.device)
        for state in self.decoder_optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=self.device)
        print('[*] Models loaded')


def tensorsFromPair(input_lang, output_lang, pair, device):
    input_tensor = tensorFromSentence(input_lang, pair[0], device)
    target_tensor = tensorFromSentence(output_lang, pair[1], device)
    return input_tensor, target_tensor


def tensorFromSentence(lang, sentence, device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

