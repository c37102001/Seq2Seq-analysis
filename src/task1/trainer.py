from __future__ import unicode_literals, print_function, division
import random
import json
import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import ipdb
from utils import tensorFromSentence
from metrics import Accuracy
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, device, encoder, decoder, word2index, batch_size, lr,
                 teacher_forcing_ratio=0.5, ckpt_path='./'):

        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.word2index = word2index
        self.batch_size = batch_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.ckpt_path = ckpt_path

        self.encoder_optim = optim.SGD(self.encoder.parameters(), lr=lr)
        self.decoder_optim = optim.SGD(self.decoder.parameters(), lr=lr)
        # self.encoder_scheduler = StepLR(self.encoder_optim, step_size=2, gamma=0.1)
        # self.decoder_scheduler = StepLR(self.decoder_optim, step_size=2, gamma=0.1)
        self.criterion = nn.NLLLoss()
        self.history = {'train': [], 'valid': []}

        self.SOS_INDEX = word2index['<SOS>']
        self.EOS_INDEX = word2index['<EOS>']

    def run_epoch(self, epoch, dataset, training):
        print('Model will be saved to %s' % self.ckpt_path)
        total_loss = 0
        accuracy = Accuracy()
        self.encoder.train(training)
        self.decoder.train(training)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=training
        )

        description = 'Train' if training else 'Valid'

        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)
        for i, batch in trange:     # (batch, max_len)
            input_tensor = batch.to(self.device)
            target_tensor = batch.to(self.device)
            loss, predict_tensor = self.run_iter(input_tensor, target_tensor)

            if training:
                loss.backward()
                self.encoder_optim.step()
                self.decoder_optim.step()

            accuracy(predict_tensor, target_tensor)
            loss = loss.item() / target_tensor.size(0)
            total_loss += loss
            trange.set_postfix(avg_loss=total_loss / (i+1),
                               score='%d/%d' % (accuracy.correct, accuracy.total),
                               accuracy="%.2f" % accuracy.value())

        if training:
            self.history['train'].append({'accuracy': accuracy.value(), 'loss': total_loss / len(dataloader)})
        else:
            self.history['valid'].append({'accuracy': accuracy.value(), 'loss': total_loss / len(dataloader)})

        # self.encoder_scheduler.step()
        # self.decoder_scheduler.step()

    def run_iter(self, input_tensor, target_tensor):       # (batch, max_len)
        loss = 0
        predict_tensor = torch.LongTensor().to(self.device)
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()

        # encoder
        encoder_hidden = self.encoder.initHidden(self.batch_size).to(self.device)      # (1,1,256)
        input_tensor = input_tensor.transpose(1, 0)         # (max_len, batch)
        for ei in range(input_tensor.size(0)):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            # input_tensor (batch), hidden (1,batch,hidden), output=hidden (1,batch,hidden)

        # decoder
        decoder_input = torch.tensor([[self.SOS_INDEX] * self.batch_size], device=self.device)     # (1, batch)
        decoder_hidden = encoder_hidden     # last encoder_hidden
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        if use_teacher_forcing and False:
            for di in range(target_tensor.size(0)):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                loss += self.criterion(decoder_output, target_tensor[di])

                topv, topi = decoder_output.topk(1)  # (1,1), (1,1)
                predict_tensor = torch.cat((predict_tensor, topi))
                decoder_input = target_tensor[di]  # Teacher forcing
        else:

            target_tensor = target_tensor.transpose(1, 0)   # (max_len, batch)
            for di in range(target_tensor.size(0)):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                # (1,b,voc_size), (1,b,hidden)
                loss += self.criterion(decoder_output.squeeze(0), target_tensor[di])   # (b,voc) (b)

                topi = decoder_output.topk(1)[1].view(1, self.batch_size)  # (b)
                predict_tensor = torch.cat((predict_tensor, topi))
                decoder_input = topi.detach()  # detach from history as input (1)
        return loss, predict_tensor.t()

    def save_models(self, epoch):
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        print('[-] Saving models')
        torch.save({
            'encoder': self.encoder.state_dict(),
            'encoder_optim': self.encoder_optim.state_dict(),
            'decoder': self.decoder.state_dict(),
            'decoder_optim': self.decoder_optim.state_dict()
        }, self.ckpt_path + 'models_epoch%d.ckpt' % epoch)

        with open(self.ckpt_path + 'history.json', 'w') as f:
            json.dump(self.history, f, indent=4)

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
