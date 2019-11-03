from __future__ import unicode_literals, print_function, division
import random
import json
import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import ipdb
from metrics import Accuracy
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, device, encoder, decoder, word2index, index2word, batch_size, lr, teacher_forcing_ratio=0.5, ckpt_path='./'):

        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.word2index = word2index
        self.index2word = index2word
        self.batch_size = batch_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.ckpt_path = ckpt_path
        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optim = optim.Adam(self.decoder.parameters(), lr=lr)
        self.encoder_scheduler = StepLR(self.encoder_optim, step_size=30, gamma=0.5)
        self.decoder_scheduler = StepLR(self.decoder_optim, step_size=30, gamma=0.5)
        self.criterion = nn.CrossEntropyLoss(ignore_index=word2index['<PAD>'])
        self.history = {'train': [], 'valid': []}

        self.SOS_INDEX = word2index['<SOS>']
        self.EOS_INDEX = word2index['<EOS>']

    def run_epoch(self, epoch, dataset, training):
        print('Model will be saved to %s' % self.ckpt_path)
        total_loss = 0
        accuracy = Accuracy(self.index2word)
        self.encoder.train(training)
        self.decoder.train(training)
        self.teacher_forcing_ratio = 0.5 if training else 0

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=training,
            num_workers=4
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

        print('accuracy: %.2f, score: %d/%d' % (accuracy.value(), accuracy.correct, accuracy.total))
        if training:
            self.history['train'].append({'accuracy': accuracy.value(), 'loss': total_loss / len(dataloader)})
        else:
            self.history['valid'].append({'accuracy': accuracy.value(), 'loss': total_loss / len(dataloader)})

        self.encoder_scheduler.step()
        self.decoder_scheduler.step()

    def run_iter(self, input_tensor, target_tensor):       # (batch, max_len)
        loss = 0
        predict_tensor = torch.LongTensor().to(self.device)
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()

        # encoder
        input_tensor = input_tensor.transpose(1, 0)         # (max_len, batch)
        encoder_hidden = self.encoder.initHidden(input_tensor.size(1)).to(self.device)      # (1,1,256)
        for ei in range(input_tensor.size(0)):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)

        # decoder
        decoder_input = torch.tensor([[self.SOS_INDEX] * input_tensor.size(1)], device=self.device)     # (1, batch)
        decoder_hidden = encoder_hidden     # last encoder_hidden
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        if use_teacher_forcing:
            target_tensor = target_tensor.transpose(1, 0)  # (max_len, batch)
            for di in range(target_tensor.size(0)):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                # (1,b,voc_size), (1,b,hidden)
                loss += self.criterion(decoder_output.squeeze(0), target_tensor[di])  # (b,voc) (b)

                topi = decoder_output.topk(1)[1].view(1, target_tensor.size(1))  # (b)
                predict_tensor = torch.cat((predict_tensor, topi))
                decoder_input = target_tensor[di].unsqueeze(0)  # Teacher forcing
        else:
            target_tensor = target_tensor.transpose(1, 0)   # (max_len, batch)
            for di in range(target_tensor.size(0)):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                # (1,b,voc_size), (1,b,hidden)
                loss += self.criterion(decoder_output.squeeze(0), target_tensor[di])   # (b,voc) (b)

                topi = decoder_output.topk(1)[1].view(1, target_tensor.size(1))  # (b)
                predict_tensor = torch.cat((predict_tensor, topi))
                decoder_input = topi.detach()  # detach from history as input (1,b)
        return loss, predict_tensor.t()

    def save_models(self, epoch):
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        print('[-] Saving models')

        if epoch % 5 == 0:
            torch.save({
                'encoder': self.encoder.state_dict(),
                'encoder_optim': self.encoder_optim.state_dict(),
                'decoder': self.decoder.state_dict(),
                'decoder_optim': self.decoder_optim.state_dict(),
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
