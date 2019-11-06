from __future__ import unicode_literals, print_function, division
import json
import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from ipdb import set_trace as pdb
from metrics import Accuracy
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
torch.manual_seed(42)


class Trainer:
    def __init__(self, device, model, word2index, index2word, batch_size, lr, teacher_forcing_ratio=0.5, ckpt_path='./'):

        self.device = device
        self.model = model
        self.word2index = word2index
        self.index2word = index2word
        self.batch_size = batch_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.ckpt_path = ckpt_path
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.opt, step_size=5, gamma=0.7)
        self.criterion = nn.CrossEntropyLoss(ignore_index=word2index['<PAD>'])
        self.accuracy = Accuracy(self.index2word)
        self.history = {'train': [], 'valid': []}

        self.SOS_INDEX = word2index['<SOS>']
        self.EOS_INDEX = word2index['<EOS>']

    def run_epoch(self, epoch, dataset, training):
        total_loss = 0
        self.model.train(training)
        self.accuracy.reset()

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=training,
        )

        description = 'Train' if training else 'Valid'

        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)
        for i, (sents, words_gt) in trange:     # (batch, max_len)
            input_tensor = sents.to(self.device)
            target_tensor = sents.to(self.device)
            loss = self.run_iter(input_tensor, target_tensor, training, words_gt)

            if training:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            loss = loss.item() / target_tensor.size(0)
            total_loss += loss
            trange.set_postfix(avg_loss=total_loss / (i+1),
                               score='%d/%d' % (self.accuracy.correct, self.accuracy.total),
                               accuracy="%.3f" % self.accuracy.value())

        print('accuracy: %.3f, score: %d/%d' % (self.accuracy.value(), self.accuracy.correct, self.accuracy.total))
        if training:
            self.history['train'].append({'accuracy': self.accuracy.value(), 'loss': total_loss / len(dataloader)})
        else:
            self.history['valid'].append({'accuracy': self.accuracy.value(), 'loss': total_loss / len(dataloader)})

        self.scheduler.step()

    def run_iter(self, input_tensor, target_tensor, training, words_gt):       # (batch, max_len)

        if training:
            output = self.model(input_tensor, target_tensor, self.teacher_forcing_ratio)    # (batch, max_len-1, voc)
        else:
            with torch.no_grad():
                output = self.model(input_tensor, target_tensor, 0)

        output = output.view(-1, output.shape[2])   # (b * max_len-1, voc)
        target = target_tensor[:, 1:].flatten()     # (b * max_len-1)
        loss = self.criterion(output, target)

        predict_indices = output.argmax(1).view(target_tensor.shape[0], -1)          # (b, max_len-1)
        self.accuracy(predict_indices, words_gt)

        return loss

    def save_models(self, epoch):
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        print('[-] Saving models to %s' % self.ckpt_path)

        if epoch % 5 == 0:
            torch.save({
                'model': self.model.state_dict(),
                'optim': self.opt.state_dict(),
                'history': self.history
            }, self.ckpt_path + 'models_epoch%d.ckpt' % epoch)

        with open(self.ckpt_path + 'history.json', 'w') as f:
            json.dump(self.history, f, indent=4)

    def load_models(self, epoch):
        print('[*] Loading model state')

        ckpt = torch.load(self.ckpt_path + 'models_epoch%d.ckpt' % epoch)
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device)
        self.opt.load_state_dict(ckpt['optim'])
        self.history = ckpt['history']
        for state in self.opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=self.device)
