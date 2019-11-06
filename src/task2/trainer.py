from __future__ import unicode_literals, print_function, division
import random
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
    def __init__(self, device, model, word2index, index2word, batch_size, lr, teacher_forcing_ratio=0.5,
                 ckpt_path='./', output_path='test.txt'):

        self.device = device
        self.model = model
        self.word2index = word2index
        self.index2word = index2word
        self.batch_size = batch_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.ckpt_path = ckpt_path
        self.output_path = output_path
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.opt, step_size=20, gamma=0.5)
        self.criterion = nn.CrossEntropyLoss(ignore_index=word2index['<PAD>'])
        self.accuracy = Accuracy(self.index2word)
        self.history = {'train': [], 'valid': []}

        self.SOS_INDEX = word2index['<SOS>']
        self.EOS_INDEX = word2index['<EOS>']

        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

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
        for i, (input_tensor, target_tensor, samples) in trange:     # (batch, max_len)
            input_tensor = input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)
            loss, predict = self.run_iter(epoch, input_tensor, target_tensor, samples, training)

            if training:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            if i % 3000 == 0:
                sample_batch = random.randint(0, input_tensor.shape[0]-1)
                with open(self.output_path, 'a+') as f:
                    f.write(self.indices2sentence(input_tensor[sample_batch]) + '\n')
                    f.write(self.indices2sentence(target_tensor[sample_batch]) + '\n')
                    f.write(self.indices2sentence(predict[sample_batch]) + '\n\n')

            loss = loss.item() / target_tensor.size(0)
            total_loss += loss
            trange.set_postfix(avg_loss=total_loss / (i+1),
                               score='%d/%d' % (self.accuracy.correct, self.accuracy.total),
                               accuracy="%.3f" % self.accuracy.value())

        print('accuracy: %.3f, score: %d/%d' %
              (self.accuracy.value(), self.accuracy.correct, self.accuracy.total))
        if training:
            self.history['train'].append({'ctrl_acc': self.accuracy.value(), 'loss': total_loss / len(dataloader)})
        else:
            self.history['valid'].append({'ctrl_acc': self.accuracy.value(), 'loss': total_loss / len(dataloader)})
        with open(self.output_path, 'a') as f:
            f.write(
                'epoch:%d, accuracy:%.3f' % (epoch, self.accuracy.value()) +
                '\n=====================================================================\n'
            )

        self.scheduler.step()

    def run_iter(self, epoch, input, target, samples, training):       # (batch, max_len)

        if training:
            predict = self.model(input, target, self.teacher_forcing_ratio)    # (batch, max_len-1, voc)
        else:
            with torch.no_grad():
                predict = self.model(input, target, 0)

        predict = predict.view(-1, predict.shape[2])   # (b * max_len-1, voc)
        loss_target = target[:, 1:].flatten()     # (b * max_len-1)
        loss = self.criterion(predict, loss_target)

        predict = predict.argmax(1).view(target.shape[0], -1)          # (b, max_len-1)
        self.accuracy(predict, target[:, 1:], samples)

        return loss, predict

    def indices2sentence(self, indices):        # (max_len - 1)
        return ' '.join([self.index2word[idx.item()] for idx in indices])

    def save_models(self, epoch):
        print('[-] Saving models to %s' % self.ckpt_path)

        if epoch % 5 == 0:
            torch.save({
                'model': self.model.state_dict(),
                'optim': self.opt.state_dict(),
                'history': self.history
            }, self.ckpt_path + 'models_epoch%d.ckpt' % epoch)

        with open(self.ckpt_path + 'history.json', 'w') as f:
            json.dump(self.history, f, indent=4)

    def load_models(self):
        print('[*] Loading model state')

        ckpt = torch.load(self.ckpt_path + 'models.ckpt')
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device)
        self.opt.load_state_dict(ckpt['optim'])
        self.history = ckpt['history']
        for state in self.opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=self.device)
