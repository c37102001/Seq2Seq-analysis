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
        self.scheduler = StepLR(self.opt, step_size=10, gamma=0.5)
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
            # num_workers=8
        )

        description = 'Train' if training else 'Valid'

        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)
        for i, (sents, labels, ctrl_indices, ctrl_labels) in trange:     # (batch, max_len)
            input_tensor = sents.to(self.device)
            target_tensor = labels.to(self.device)
            loss, sample_predict = self.run_iter(epoch, input_tensor, target_tensor, ctrl_indices, ctrl_labels, training)

            if i % 2000 == 0:
                with open(self.output_path, 'a') as f:
                    f.write(self.indices2sentence(input_tensor[0]) + '\n')
                    f.write(sample_predict + '\n\n')

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

    def run_iter(self, epoch, input_tensor, target_tensor, ctrl_indices, ctrl_labels, training):       # (batch, max_len)

        if training:
            output = self.model(input_tensor, target_tensor, self.teacher_forcing_ratio)    # (batch, max_len-1, voc)
        else:
            with torch.no_grad():
                output = self.model(input_tensor, target_tensor, 0)

        ctrl_labels = ctrl_labels.to(self.device)
        ctrl_predict = torch.stack([output[(i, ctrl_indices[i])] for i in range(len(ctrl_indices))])   # (batch, voc)
        ctrl_loss = self.criterion(ctrl_predict, ctrl_labels)
        self.accuracy(ctrl_predict.argmax(1), ctrl_labels)  # (batch)

        output = output.view(-1, output.shape[2])   # (b * max_len-1, voc)
        target = target_tensor[:, 1:].flatten()     # (b * max_len-1)
        sentence_loss = self.criterion(output, target)

        loss = sentence_loss if epoch < 5 else sentence_loss + ctrl_loss
        # loss = ctrl_loss + sentence_loss * 10

        predict_indices = output.argmax(1).view(target_tensor.shape[0], -1)          # (b, max_len-1)
        return loss, self.indices2sentence(predict_indices[0])

    def indices2sentence(self, indices):        # (max_len - 1)
        return ' '.join([str(self.index2word[idx.item()]) for idx in indices])

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
