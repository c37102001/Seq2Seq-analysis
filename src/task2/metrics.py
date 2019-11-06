import torch
from ipdb import set_trace as pdb

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class CtrlAccuracy(Metric):

    def __init__(self, index2word):
        super(CtrlAccuracy, self).__init__()
        self.correct = 0
        self.total = 0
        self.reset()
        self.index2word = index2word

    def __call__(self, logits, target):     # (batch)
        self.correct += torch.sum(logits == target).item()
        self.total += logits.shape[0]

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return float(self.correct) / self.total

    def name(self):
        return 'accuracy'


class SentAccuracy(Metric):

    def __init__(self, index2word):
        super(SentAccuracy, self).__init__()
        self.correct = 0
        self.total = 0
        self.reset()
        self.index2word = index2word

    def __call__(self, logits, target):     # (batch, max_len-1)
        batch_num = logits.size(0)
        for i in range(batch_num):
            pad_idx = (target[i] == 0).nonzero()[0].item() if 0 in target[i] else len(target[i])
            if torch.equal(logits[i][:pad_idx], target[i][:pad_idx]):
                self.correct += 1
        self.total += batch_num

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return float(self.correct) / self.total

    def name(self):
        return 'accuracy'
