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


class Accuracy(Metric):
    '''
    计算准确度
    可以使用topK参数设定计算K准确度
    Examples:
        >>> metric = Accuracy(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    '''
    def __init__(self, index2word):
        super(Accuracy, self).__init__()
        self.correct = 0
        self.total = 0
        self.reset()
        self.index2word = index2word

    def __call__(self, logits, target):     # (batch, max_len-1)
        batch_num = logits.size(0)
        # if self.total % 300000 == 0:
        #     pdb()
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
