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
