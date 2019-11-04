from ipdb import set_trace as pdb
import torch
import random

x = torch.randn(4, 3, 2)
print('x: ', x)
ctrl = [1, 0, 2, 1]
print('ctrl: ', ctrl)

y = torch.stack([x[(i, ctrl[i])] for i in range(len(ctrl))])

pdb()
