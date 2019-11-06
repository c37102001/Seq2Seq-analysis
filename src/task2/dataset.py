import torch
from torch.utils.data import Dataset
import random


class PairDataset(Dataset):
    """
    Args:
        data: list of indexed sentence: [1, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 2]
        label: list of indexed next_sentence: [1, 41, 42, 31, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 2]
    """

    def __init__(self, data, label, word_dict, n_ctrl, max_len, testing=False):
        self.data = data
        self.label = label
        self.pad_index = word_dict['<PAD>']
        self.word_dict = word_dict
        self.n_ctrl = n_ctrl
        self.max_len = max_len + 2 + 2 * n_ctrl
        self.testing = testing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def collate_fn(self, datas):

        batch_index = []
        batch_label = []
        batch_samples = []
        for (index, label) in datas:
            samples = self.get_n_ctrl_idx(self.n_ctrl, len(label)-2)
            for sample in samples:
                index = index + [self.word_dict[str(sample)]] + [label[sample]]
            batch_index.append(index + [self.pad_index] * (self.max_len - len(index)))
            batch_label.append(label + [self.pad_index] * (self.max_len - len(label)))
            batch_samples.append(samples)

        return torch.LongTensor(batch_index), torch.LongTensor(batch_label), batch_samples

    def get_n_ctrl_idx(self, ncontrols, sent_len):
        sample_num = min(random.randint(1, ncontrols), sent_len)
        samples = random.sample([i for i in range(1, sent_len + 1)], sample_num)
        samples.sort()
        return samples
