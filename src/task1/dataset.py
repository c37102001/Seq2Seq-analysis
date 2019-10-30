import torch
from torch.utils.data import Dataset


class VocabDataset(Dataset):
    """
    Args:
    data (list): size:(382785, various)
        [[1, 35, 6, 2], [1, 35, 6, 1577, 230, ..., 2], ...]
    """
    def __init__(self, data, pad_index):
        self.data = data
        self.pad_index = pad_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, datas):

        max_len = max([len(data) for data in datas])

        batch = [data + [self.pad_index] * (max_len - len(data)) for data in datas]     # (batch, max_len)
        batch = torch.LongTensor(batch)                                                 # (batch, max_len)
        return batch
