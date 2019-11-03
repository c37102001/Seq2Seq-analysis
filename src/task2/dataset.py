import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """
    Args:
    data (list of dictionary):
    {
        'sentence': [1, 90, 1081, 242, 1151, 1597, 1112, 140, 2, 4, 321]
        'label': [321, 2]
    }
    """

    def __init__(self, data, pad_index):
        self.data = data
        self.pad_index = pad_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, datas):

        sentences = [data['sentence'] for data in datas]
        labels = [data['label'] for data in datas]

        max_sent_len = max([len(s) for s in sentences])
        max_label_len = max([len(l) for l in labels])

        padded_sents = [sent + [self.pad_index] * (max_sent_len - len(sent)) for sent in sentences]
        padded_labels = [label + [self.pad_index] * (max_label_len - len(label)) for label in labels]

        return torch.LongTensor(padded_sents), torch.LongTensor(padded_labels)
