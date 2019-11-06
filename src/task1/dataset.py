import torch
from torch.utils.data import Dataset


class VocabDataset(Dataset):
    """
    Args:
    data (list of dict):
        data['indexed_sentence']: [1, 35, 6, 51, 2]
        data['label']: ['tom', 'seemed', 'annoyed', '<EOS>']
    """
    def __init__(self, data, pad_index, testing=False):
        self.data = data
        self.pad_index = pad_index
        self.testing = testing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, datas):

        sents = [data['indexed_sentence'] for data in datas]
        labels = [data['label'] for data in datas]

        if not self.testing:
            sents_max_len = max([len(sent) for sent in sents])
            labels_max_len = sents_max_len - 1
        else:
            sents_max_len = max(max([len(sent) for sent in sents]), 10)
            labels_max_len = sents_max_len - 1
        batch_sents = [sent + [self.pad_index] * (sents_max_len - len(sent)) for sent in sents]     # (batch, max_len)
        batch_words_gt = [label + ['<PAD>'] * (labels_max_len - len(label)) for label in labels]      # (batch, max_len)
        return torch.LongTensor(batch_sents), batch_words_gt
