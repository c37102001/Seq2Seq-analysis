import torch
import pickle


def tensorFromSentence(word2index, sentence, device):
    return torch.tensor([word2index[word] for word in sentence.split(' ')]).view(-1, 1).to(device)


def load_pkl(pkl_path):
    with open(pkl_path, mode='rb') as f:
        obj = pickle.load(f)
    return obj
