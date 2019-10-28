import torch


def tensorFromSentence(word2index, sentence, device):
    return torch.tensor([word2index[word] for word in sentence.split(' ')]).view(-1, 1).to(device)

