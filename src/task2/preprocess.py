import argparse
import pickle
import os
import random
from dataset import PairDataset
from sklearn.model_selection import train_test_split
from ipdb import set_trace as pdb
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, default='../../data/task2/hw2.1_corpus.txt')
    parser.add_argument('--dataset_path', type=str, default='../../dataset/task2/')
    parser.add_argument('--n_ctrl', type=int, default=1)
    args = parser.parse_args()
    return args


def collect_words(dataset, max_len):
    word_dict = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    index_dict = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}

    for num in range(1, max_len + 1):
        index_dict[len(word_dict)] = str(num)
        word_dict[str(num)] = len(word_dict)

    for sent in dataset:
        for word in sent:
            if word not in word_dict:
                index_dict[len(word_dict)] = word
                word_dict[word] = len(word_dict)
    return word_dict, index_dict


def main(args):

    # make dictionary
    data = open(args.raw_data_path, encoding='utf-8').read().strip().split('\n')
    del data[10148]    # error data

    max_len = 25
    n_ctrl = args.n_ctrl
    data = [['<SOS>'] + [word for word in sent[:max_len]] + ['<EOS>'] for sent in data]
    word_dict, index_dict = collect_words(data, max_len=max_len)

    # make dataset
    index_dataset = [[word_dict[word] for word in sent] for sent in data]
    index_data = index_dataset[:-1]
    index_label = index_dataset[1:]
    train_index, valid_index, train_label, valid_label = \
        train_test_split(index_data, index_label, test_size=0.2, random_state=42)
    train_data = PairDataset(train_index, train_label, word_dict['<PAD>'], n_ctrl, max_len)
    valid_data = PairDataset(valid_index, valid_label, word_dict['<PAD>'], n_ctrl, max_len)

    train_data_path = os.path.join(args.dataset_path, 'train_dataset.pkl')
    valid_data_path = os.path.join(args.dataset_path, 'valid_dataset.pkl')
    word2index_path = os.path.join(args.dataset_path, 'word2index.pkl')
    index2word_path = os.path.join(args.dataset_path, 'index2word.pkl')

    with open(train_data_path, 'wb') as f:
        pickle.dump(train_data, f)
    with open(valid_data_path, 'wb') as f:
        pickle.dump(valid_data, f)
    with open(word2index_path, 'wb') as f:
        pickle.dump(word_dict, f)
    with open(index2word_path, 'wb') as f:
        pickle.dump(index_dict, f)
    print('Pickles saved.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
