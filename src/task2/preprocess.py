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
    parser.add_argument('--test_data_path', type=str, default='../../data/task2/hw2.1-1_sample_testing_data.txt')
    parser.add_argument('--dataset_path', type=str, default='../../dataset/task2/')
    parser.add_argument('--n_ctrl', type=int, required=True)
    args = parser.parse_args()
    return args


def collect_words(dataset, max_len):
    word2index = {'<PAD>': 0}
    index2word = {0: '<PAD>'}

    for num in range(1, max_len + 1):
        index2word[len(word2index)] = str(num)
        word2index[str(num)] = len(word2index)

    for sent in dataset:
        for word in sent:
            if word not in word2index:
                index2word[len(word2index)] = word
                word2index[word] = len(word2index)

    index2word[len(word2index)] = '<UNK>'
    word2index['<UNK>'] = len(word2index)

    return word2index, index2word


def main(args):

    # make dictionary
    data = open(args.raw_data_path, encoding='utf-8').read().strip().split('\n')

    max_len = 25
    n_ctrl = args.n_ctrl
    data = [['<SOS>'] + [word for word in sent[:max_len]] + ['<EOS>'] for sent in data]
    word2index, index2word = collect_words(data, max_len=max_len)

    # make train/valid dataset
    index_dataset = [[word2index[word] for word in sent] for sent in data]
    index_data = index_dataset[:-1]
    index_label = index_dataset[1:]
    train_index, valid_index, train_label, valid_label = \
        train_test_split(index_data, index_label, test_size=0.2, random_state=42)
    train_dataset = PairDataset(train_index, train_label, word2index, n_ctrl, max_len)
    valid_dataset = PairDataset(valid_index, valid_label, word2index, n_ctrl, max_len)

    train_data_path = os.path.join(args.dataset_path, 'train_dataset_%d.pkl' % n_ctrl)
    valid_data_path = os.path.join(args.dataset_path, 'valid_dataset_%d.pkl' % n_ctrl)
    word2index_path = os.path.join(args.dataset_path, 'word2index.pkl')
    index2word_path = os.path.join(args.dataset_path, 'index2word.pkl')

    with open(train_data_path, 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(valid_data_path, 'wb') as f:
        pickle.dump(valid_dataset, f)
    with open(word2index_path, 'wb') as f:
        pickle.dump(word2index, f)
    with open(index2word_path, 'wb') as f:
        pickle.dump(index2word, f)
    print('Pickles saved.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
