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
    args = parser.parse_args()
    return args


def main(args):

    # make dictionary
    sentences = open(args.raw_data_path, encoding='utf-8').read().strip().split('\n')
    del sentences[10148]    # error data
    max_len = min(max([len(s) for s in sentences]), 20)     # max_len=20
    for i in range(len(sentences)):
        sentences[i] = sentences[i][:max_len]

    word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}

    for num in range(1, max_len + 1):
        index2word[len(word2index)] = num
        word2index[num] = len(word2index)

    for sentence in sentences:
        for word in sentence:
            if word not in word2index:
                index2word[len(word2index)] = word
                word2index[word] = len(word2index)

    # preprocess data
    processed = []
    for i in range(len(sentences) - 1):
        s1 = [w for w in sentences[i]]
        s2 = [w for w in sentences[i + 1]]

        s2_idx = random.sample(range(len(s2)), k=1)[0]
        s2_word = s2[s2_idx]
        s1 = ['<SOS>'] + s1 + ['<EOS>', s2_idx+1, s2_word]

        processed.append({
            'sentence': [word2index[w] for w in s1],
            'sentence_label': [word2index['<SOS>']] + [word2index[w] for w in s2] + [word2index['<EOS>']],
            'control_idx': s1[-2],
            'control_label': word2index[s1[-1]]
        })

    # make dataset
    train_data, valid_data = train_test_split(processed, test_size=0.1, random_state=520)
    train_dataset = PairDataset(train_data, word2index['<PAD>'])
    valid_dataset = PairDataset(valid_data, word2index['<PAD>'])

    train_dataset_path = os.path.join(args.dataset_path, 'train_dataset.pkl')
    valid_dataset_path = os.path.join(args.dataset_path, 'valid_dataset.pkl')
    word2index_path = os.path.join(args.dataset_path, 'word2index.pkl')
    index2word_path = os.path.join(args.dataset_path, 'index2word.pkl')

    with open(train_dataset_path, 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(valid_dataset_path, 'wb') as f:
        pickle.dump(valid_dataset, f)
    with open(word2index_path, 'wb') as f:
        pickle.dump(word2index, f)
    with open(index2word_path, 'wb') as f:
        pickle.dump(index2word, f)
    print('Pickles saved.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
