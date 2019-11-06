import argparse
import pickle
import os
from ipdb import set_trace as pdb
from dataset import VocabDataset
from sklearn.model_selection import train_test_split
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, default='../../data/task1/train.txt')
    parser.add_argument('--dataset_path', type=str, default='../../dataset/task1/')
    parser.add_argument('--vocab', type=int, default=10)
    args = parser.parse_args()
    return args


def main(args):
    word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
    unk_idx = word2index['<UNK>']

    sentences = open(args.raw_data_path, encoding='utf-8').read().strip().split('\n')

    word_counter = Counter()
    for sentence in sentences:
        word_counter.update(sentence.split(' ')[1:-1])

    # print(sum([1 for word, time in word_counter.most_common() if time > 10]))
    for word, time in word_counter.most_common():
        if time > args.vocab and word not in word2index:
            index2word[len(word2index)] = word
            word2index[word] = len(word2index)
    print('vocab size: ', len(word2index))

    processed_datas = []
    for sentence in sentences:
        data = dict()
        data['indexed_sentence'] = [word2index.get(word, unk_idx) for word in sentence.split(' ')]
        data['label'] = sentence.split(' ')[1:]
        processed_datas.append(data)

    train_data, valid_data = train_test_split(processed_datas, test_size=0.1, random_state=520)
    train_dataset = VocabDataset(train_data, word2index['<PAD>'])
    valid_dataset = VocabDataset(valid_data, word2index['<PAD>'])

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
