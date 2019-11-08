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
    sentences = open(args.raw_data_path, encoding='utf-8').read().strip().split('\n')

    word_counter = Counter()
    for sentence in sentences:
        word_counter.update(sentence.split(' ')[1:-1])

    # print(sum([1 for word, time in word_counter.most_common() if time > 10])))
    reduce_vocab_num = []
    unk_sent_total = [0]
    total_sent_num = 382785
    for i in range(1, 15):
        print(i)
        reduce_vocab_num.append(sum([1 for word, time in word_counter.most_common() if time <= i]))

        word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        unk_sent_count = 0
        for word, time in word_counter.most_common():
            if time > i and word not in word2index:
                word2index[word] = len(word2index)

        for j, sent in enumerate(sentences):
            for word in sent.split(' ')[1:-1]:
                if word not in word2index:
                    unk_sent_count += 1
                    continue
        unk_sent_total.append(unk_sent_count)
    # ratio = [i/j for i, j in zip(reduce_vocab_num, unk_sent_ratio[1:])]
    pdb()


if __name__ == '__main__':
    args = parse_args()
    main(args)
