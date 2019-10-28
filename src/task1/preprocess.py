import argparse
import pickle
import ipdb
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, default='../../data/task1/run_iter.txt')
    parser.add_argument('--dataset_path', type=str, default='../../dataset/task1/')
    args = parser.parse_args()
    return args


def main(args):
    word2index = dict()
    index2word = dict()
    sentences = []

    # lines = open(args.raw_data_path, encoding='utf-8').read().strip().split('\n')
    # for line in lines:
    #     for word in line.split(' '):
    #         if word not in word2index:
    #             index2word[len(word2index)] = word
    #             word2index[word] = len(word2index)

    with open(args.raw_data_path) as fp:
        for sentence in fp:
            sentence = sentence.replace("\n", "")
            sentence = sentence.split(' ')
            for word in sentence:
                if word not in word2index:
                    index2word[len(word2index)] = word
                    word2index[word] = len(word2index)

    dataset_path = args.dataset_path
    word2index_path = os.path.join(dataset_path, 'word2index.pkl')
    index2word_path = os.path.join(dataset_path, 'index2word.pkl')

    with open(word2index_path, 'wb') as f:
        pickle.dump(word2index, f)
    with open(index2word_path, 'wb') as f:
        pickle.dump(index2word, f)
    print('Pickles saved.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
