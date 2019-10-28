import argparse
import pickle
import ipdb
import os
from embedding import Embedding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, default='../../data/task1/train.txt')
    parser.add_argument('--dataset_path', type=str, default='../../dataset/task1/')
    args = parser.parse_args()
    return args


def main(args):

    lines = open(args.raw_data_path, encoding='utf-8').read().strip().split('\n')
    words = set([word for line in lines for word in line.split(' ')])

    print('[*] Loading embedding...')
    embedding = Embedding('../../data/task1/cc.en.300.vec/cc.en.300.vec', words)
    embedding_path = os.path.join(args.dataset_path, 'embedding.pkl')

    with open(embedding_path, 'wb') as f:
        pickle.dump(embedding, f)
    print('[-] Preprocess done!')


if __name__ == '__main__':
    args = parse_args()
    main(args)
