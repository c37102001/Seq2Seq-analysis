import argparse
import pickle
import os
import ipdb
import torch
from model import EncoderRNN, DecoderRNN
from trainer import Trainer
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, default='../../data/task1/train.txt')
    parser.add_argument('--dataset_path', type=str, default='../../dataset/task1/')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--checkpoint_path', type=str, default='../../models/task1/')
    parser.add_argument('--load_models', action='store_true')
    args = parser.parse_args()
    return args


def main(args):
    word2index_path = os.path.join(args.dataset_path, 'word2index.pkl')
    index2word_path = os.path.join(args.dataset_path, 'index2word.pkl')
    with open(word2index_path, 'rb') as f:
        word2index = pickle.load(f)
    with open(index2word_path, 'rb') as f:
        index2word = pickle.load(f)

    lines = open(args.raw_data_path, encoding='utf-8').read().strip().split('\n')
    # lines = lines[: len(lines)//1000]
    train_data, valid_data = train_test_split(lines, test_size=0.2, random_state=520)

    n_words = len(word2index)
    hidden_size = args.hidden_size
    lr = args.lr
    checkpoint_path = args.checkpoint_path + "hidden%d_lr%.0e/" % (hidden_size, lr)
    device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")

    encoder = EncoderRNN(n_words, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, n_words).to(device)

    trainer = Trainer(device, encoder, decoder, word2index, lr, ckpt_path=checkpoint_path)
    if args.load_models:
        trainer.load_models()

    for epoch in range(args.epochs):
        print('Epoch: {}'.format(epoch))
        trainer.run_epoch(epoch, train_data, True)
        trainer.run_epoch(epoch, valid_data, False)
        trainer.save_models(epoch)

    ipdb.set_trace()


if __name__ == '__main__':
    args = parse_args()
    main(args)
