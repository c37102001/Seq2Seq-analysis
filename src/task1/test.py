import argparse
import pickle
import os
import ipdb
import torch
from model import EncoderRNN, DecoderRNN
from trainer import Trainer
from utils import load_pkl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, default='../../data/task1/train.txt')
    parser.add_argument('--dataset_path', type=str, default='../../dataset/task1/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--checkpoint_path', type=str, default='../../models/task1/')
    parser.add_argument('--load_models', action='store_true')
    args = parser.parse_args()
    return args


def main(args):
    train_dataset = load_pkl(os.path.join(args.dataset_path, 'train_dataset.pkl'))
    valid_dataset = load_pkl(os.path.join(args.dataset_path, 'valid_dataset.pkl'))
    word2index = load_pkl(os.path.join(args.dataset_path, 'word2index.pkl'))
    index2word = load_pkl(os.path.join(args.dataset_path, 'index2word.pkl'))

    n_words = len(word2index)
    hidden_size = args.hidden_size
    lr = args.lr
    batch_size = args.batch_size
    checkpoint_path = args.checkpoint_path + "hidden%d_lr%.0e/" % (hidden_size, lr)
    device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")

    encoder = EncoderRNN(n_words, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, n_words).to(device)

    trainer = Trainer(device, encoder, decoder, word2index, index2word, batch_size, lr, ckpt_path=checkpoint_path)
    if args.load_models:
        trainer.load_models()

    for epoch in range(args.epochs):
        print('Epoch: {}'.format(epoch))
        trainer.run_epoch(epoch, train_dataset, True)
        trainer.run_epoch(epoch, valid_dataset, False)
        trainer.save_models(epoch)

    ipdb.set_trace()


if __name__ == '__main__':
    args = parse_args()
    main(args)
