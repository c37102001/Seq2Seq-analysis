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
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--checkpoint_path', type=str, default='../../models/task1/')
    parser.add_argument('--load_models', action='store_true')
    args = parser.parse_args()
    return args


def main(args):
    embedding_path = os.path.join(args.dataset_path, 'embedding.pkl')
    with open(embedding_path, 'rb') as f:
        embedding = pickle.load(f)

    lines = open(args.raw_data_path, encoding='utf-8').read().strip().split('\n')
    lines = lines[: len(lines)//10]
    train_data, valid_data = train_test_split(lines, test_size=0.2, random_state=520)

    checkpoint_path = args.checkpoint_path + "hidden%d_lr%.0e_emb/" % (args.hidden_size, args.lr)
    device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")

    encoder = EncoderRNN(embedding, args.hidden_size).to(device)
    decoder = DecoderRNN(embedding, args.hidden_size).to(device)

    trainer = Trainer(device, encoder, decoder, embedding, args.lr, ckpt_path=checkpoint_path)
    if args.load_models:
        trainer.load_models()

    for epoch in range(args.epochs):
        print('Epoch: {}'.format(epoch))
        trainer.run_epoch(train_data, True)
        trainer.run_epoch(valid_data, False)
        trainer.save_models(epoch)

    ipdb.set_trace()


if __name__ == '__main__':
    args = parse_args()
    main(args)
