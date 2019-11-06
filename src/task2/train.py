import argparse
import pickle
import os
import ipdb
import torch
from model import Seq2Seq
from trainer import Trainer
from utils import load_pkl
torch.manual_seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, default='../../dataset/task2/')
    parser.add_argument('--n_ctrl', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--checkpoint_path', type=str, default='../../models/task2/')
    parser.add_argument('--load_models', action='store_true')
    args = parser.parse_args()
    return args


def main(args):
    train_dataset = load_pkl(os.path.join(args.dataset_path, 'train_dataset_%d.pkl' % args.n_ctrl))
    valid_dataset = load_pkl(os.path.join(args.dataset_path, 'valid_dataset_%d.pkl' % args.n_ctrl))
    word2index = load_pkl(os.path.join(args.dataset_path, 'word2index.pkl'))
    index2word = load_pkl(os.path.join(args.dataset_path, 'index2word.pkl'))

    vocab_size = len(word2index)
    batch_size = args.batch_size
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    lr = args.lr
    device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
    checkpoint_path = '%s%s/' % (args.checkpoint_path, args.arch)
    output_path = '%s%s.txt' % (checkpoint_path, args.arch)

    model = Seq2Seq(vocab_size, embedding_size, hidden_size, device).to(device)

    trainer = Trainer(device, model, word2index, index2word, batch_size, lr, ckpt_path=checkpoint_path,
                      output_path=output_path)
    if args.load_models:
        trainer.load_models()

    for epoch in range(args.epochs):
        print('Epoch: {}'.format(epoch))
        trainer.run_epoch(epoch, train_dataset, True)
        trainer.run_epoch(epoch, valid_dataset, False)
        trainer.save_models(epoch)


if __name__ == '__main__':
    args = parse_args()
    main(args)
