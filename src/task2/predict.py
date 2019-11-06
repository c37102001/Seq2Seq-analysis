import argparse
import pickle
import os
import ipdb
import torch
from model import Seq2Seq
from dataset import PairDataset
from trainer import Trainer
from utils import load_pkl
from torch.utils.data import DataLoader
torch.manual_seed(42)
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--n_ctrl', type=int, required=True)
    parser.add_argument('--ckpt_epoch', type=int, required=True)
    parser.add_argument('--dataset_path', type=str, default='../../dataset/task2/')
    parser.add_argument('--predict_data_path', type=str, default='../../data/task2/hw2.1-2_testing_data.txt')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--ckpt_path', type=str, default='../../models/task2/')
    args = parser.parse_args()
    return args


def main(args):

    print('[*] Making test dataset...')
    n_ctrl = args.n_ctrl
    word2index = load_pkl(os.path.join(args.dataset_path, 'word2index.pkl'))
    index2word = load_pkl(os.path.join(args.dataset_path, 'index2word.pkl'))
    data = open(args.predict_data_path, encoding='utf-8').read().strip().split('\n')
    max_len = max([len(sent) for sent in data])
    test_data = [[word2index.get(word, word2index['<UNK>']) for word in sent.split(' ')] for sent in data]
    test_dataset = PairDataset(test_data, None, word2index, n_ctrl, max_len, testing=True)

    print('[*] Loading model...')
    vocab_size = len(word2index) - 1
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
    ckpt_path = '%s%s/models_epoch%d.ckpt' % (args.ckpt_path, args.arch, args.ckpt_epoch)

    model = Seq2Seq(vocab_size, embedding_size, hidden_size, device).to(device)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    model.to(device)

    dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
    )

    print('[*] Start predicting...')
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Predicting')
    for i, (input_tensor, target_tensor) in trange:  # (batch, max_len)
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        with torch.no_grad():
            predict = model(input_tensor, target_tensor, teacher_forcing_ratio=0)     # (batch, max_len-1, voc)
        predict = predict.argmax(2)  # (b, max_len-1)

        with open('%s_output.txt' % args.arch, 'a+') as f:
            for predict_sent in predict:
                sent = ' '.join(['<SOS>'] + [str(index2word[idx.item()]) for idx in predict_sent])
                f.write(sent + '\n')


if __name__ == '__main__':
    args = parse_args()
    main(args)
