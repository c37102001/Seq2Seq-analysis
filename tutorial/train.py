from __future__ import unicode_literals, print_function, division
import pickle
import argparse
from pathlib import Path
import torch
import json
import os
import ipdb

from models import EncoderRNN, AttnDecoderRNN
from trainer import Trainer
from utils import CustomUnpickler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./')
    parser.add_argument('--train_iters', type=int, default=75000, help='total run_iter iters')
    parser.add_argument('--load_models', action='store_true')
    parser.add_argument('--ordinal', type=int, default=0)
    parser.add_argument('--fig_name', type=str, default='lc')
    args = parser.parse_args()
    return args


def main(args):
    config_path = os.path.join(args.config_path, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    print('[-] Loading pickles')
    dataset_path = Path(config["dataset_path"])
    input_lang = CustomUnpickler(open(dataset_path / 'input_lang.pkl', 'rb')).load()
    output_lang = CustomUnpickler(open(dataset_path / 'output_lang.pkl', 'rb')).load()
    pairs = CustomUnpickler(open(dataset_path / 'pairs.pkl', 'rb')).load()

    # input_lang = load_pkl(dataset_path / 'input_lang.pkl')
    # output_lang = load_pkl(dataset_path / 'output_lang.pkl')
    # pairs = load_pkl(dataset_path / 'pairs.pkl')

    max_len = config["max_len"]
    lr = config["model_cfg"]["lr"]
    hidden_size = config["model_cfg"]["hidden_size"]
    train_iters = args.train_iters
    device = torch.device("cuda:%s" % args.ordinal if torch.cuda.is_available() else "cpu")

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, max_len, dropout_p=0.1).to(device)

    trainer = Trainer(device, encoder, attn_decoder, input_lang, output_lang, pairs, max_len, lr,
                      ckpt_path=config["models_path"])
    if args.load_models:
        trainer.load_models()
    trainer.run_epoch(train_iters)


if __name__ == '__main__':
    args = parse_args()
    main(args)
