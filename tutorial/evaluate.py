from __future__ import unicode_literals, print_function, division
import pickle
import argparse
from pathlib import Path
import torch
import random
import json
import os

from models import EncoderRNN, AttnDecoderRNN
from preprocess import Lang
from trainer import tensorFromSentence
from utils import SOS_token, EOS_token, load_pkl

from matplotlib import pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker


class Evaluater:
    def __init__(self, device, encoder, decoder, input_lang, output_lang, max_length):
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.max_length = max_length

    def evaluate(self, sentence):
        with torch.no_grad():
            input_tensor = tensorFromSentence(self.input_lang, sentence, self.device)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden().to(self.device)

            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=self.device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(self.max_length, self.max_length)

            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(self.output_lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]

    def evaluateRandomly(self, pairs, n=10):
        print('[-] Start evaluating')
        for i in range(n):
            pair = random.choice(pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = self.evaluate(pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

    def evaluateAndShowAttention(self, input_sentence):
        output_words, attentions = self.evaluate(input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        self.showAttention(input_sentence, output_words, attentions)

    def showAttention(self, input_sentence, output_words, attentions):
        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + input_sentence.split(' ') +
                           ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./')
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

    hidden_size = config["model_cfg"]["hidden_size"]
    max_len = config["max_len"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, max_len, dropout_p=0.1).to(device)

    print('[-] Loading models')
    ckpt = torch.load(config["models_path"] + 'models.ckpt')
    encoder.load_state_dict(ckpt['encoder'])
    encoder.to(device)
    decoder.load_state_dict(ckpt['decoder'])
    decoder.to(device)

    evaluator = Evaluater(device, encoder, decoder, input_lang, output_lang, max_len)

    # Evaluate random samples
    evaluator.evaluateRandomly(pairs)

    evaluator.evaluateAndShowAttention("elle a cinq ans de moins que moi .")
    # evaluator.evaluateAndShowAttention("elle est trop petit .")
    # evaluator.evaluateAndShowAttention("je ne crains pas de mourir .")
    # evaluator.evaluateAndShowAttention("c est un jeune directeur plein de talent .")
    plt.savefig('attention.png')


if __name__ == '__main__':
    args = parse_args()
    main(args)
