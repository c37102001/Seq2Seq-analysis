from __future__ import unicode_literals, print_function, division
from io import open
import random
import pickle
import argparse
import os
import re
import unicodedata
import json


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def readLangs(raw_data_path, lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('%s/%s-%s.txt' % (raw_data_path, lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p, max_length):
    return len(p[0].split(' ')) < max_length and \
        len(p[1].split(' ')) < max_length and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]


def prepareData(raw_data_path, lang1, lang2, max_length, reverse=False):
    input_lang, output_lang, pairs = readLangs(raw_data_path, lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./')
    parser.add_argument('--lang1', type=str, default='eng', help='lang1')
    parser.add_argument('--lang2', type=str, default='fra', help='lang2')
    parser.add_argument('--no-reverse', dest='reverse', action='store_false')
    parser.set_defaults(reverse=True)
    args = parser.parse_args()
    return args


def main(args):
    config_path = os.path.join(args.config_path, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    lang1 = args.lang1
    lang2 = args.lang2
    max_len = config['max_len']
    reverse = args.reverse
    raw_data_path = config['raw_data_path']
    input_lang, output_lang, pairs = prepareData(raw_data_path, lang1, lang2, max_len, reverse)
    print(random.choice(pairs))

    dataset_path = config['dataset_path']
    input_lang_path = os.path.join(dataset_path, 'input_lang.pkl')
    output_lang_path = os.path.join(dataset_path, 'output_lang.pkl')
    pairs_path = os.path.join(dataset_path, 'pairs.pkl')

    with open(input_lang_path, 'wb') as f:
        pickle.dump(input_lang, f)
    with open(output_lang_path, 'wb') as f:
        pickle.dump(output_lang, f)
    with open(pairs_path, 'wb') as f:
        pickle.dump(pairs, f)
    print('Pickles saved.')


if __name__ == '__main__':
    args = parse_args()
    main(args)

