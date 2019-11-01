import argparse
import pickle
import os
import ipdb
import torch
from model import EncoderRNN, DecoderRNN
from trainer import Trainer
from utils import load_pkl
from tqdm import tqdm
from dataset import VocabDataset
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--test_data_path', type=str, default='../../data/task1/train.txt')
    parser.add_argument('--dataset_path', type=str, default='../../dataset/task1/')
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--checkpoint_path', type=str, default='../../models/task1/')
    parser.add_argument('--load_models', action='store_true')
    parser.add_argument('--ckpt_epoch', type=int, default=-1)
    args = parser.parse_args()
    return args


def main(args):
    batch_size = 1280

    word2index = load_pkl(os.path.join(args.dataset_path, 'word2index.pkl'))
    index2word = load_pkl(os.path.join(args.dataset_path, 'index2word.pkl'))

    n_words = len(word2index)
    hidden_size = args.hidden_size
    device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")

    encoder = EncoderRNN(n_words, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, n_words).to(device)

    print('Loading model from ' + args.checkpoint_path + 'models_epoch%d.ckpt' % args.ckpt_epoch)
    ckpt = torch.load('%s%s/models_epoch%d.ckpt' % (args.checkpoint_path, args.arch, args.ckpt_epoch))
    encoder.load_state_dict(ckpt['encoder'])
    encoder.to(device)
    decoder.load_state_dict(ckpt['decoder'])
    decoder.to(device)

    print('Making dataset...')
    sentences = open(args.test_data_path, encoding='utf-8').read().strip().split('\n')
    sentences = sentences[:10000]
    max_len = max(len(sentence.split(' ')[1:]) for sentence in sentences)
    print(max_len)
    indexed_sentences = []
    for sentence in sentences:
        sentence = sentence.split(' ')[1:]
        indexed_sentences.append([word2index[word] for word in sentence] +
                                 [word2index['<PAD>']] * (max_len - len(sentence)))
    test_dataset = VocabDataset(indexed_sentences, word2index['<PAD>'])
    dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn, shuffle=False)

    print('Start predicting!')
    predict_sentences = torch.LongTensor().to(device)
    with torch.no_grad():
        for i, input_tensor in tqdm(enumerate(dataloader), total=len(dataloader), desc='Test'):      # (b, len)

            predict_words = torch.LongTensor().to(device)
            input_tensor = input_tensor.to(device)
            batch_size = input_tensor.size(0)
            seq_len = input_tensor.size(1)

            # encoder
            input_tensor = input_tensor.t()  # (len, b)
            encoder_hidden = encoder.initHidden(batch_size).to(device)  # (1,b,h)
            for ei in range(seq_len):
                encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
                # input_tensor (batch), hidden (1,batch,hidden), output=hidden (1,batch,hidden)

            # decoder
            decoder_input = torch.tensor([[word2index['<SOS>']] * batch_size], device=device)  # (1, b)
            decoder_hidden = encoder_hidden  # last encoder_hidden

            for _ in range(seq_len):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)  # (1,b) (1,b,h)
                topi = decoder_output.topk(1)[1].view(1, -1)    # (1,b,voc_size) -> (1,b,1) -> (1,b)
                decoder_input = topi.detach()  # detach from history as input (1,b)
                predict_words = torch.cat((predict_words, topi))    # (len, b)
                # ipdb.set_trace()

            predict_sentences = torch.cat((predict_sentences, predict_words.t()))
    predict_sentences = predict_sentences.cpu().numpy().tolist()

    result = []
    for sentence in predict_sentences:
        s = ['<SOS>']
        for index in sentence:
            s.append(index2word[index])
            if index2word[index] == '<EOS>':
                break
        result.append(' '.join(s))
    print(result[0:5])
    ipdb.set_trace()

    correct = sum([1 if p == t else 0 for p, t in zip(predict_sentences, indexed_sentences)])
    total = len(indexed_sentences)
    print('correct=%d, total=%d, accuracy=%.2f' % (correct, total, correct/total))


if __name__ == '__main__':
    args = parse_args()
    main(args)
