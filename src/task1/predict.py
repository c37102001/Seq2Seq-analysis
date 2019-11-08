import torch
import argparse
import os
from model import Seq2Seq
from utils import load_pkl
from ipdb import set_trace as pdb
from dataset import VocabDataset
from torch.utils.data import DataLoader
torch.manual_seed(42)
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, default='../../data/task1/hw2.0_testing_data.txt')

    parser.add_argument('--arch', type=str, default='24_0_b32e512h128_linearRelu_vocab10')
    parser.add_argument('--ckpt_epoch', type=int, default=40)
    parser.add_argument('--output_path', type=str, default='../../result/task1.txt')

    parser.add_argument('--dataset_path', type=str, default='../../dataset/task1/')
    parser.add_argument('--ckpt_path', type=str, default='../../models/task1/')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--cuda', type=int, default=0)

    args = parser.parse_args()
    return args


def main(args):

    print('[*] Making dataset...')
    word2index = load_pkl(os.path.join(args.dataset_path, 'word2index.pkl'))
    index2word = load_pkl(os.path.join(args.dataset_path, 'index2word.pkl'))
    sentences = open(args.test_data_path, encoding='utf-8').read().strip().split('\n')
    EOS_IDX = word2index['<EOS>']

    processed_datas = []
    for sentence in sentences:
        data = dict()
        data['indexed_sentence'] = [word2index.get(word, word2index['<UNK>']) for word in sentence.split(' ')]
        data['label'] = sentence.split(' ')[1:]
        processed_datas.append(data)
    test_dataset = VocabDataset(processed_datas, word2index['<PAD>'], testing=True)

    print('[*] Loading model...')
    vocab_size = len(word2index)
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
    ckpt_path = '%s%s/models_epoch%d.ckpt' % (args.ckpt_path, args.arch, args.ckpt_epoch)

    ckpt = torch.load(ckpt_path)
    model = Seq2Seq(vocab_size, embedding_size, hidden_size, device).to(device)
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
    for i, (sents, words_gt) in trange:  # (batch, max_len)
        input_tensor = sents.to(device)
        target_tensor = sents.to(device)
        with torch.no_grad():
            predict = model(input_tensor, target_tensor, teacher_forcing_ratio=0)   # (batch, max_len-1, voc)
        predict = predict.argmax(2)   # (batch, max_len-1)

        with open(args.output_path, 'a+') as f:
            for predict_sent in predict:
                eos_idx = (predict_sent == EOS_IDX).nonzero()[0].item() + 1 if EOS_IDX in predict_sent \
                    else len(predict_sent)
                sent = ' '.join(['<SOS>'] + [str(index2word[idx.item()]) for idx in predict_sent[:eos_idx]])
                f.write(sent + '\n')


if __name__ == '__main__':
    args = parse_args()
    main(args)
