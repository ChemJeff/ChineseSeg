#coding = utf-8
'''
pyTorch version implementation of BiLSTM-CRF model
for sequence tagging tasks
code for python 3.x and pyTorch == 0.4
'''

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import argparse
import utils
import pickle
import dataLoader
import dataPreprocess
import model
import time
import datetime

if __name__ == "__main__":

    def visual(_, packed_tag_seq, packed_sent_seq):
        padded_tag_seq, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_tag_seq, batch_first=True, padding_value=0)
        padded_sent_seq, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_sent_seq, batch_first=True, padding_value=0)
        batch_size = len(seq_lengths)
        for idx in range(batch_size):
            sent_seq = padded_sent_seq[idx][:seq_lengths[idx]]
            tag_seq = padded_tag_seq[idx][:seq_lengths[idx]]
            for word, tag in list(zip(sent_seq, tag_seq)):
                if ix_to_tag[tag.item()] in ['S', 'E']:
                    print(ix_to_word.get(word.item(), '[UNK]'),end="  ")
                elif ix_to_tag[tag.item()] in ['B', 'M']:
                    print(ix_to_word.get(word.item(), '[UNK]'),end="")
                else:
                    print('[UNK_tag]',end="  ")
            print()

    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    ckpt_path = "./base_ckpt/checkpoint_17000_iter.cpkt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('vocab_tag.pkl', 'rb') as f:
        word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = pickle.load(f)
    # word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = dataPreprocess.build_vocab_tag('train.txt')
    # with open('vocab_tag.pkl', 'wb') as f:
    #     pickle.dump((word_to_ix, ix_to_word, tag_to_ix, ix_to_tag), file=f)
    opt = argparse.Namespace()
    opt.device = device
    # opt.corpus = 'train_corpus.pkl'
    opt.vocab_tag = 'vocab_tag.pkl'
    opt.embedding_dim = 64
    opt.hidden_dim = 128
    opt.batch_size = 1
    opt.vocab_size = len(word_to_ix)
    opt.START_TAG = START_TAG
    opt.STOP_TAG = STOP_TAG
    ix_to_tag[4] = START_TAG
    ix_to_tag[5] = STOP_TAG
    tag_to_ix[START_TAG] = 4
    tag_to_ix[STOP_TAG] = 5
    # train_corpus = dataPreprocess.corpus_convert('train.txt', 'train')
    # with open('train_corpus.pkl', 'wb') as f:
    #     pickle.dump(train_corpus, file=f)
    # dataset = dataLoader.DataSet(opt)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=opt.batch_size, collate_fn=dataLoader.collate, drop_last=True)

    opt.corpus = 'val_corpus.pkl'

    testdataset = dataLoader.DataSet(opt)
    testdataloader = torch.utils.data.DataLoader(
        testdataset, batch_size=opt.batch_size, collate_fn=dataLoader.collate, drop_last=True)

    model = model.BiLSTM_CRF(opt, tag_to_ix).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # load parameters from checkpoint given
    model.load_state_dict(torch.load(ckpt_path))

    # Check predictions from pretrained checkpoint
    with torch.no_grad():
        sample_cnt = 100
        for packed_sent, packed_tag in testdataloader:
            visual(*model(packed_sent), packed_sent)
            # sample_cnt -= opt.batch_size
            if sample_cnt == 0:
                break

    pass
