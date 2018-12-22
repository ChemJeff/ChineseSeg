# coding=utf-8
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
import datetime
import pickle
import time

from model.model import *
from utils.utils import *
import utils.dataLoader as dataLoader
import utils.dataPreprocess as dataPreprocess

if __name__ == "__main__":

    ckpt_path = "./base_ckpt/base_e64_h128_iter150000.cpkt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

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
    opt.tagset_size = len(tag_to_ix)
    # train_corpus = dataPreprocess.corpus_convert('train.txt', 'train')
    # with open('train_corpus.pkl', 'wb') as f:
    #     pickle.dump(train_corpus, file=f)
    # dataset = dataLoader.DataSet(opt)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=opt.batch_size, collate_fn=dataLoader.collate, drop_last=True)

    opt.corpus = 'test_corpus.pkl'

    class BiLSTM_CRF(nn.Module):
        '''
        几个模块的拼接，测试
        '''
        def __init__(self, opt):
            super(BiLSTM_CRF, self).__init__()
            self.opt = opt
            self.embeds = WordEmbedding(self.opt)
            self.opt.tagset_size += 2    # for [START_TAG] and [STOP_TAG]
            self.bilstm = BiLSTM(self.opt)
            opt.tagset_size -= 2    # done inside CRF
            self.CRF = CRF(self.opt)

        def forward(self, packed_sent):
            packed_embeds = self.embeds(packed_sent)
            packed_feats = self.bilstm(packed_embeds)
            score, best_paths = self.CRF.decode(packed_feats)
            return score, best_paths

        def neg_log_likelihood(self, packed_sent, packed_tags):
            packed_embeds = self.embeds(packed_sent)
            packed_feats = self.bilstm(packed_embeds)
            return self.CRF.neg_log_likelihood(packed_feats, packed_tags)

    testdataset = dataLoader.DataSet(opt)
    testdataloader = torch.utils.data.DataLoader(
        testdataset, batch_size=opt.batch_size, collate_fn=dataLoader.collate, drop_last=True)

    model = BiLSTM_CRF(opt).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # load parameters from checkpoint given
    model.load_state_dict(torch.load(ckpt_path))

    # Check predictions from pretrained checkpoint
    with torch.no_grad():
        sample_cnt = 100
        for packed_sent, idx_unsort, words in testdataloader:
            _, packed_tags = model(packed_sent)
            visualize(packed_sent, packed_tags, ix_to_word, ix_to_tag, idx_unsort, words)
            # sample_cnt -= opt.batch_size
            if sample_cnt == 0:
                break

    pass
