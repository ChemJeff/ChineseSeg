# coding=utf-8
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

if __name__ == "__main__":

    ckpt_path = "./base_ckpt/"
    data_path = "./data/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = dataPreprocess.build_vocab_tag(data_path + 'train.txt')
    with open(data_path + 'vocab_tag.pkl', 'wb') as f:
        pickle.dump((word_to_ix, ix_to_word, tag_to_ix, ix_to_tag), file=f)
    with open(data_path + 'vocab_tag.pkl', 'rb') as f:
        word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = pickle.load(f)
    opt = argparse.Namespace()
    opt.device = device
    opt.corpus = data_path + 'train_corpus.pkl'
    opt.vocab_tag = data_path + 'vocab_tag.pkl'
    opt.embedding_dim = 64
    opt.hidden_dim = 128
    opt.batch_size = 5
    opt.vocab_size = len(word_to_ix)
    opt.tagset_size = len(tag_to_ix)

    train_corpus = dataPreprocess.corpus_convert(data_path + 'train.txt', 'train')
    with open(data_path + 'train_corpus.pkl', 'wb') as f:
        pickle.dump(train_corpus, file=f)

    dataset = dataLoader.DataSet(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, collate_fn=dataLoader.collate, drop_last=True, shuffle=True)

    train_corpus = dataPreprocess.corpus_convert(data_path + 'test.txt', 'test')
    with open(data_path + 'test_corpus.pkl', 'wb') as f:
        pickle.dump(train_corpus, file=f)
    opt.corpus = data_path + 'test_corpus.pkl'

    testdataset = dataLoader.DataSet(opt)
    testdataloader = torch.utils.data.DataLoader(
        testdataset, batch_size=opt.batch_size, collate_fn=dataLoader.collate, drop_last=True)

    model = BiLSTM_CRF(opt).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    with torch.no_grad():
        for packed_sent, idx_unsort, words in testdataloader:
            _, packed_tag = model(packed_sent)
            visualize(packed_sent, packed_tag, ix_to_word, ix_to_tag, idx_unsort, words)
            break

    iter_cnt = 0
    for epoch in range(
            40):  # again, normally you would NOT do 300 epochs, it is toy data
        stime = time.time()
        lastloss = 0.0
        for (packed_sent, packed_tag), idx_unsort, words in dataloader:
            model.zero_grad()

            loss = model.neg_log_likelihood(packed_sent, packed_tag)
            lastloss = loss.item()

            loss.backward()
            optimizer.step()
            iter_cnt += 1
            if iter_cnt % 100 == 0 :
                print("%s  last 100 iters: %d s, loss = %f" %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S, %Z'), time.time() - stime, loss.item()))
                stime = time.time()
            if iter_cnt % 1000 == 0 :
                try:
                    torch.save(model.state_dict(), ckpt_path + "base_e64_h128_iter%d.cpkt" % (iter_cnt))
                    print("checkpoint saved at \'%s\'" % (ckpt_path + "base_e64_h128_iter%d.cpkt" % (iter_cnt)))
                except Exception as e:
                    print(e)
                with torch.no_grad():
                    for packed_sent, idx_unsort, words in testdataloader:
                        _, packed_tag = model(packed_sent)
                        visualize(packed_sent, packed_tag, ix_to_word, ix_to_tag, idx_unsort, words)
                        break

        print("%s  last %d iters: %d s, loss = %f" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S, %Z'), iter_cnt % 100, time.time() - stime, lastloss))
        with torch.no_grad():
            for packed_sent, idx_unsort, words in testdataloader:
                _, packed_tag = model(packed_sent)
                visualize(packed_sent, packed_tag, ix_to_word, ix_to_tag, idx_unsort, words)
                break
