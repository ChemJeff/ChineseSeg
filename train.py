# coding=utf-8
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import argparse
import datetime
import pickle
import time
import os
import sys

from model.model import *
from utils.utils import *
import utils.dataLoader as dataLoader
import utils.dataPreprocess as dataPreprocess

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

def add_summary_value(writer, key, value, iteration):
    if writer is None:
        return
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']

class Logger(object):
    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.log = open(file_name, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

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

    ckpt_path = "./checkpoint/"
    data_path = "./data/"
    log_dir = "./log/"

    time_suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    log_dir = log_dir + "run_%s/" % (time_suffix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    stime = time.time()
    sys.stdout = Logger(log_dir + "log")
    tf_summary_writer = tf and tf.summary.FileWriter(log_dir + "tflog")
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice: %s" % (device))

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

    opt.lr = 1e-4
    opt.weight_decay = 1e-4
    opt.iter_cnt = 0       # if non-zero, load checkpoint at iter (#iter_cnt)

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

    print("All necessites prepared, time used: %f s\n" % (time.time() - stime))
    model = BiLSTM_CRF(opt).to(device)

    from_ckpt = False
    try:
        print("Load checkpoint at %s" %(ckpt_path + "base_e64_h128_iter%d.cpkt" % (opt.iter_cnt)))
        # load parameters from checkpoint given
        model.load_state_dict(torch.load(ckpt_path + "base_e64_h128_iter%d.cpkt" % (opt.iter_cnt)))
        print("Success\n")
        from_ckpt = True
    except Exception as e:
        print("Failed, check the path and permission of the checkpoint")
        exit(0)

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    # Check predictions before training
    with torch.no_grad():
        for packed_sent, idx_unsort, words in testdataloader:
            _, packed_tag = model(packed_sent)
            visualize(packed_sent, packed_tag, ix_to_word, ix_to_tag, idx_unsort, words)
            break

    iter_cnt = opt.iter_cnt if from_ckpt is True else 0
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
                print("%s  last 100 iters: %d s, iter = %d, loss = %f" %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S, %Z'), time.time() - stime, iter_cnt, loss.item()))
                sys.stdout.flush()
                add_summary_value(tf_summary_writer, "train_loss", loss.item(), iter_cnt)
                add_summary_value(tf_summary_writer, 'learning_rate', get_lr(optimizer), iter_cnt)
                tf_summary_writer.flush()
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
