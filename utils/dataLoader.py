#coding = utf-8

'''
Load preprocessed data given in .pkl file and convert it
to the format used in pyTorch network 
'''
import os
import argparse
import numpy as np
import random
import torch
import torch.utils.data as data
import pickle
import multiprocessing


class DataSet(data.Dataset) :

    def __init__(self, opt) :
        self.opt = opt
        with open(self.opt.corpus, 'rb') as f :
            self.dataset = pickle.load(f)
        print("Dataset from \'%s\' loaded" % (self.opt.corpus))
        self.split = self.dataset['split']
        print("Dataset split: %s" %(self.split))
        with open(self.opt.vocab_tag, 'rb') as f :
            self.word2id, self.id2word, self.tag2id, self.id2tag = \
                pickle.load(f)
        self.vocab_size = len(self.id2word)
        print("Vocab size: %d" % (self.vocab_size))

    def __getitem__(self, index) :
        # print("Fetch next sample in dataset...")
        words = self.dataset['words'][index]
        # 未登录字全部转换到[UNK]对应的序号即vocab_size - 1
        wordids = [self.word2id.get(word, self.vocab_size - 1) for word in words]
        if self.split == 'train' :
            tags = self.dataset['tags'][index]
            tagids = [self.tag2id[tag] for tag in tags]
            return torch.tensor(wordids), torch.tensor(tagids)
        return torch.tensor(wordids)

    def __len__(self) :
        return len(self.dataset['words'])


def packBatch(batch_in_list) :
    '''
    return a torch.nn.utils.rnn.PackedSequence and indices to restore 
    the original order for a batch sorted by length in a decreasing order
    '''
    item_count = len(batch_in_list[0])
    split = 'train' if item_count == 2 else 'test'
    batch_len = [x for x in map(lambda item:len(item[0]), batch_in_list)]
    # print(batch_len)
    _, idx_sort = torch.sort(torch.tensor(batch_len), dim=0, descending=True)
    _, idx_unsort = torch.sort(idx_sort, dim=0)
    word_batch_in_list = [batch_in_list[i][0] for i in idx_sort]
    # batch_in_list.sort(key=lambda x:len(x), reverse=True)
    # print(word_batch_in_list)
    packed_words = torch.nn.utils.rnn.pack_sequence(word_batch_in_list)
    # print(packed_words)
    if split == 'train' :
        tag_batch_in_list = [batch_in_list[i][1] for i in idx_sort]
        packed_tags = torch.nn.utils.rnn.pack_sequence(tag_batch_in_list)
        return (packed_words, packed_tags), idx_unsort
    return packed_words, idx_unsort

def collate(batch_in_list) :
    '''
    collate function to convert raw data fetched from dataset
    to a torch.nn.utils.rnn.PackedSequence for LSTM input
    '''
    batch_size = len(batch_in_list)
    for i in range(batch_size) :
        batch_in_list[i] = [torch.tensor(item, dtype=torch.int64) for item in batch_in_list[i]]
    # print(batch_in_list)
    packed, _ = packBatch(batch_in_list)
    return packed

if __name__ == "__main__" :
    opt = argparse.Namespace()
    opt.corpus = 'train_corpus.pkl'
    opt.vocab_tag = 'vocab_tag.pkl'
    dataset = DataSet(opt)
    dataloader = data.DataLoader(dataset, batch_size=2, collate_fn=collate)
    sample = 10
    for data in dataloader :
        sample -= 1
        print(data, sep='\n')
        if sample < 0 :
            break
    pass