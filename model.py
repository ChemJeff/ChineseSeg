#coding = utf-8
'''
pyTorch version implementation of BiLSTM-CRF model
for sequence tagging tasks
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
import time
import datetime

# torch.manual_seed(1)

#####################################################################
# Helper functions to make the code more readable.


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix.get(w, len(to_ix) - 1) for w in seq]
    return torch.tensor(idxs, dtype=torch.int64)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):
    '''
    BiLSTM-CRF模型的pyTorch实现，包含CRF的相关算法，
    CRF的输入特征由BiLSTM提取
    '''
    def __init__(self, opt, tag_to_ix):
        super(BiLSTM_CRF, self).__init__()
        self.opt = opt
        self.device = self.opt.device
        self.embedding_dim = self.opt.embedding_dim
        self.hidden_dim = self.opt.hidden_dim
        self.batch_size = self.opt.batch_size
        self.vocab_size = self.opt.vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # 字向量从头训练，其中分配一个词向量给[UNK]即未登录字
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        # 状态转移特征的矩阵
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size, device=self.device))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2, device=self.device),
                torch.randn(2, self.batch_size, self.hidden_dim // 2, device=self.device))

    def _forward_alg(self, packed_feats):
        # 运行前向算法,为Batch化操作进行了简单修改，待优化

        padded_feats, feat_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_feats, batch_first=True, padding_value=0.0)

        batch_size = len(feat_lengths)
        alphas = torch.full((batch_size, ), 0.0, device=self.device)

        for idx in range(batch_size):
            feats = padded_feats[idx][:feat_lengths[idx]]

            # Do the forward algorithm to compute the partition function
            init_alphas = torch.full((1, self.tagset_size), -10000., device=self.device)
            # START_TAG has all of the score.
            init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

            # Wrap in a variable so that we will get automatic backprop
            forward_var = init_alphas

            # Iterate through the sentence
            for feat in feats:
                alphas_t = []  # The forward tensors at this timestep
                for next_tag in range(self.tagset_size):
                    # broadcast the emission score: it is the same regardless of
                    # the previous tag
                    emit_score = feat[next_tag].view(
                        1, -1).expand(1, self.tagset_size)
                    # the ith entry of trans_score is the score of transitioning to
                    # next_tag from i
                    trans_score = self.transitions[next_tag].view(1, -1)
                    # The ith entry of next_tag_var is the value for the
                    # edge (i -> next_tag) before we do log-sum-exp
                    next_tag_var = forward_var + trans_score + emit_score
                    # The forward variable for this tag is log-sum-exp of all the
                    # scores.
                    alphas_t.append(log_sum_exp(next_tag_var).view(1))
                forward_var = torch.cat(alphas_t).view(1, -1)
            terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
            alpha = log_sum_exp(terminal_var)
            alphas[idx] = alpha

        return alphas.view(-1)

    def _get_lstm_features(self, packed_sentence):
        # 注意全部替换成了使用packedSequence的Batch化操作，需要多次pack和pad
        # 整个模型中统一使用batch_first的顺序
        self.hidden = self.init_hidden()
        sentences, sent_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_sentence, batch_first=True, padding_value=0)
        embeds = self.word_embeds(sentences.to(self.device))
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, sent_lengths, batch_first=True)
        packed_lstm_out, self.hidden = self.lstm(packed_embeds, self.hidden)
        padded_lstm_out, sent_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_lstm_out, batch_first=True, padding_value=0.0)
        lstm_feats = self.hidden2tag(padded_lstm_out)
        packed_lstm_feats = torch.nn.utils.rnn.pack_padded_sequence(
            lstm_feats, sent_lengths, batch_first=True)
        return packed_lstm_feats

    def _score_sentence(self, packed_feats, packed_tags):
        # 为了进行Batch化操作进行简单修改急需优化
        # Gives the score of a provided tag sequence

        padded_feats, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_feats, batch_first=True, padding_value=0.0)
        padded_tags, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_tags, batch_first=True, padding_value=0.0)
        batch_size = len(seq_lengths)
        scores = alphas = torch.full((batch_size, ), 0.0, device=self.device)

        for idx in range(batch_size):
            tags = padded_tags[idx][:seq_lengths[idx]].to(self.device)
            feats = padded_feats[idx][:seq_lengths[idx]]

            score = torch.zeros(1, device=self.device)
            tags = torch.cat(
                [torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long, device=self.device), tags])
            for i, feat in enumerate(feats):
                score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
            score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
            scores[idx] = score

        return scores.view(-1)

    def _viterbi_decode(self, packed_feats):
        # 为了进行Batch化操作进行了简单的修改，有待优化
        path_scores = []
        best_paths = []
        padded_feats, feat_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_feats, batch_first=True, padding_value=0.0)
        batch_size = len(feat_lengths)

        for idx in range(batch_size):
            backpointers = []

            # Initialize the viterbi variables in log space
            init_vvars = torch.full((1, self.tagset_size), -10000., device=self.device)
            init_vvars[0][self.tag_to_ix[START_TAG]] = 0

            # forward_var at step i holds the viterbi variables for step i-1
            forward_var = init_vvars
            for feat in padded_feats[idx][:feat_lengths[idx]]:
                bptrs_t = []  # holds the backpointers for this step
                viterbivars_t = []  # holds the viterbi variables for this step

                for next_tag in range(self.tagset_size):
                    # next_tag_var[i] holds the viterbi variable for tag i at the
                    # previous step, plus the score of transitioning
                    # from tag i to next_tag.
                    # We don't include the emission scores here because the max
                    # does not depend on them (we add them in below)
                    next_tag_var = forward_var + self.transitions[next_tag]
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                # Now add in the emission scores, and assign forward_var to the set
                # of viterbi variables we just computed
                forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
                backpointers.append(bptrs_t)

            # Transition to STOP_TAG
            terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
            best_tag_id = argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]

            # Follow the back pointers to decode the best path.
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            # Pop off the start tag (we dont want to return that to the caller)
            start = best_path.pop()
            assert start == self.tag_to_ix[START_TAG]  # Sanity check
            best_path.reverse()
            path_scores.append(path_score)
            best_paths.append(torch.tensor(best_path))

        assert len(path_scores) == len(feat_lengths)
        packed_paths = torch.nn.utils.rnn.pack_sequence(best_paths)
        return path_scores, packed_paths

    def neg_log_likelihood(self, packed_sentence, packed_tags):
        # 为了Batch化操作进行简单的修改，注意这里的score都是长度为batch_size的向量
        packed_feats = self._get_lstm_features(packed_sentence)
        forward_score = self._forward_alg(packed_feats)
        gold_score = self._score_sentence(packed_feats, packed_tags)
        # print(forward_score, gold_score,sep="\n")
        # print(torch.sum(forward_score - gold_score))
        return torch.sum(forward_score - gold_score) / self.batch_size

    def forward(self, packed_sentence):
        # 为了Batch化进行了相应的修改
        # Get the emission scores from the BiLSTM
        packed_lstm_feats = self._get_lstm_features(packed_sentence)

        # Find the best path, given the features.
        scores, packed_tag_seq = self._viterbi_decode(packed_lstm_feats)
        return scores, packed_tag_seq

    pass

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
                    print(ix_to_word.get(word.item(), '[UNK]'),end=" ")
                elif ix_to_tag[tag.item()] in ['B', 'M']:
                    print(ix_to_word.get(word.item(), '[UNK]'),end="")
                else:
                    print('[UNK_tag]',end=" ")
            print()

    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    ckpt_path = "./base_ckpt/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # with open('vocab_tag.pkl', 'rb') as f:
    #     word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = pickle.load(f)
    word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = dataPreprocess.build_vocab_tag('train.txt')
    with open('vocab_tag.pkl', 'wb') as f:
        pickle.dump((word_to_ix, ix_to_word, tag_to_ix, ix_to_tag), file=f)
    opt = argparse.Namespace()
    opt.device = device
    opt.corpus = 'train_corpus.pkl'
    opt.vocab_tag = 'vocab_tag.pkl'
    opt.embedding_dim = 64
    opt.hidden_dim = 128
    opt.batch_size = 5
    opt.vocab_size = len(word_to_ix)
    ix_to_tag[4] = START_TAG
    ix_to_tag[5] = STOP_TAG
    tag_to_ix[START_TAG] = 4
    tag_to_ix[STOP_TAG] = 5
    train_corpus = dataPreprocess.corpus_convert('train.txt', 'train')
    with open('train_corpus.pkl', 'wb') as f:
        pickle.dump(train_corpus, file=f)
    dataset = dataLoader.DataSet(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, collate_fn=dataLoader.collate, drop_last=True)

    opt.corpus = 'val_corpus.pkl'

    testdataset = dataLoader.DataSet(opt)
    testdataloader = torch.utils.data.DataLoader(
        testdataset, batch_size=opt.batch_size, collate_fn=dataLoader.collate, drop_last=True)

    model = BiLSTM_CRF(opt, tag_to_ix).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    with torch.no_grad():
        for packed_sent, packed_tag in testdataloader:
            visual(*model(packed_sent), packed_sent)
            break

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(
            10):  # again, normally you would NOT do 300 epochs, it is toy data
        stime = time.time()
        iter_cnt = 0
        lastloss = 0.0
        for packed_sent, packed_tag in dataloader:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # # Step 2. Get our inputs ready for the network, that is,
            # # turn them into Tensors of word indices.
            # sentence_in = prepare_sequence(sentence, word_to_ix).cuda()
            # targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.int64).cuda()

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(packed_sent, packed_tag)
            lastloss = loss.item()

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
            iter_cnt += 1
            if iter_cnt % 100 == 0 :
                print("%s  last 100 iters: %d s, loss = %f" %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S, %Z'), time.time() - stime, loss.item()))
                stime = time.time()
            if iter_cnt % 1000 == 0 :
                try:
                    torch.save(model.state_dict(), ckpt_path+"checkpoint_%d_iter.cpkt" % (iter_cnt))
                except Exception as e:
                    print(e)
        print("%s  last %d iters: %d s, loss = %f" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S, %Z'), iter_cnt % 100, time.time() - stime, lastloss))

    # Check predictions after training
    with torch.no_grad():
        for packed_sent, packed_tag in testdataloader:
            visual(*model(packed_sent), packed_sent)
            break   

    pass
