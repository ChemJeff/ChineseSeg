#coding = utf-8
'''
pyTorch version implementation of some model
and some structured preceptron model
for sequence tagging tasks
code for python 3.x and pyTorch == 0.4
'''
import argparse

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

class WordEmbedding(nn.Module):
    '''
    所使用的字向量，可以选择载入预训练的字向量(fixed or fine-tune)，也可以从头训练
    输入 torch.nn.utils.rnn.packedSequence:按batch打包好的句子的word_id
    输出 torch.nn.utils.rnn.packedSequence:按batch打包好的句子的word_embedding
    '''
    def __init__(self, opt):
        super(WordEmbedding, self).__init__()
        self.opt = opt
        self.device = self.opt.device
        self.vocab_size = self.opt.vocab_size
        self.embedding_dim = self.opt.embedding_dim
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
    
    def forward(self, packed_sentence):
        sentences, sent_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_sentence, batch_first=True, padding_value=0)
        embeds = self.word_embeds(sentences.to(self.device))
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, sent_lengths, batch_first=True)
        return packed_embeds

class BiLSTM(nn.Module):
    '''
    用于特征提取部分的BiLSTM网络
    输入 torch.nn.utils.rnn.packedSequence:按batch打包好的句子的word_embedding
    输出 torch.nn.utils.rnn.packedSequence:按batch打包好的用于计算tag的特征
    '''
    def __init__(self, opt):
        super(BiLSTM, self).__init__()
        self.opt = opt
        self.device = self.opt.device
        self.embedding_dim = self.opt.embedding_dim
        self.hidden_dim = self.opt.hidden_dim
        self.batch_size = self.opt.batch_size
        self.vocab_size = self.opt.vocab_size
        self.tagset_size = self.opt.tagset_size     # TO DO
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 这里使用简单地在每一次数据输入时都随机初始化(h_0, c_0)的方式
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2, device=self.device),
                torch.randn(2, self.batch_size, self.hidden_dim // 2, device=self.device))

    def forward(self, packed_embeds):
        # 整个模型中统一使用batch_first的顺序
        self.hidden = self.init_hidden()
        packed_lstm_out, self.hidden = self.lstm(packed_embeds, self.hidden)
        # 不需要记录(h_t, c_t)，因此直接覆盖
        padded_lstm_out, sent_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_lstm_out, batch_first=True, padding_value=0.0)
        lstm_feats = self.hidden2tag(padded_lstm_out)
        packed_lstm_feats = torch.nn.utils.rnn.pack_padded_sequence(
            lstm_feats, sent_lengths, batch_first=True)
        return packed_lstm_feats

class CRF(object):
    '''
    CRF算法的一个包装类
    方法 likelihood:计算给定句子和
        输入 
        输出 
    方法 decode:
        输入 
        输出 
    方法 neg_log_likelihood:
        输入 
        输出 
    '''

    # 辅助函数们
    def argmax(self, vec):
        _, idx = torch.max(vec, 1)
        return idx.item()

    def log_sum_exp(self, vec):
        '''
        一种比较数值稳定的方式将对数表示的数求和后再取对数
        '''
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + \
            torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def __init__(self, opt):
        self.opt = opt
        self.device = self.opt.device
        self.batch_size = self.opt.batch_size
        self.tagset_size = self.opt.tagset_size
        # 临时分配两个[START]和[STOP]tag，标号连续
        self.START_TAG = self.tagset_size
        self.STOP_TAG = self.tagset_size + 1
        self.tagset_size += 2   # 仅在内部计算有效，从外部看来，tag集并没有修改
        # 状态转移特征的矩阵
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size, device=self.device))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.START_TAG, :] = -10000
        self.transitions.data[:, self.STOP_TAG] = -10000        
    
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
            init_alphas[0][self.START_TAG] = 0.

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
                    alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
                forward_var = torch.cat(alphas_t).view(1, -1)
            terminal_var = forward_var + self.transitions[self.STOP_TAG]
            alpha = self.log_sum_exp(terminal_var) 
            alphas[idx] = alpha

        return alphas.view(-1)

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
                [torch.tensor([self.START_TAG], dtype=torch.long, device=self.device), tags])
            for i, feat in enumerate(feats):
                score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
            score = score + self.transitions[self.STOP_TAG, tags[-1]]
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
            init_vvars[0][self.START_TAG] = 0

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
                    best_tag_id = self.argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                # Now add in the emission scores, and assign forward_var to the set
                # of viterbi variables we just computed
                forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
                backpointers.append(bptrs_t)

            # Transition to STOP_TAG
            terminal_var = forward_var + self.transitions[self.STOP_TAG]
            best_tag_id = self.argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]

            # Follow the back pointers to decode the best path.
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            # Pop off the start tag (we dont want to return that to the caller)
            start = best_path.pop()
            assert start == self.START_TAG  # Sanity check
            best_path.reverse()
            path_scores.append(path_score)
            best_paths.append(torch.tensor(best_path))

        assert len(path_scores) == len(feat_lengths)
        packed_paths = torch.nn.utils.rnn.pack_sequence(best_paths)
        return path_scores, packed_paths

    def likelihood(self, packed_feats, packed_tags):
        return self._score_sentence(packed_feats, packed_tags)

    def decode(self, packed_feats):
        return self._viterbi_decode(packed_feats)

    def neg_log_likelihood(self, packed_feats, packed_tags):
        # 为了Batch化操作进行简单的修改，注意这里的score都是长度为batch_size的向量
        forward_score = self._forward_alg(packed_feats)
        gold_score = self._score_sentence(packed_feats, packed_tags)
        return torch.sum(forward_score - gold_score) / self.batch_size
