# coding=utf-8
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

class CRF(nn.Module):
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
        super(CRF, self).__init__()
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
    
    def _forward_alg(self, feats):
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
        return alpha

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1, device=self.device)
        tags = torch.cat(
            [torch.tensor([self.START_TAG], dtype=torch.long, device=self.device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.STOP_TAG, tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000., device=self.device)
        init_vvars[0][self.START_TAG] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
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
        return path_score, best_path

    def likelihood(self, packed_feats, packed_tags):
        # 为了Batch化操作进行简单的修改
        padded_feats, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_feats, batch_first=True, padding_value=0.0)
        padded_tags, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_tags, batch_first=True, padding_value=0.0)
        batch_size = len(seq_lengths)
        scores = torch.full((batch_size, ), 0.0, device=self.device)

        for idx in batch_size:
            feats = padded_feats[idx][:seq_lengths[idx]]
            tags = padded_tags[idx][:seq_lengths[idx]].to(self.device)
            score = self._score_sentence(feats, tags)
            scores[idx] = score
        return scores.view(-1)

    def decode(self, packed_feats):
        # 为了Batch化操作进行简单的修改
        padded_feats, feat_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_feats, batch_first=True, padding_value=0.0)
        batch_size = len(feat_lengths)
        path_scores = []
        best_paths = []

        for idx in range(batch_size):
            feats = padded_feats[idx][:feat_lengths[idx]]
            path_score, best_path = self._viterbi_decode(feats)
            path_scores.append(path_score)
            best_paths.append(torch.tensor(best_path))
        packed_paths = torch.nn.utils.rnn.pack_sequence(best_paths)
        return path_scores, packed_paths

    def neg_log_likelihood(self, packed_feats, packed_tags):
        # 为了Batch化操作进行简单的修改，注意这里的score都是长度为batch_size的向量
        padded_feats, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_feats, batch_first=True, padding_value=0.0)
        padded_tags, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_tags, batch_first=True, padding_value=0.0)
        batch_size = len(seq_lengths)
        forward_score = torch.full((batch_size, ), 0.0, device=self.device)
        gold_score = torch.full((batch_size, ), 0.0, device=self.device)

        for idx in range(batch_size):
            feats = padded_feats[idx][:seq_lengths[idx]]
            tags = padded_tags[idx][:seq_lengths[idx]].to(self.device)
            forward_score[idx] = self._forward_alg(feats)
            gold_score[idx] = self._score_sentence(feats, tags)
        return torch.sum(forward_score - gold_score) / batch_size

class CRF_layer(nn.Module):
    '''
    CRF model
    decode with Viterbi Algorithm
    '''
    def __init__(self, opt, num_tags):
        super().__init__()
        self.opt = opt
        self.device = self.opt.device
        self.batch_size = self.opt.batch_size
        self.tagset_size = self.opt.tagset_size

        self.num_tags = num_tags
        self.st_transitions = nn.Parameter(torch.empty(self.tagset_size, device=self.device))
        self.ed_transitions = nn.Parameter(torch.empty(self.tagset_size, device=self.device))
        self.transitions = nn.Parameter(torch.empty(self.tagset_size, self.tagset_size, device=self.device))

        # uniform initialization; can be adapted to randn
        nn.init.uniform_(self.st_transitions, -0.1, 0.1)
        nn.init.uniform_(self.ed_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        # self.transition = torch.nn.init.normal(nn.Parameter(torch.randn(num_tags, num_tags + 1)), -1, 0.1)

    def crf_forward(self, emission_scores, seq_tags, is_reduce = True, seq_masks = None):
        '''
        Forward computation
        compute log likelihood between tags and score

        Arguments
        emission_scores: torch.Tensor (seq_len, batch_size, num_tags) from biLSTM
        seq_tags: torch.LongTensor (seq_len, batch_size)
        is_reduce: bool (a flag about summing up log_likelihood over batch)
        seq_masks: torch.ByteTensor (seq_len, batch_size)

        Returns
        log_likelihood: torch.Tensor (1) for is_reduce==True (batch_size) for False
        '''
        if seq_masks is None:
            seq_masks = torch.ones_likes(seq_tags, dtype=torch.uint8)

        numerator = self._compute_likelihood_joint(emission_scores, seq_tags, seq_masks)
        denominator = self._compute_likelihood_log((emission_scores, seq_masks))
        likelihood = numerator - denominator
        if is_reduce:
            return torch.sum(likelihood)
        else:
            return likelihood

    def decode(self, emission_scores, seq_masks):
        '''
        decode sequence of tags with Viterbi algorithm

        Arguments
        emission_scores: torch.Tensor (seq_len, batch_size, num_tags)
        seq_masks: torch.ByteTensor (seq_len, batch_size)

        Returns
        sequence of best tags over batch: list[list]
        '''

        return self._viterbi_decode(emission_scores, seq_masks)

    def _compute_likelihood_joint(self, emission_scores, seq_tags, seq_masks):
        '''
        compute numerator part of likelihood

        params
        emission_scores: torch.Tensor (seq_len, batch_size, num_tags)
        seq_tags: torch.LongTensor (seq_len, batch_size)
        seq_masks: torch.ByteTensor (seq_len, batch_size)

        returns
        numerator part of likelihood: list
        '''

        seq_len = emission_scores.size(0)
        seq_masks = seq_masks.float()
        joint_lh = self.st_transitions[seq_tags[0]]

        for i in range(seq_len - 1):
            temp_tag = seq_tags[i]
            next_tag = seq_tags[i+1]
            joint_lh += emission_scores[i].gather(1, temp_tag.view(-1, 1)).squeeze(1) * seq_masks[i]
            transition_score = self.transitions[temp_tag, next_tag]
            joint_lh += transition_score * seq_masks[i+1]

        last_tag_per = seq_masks.long().sum(0) - 1
        last_tags = seq_tags.gather(0, last_tag_per.view(1, -1)).squeeze(0)

        joint_lh += self.ed_transitions[last_tags]
        joint_lh += emission_scores[-1].gather(1, last_tags.view(-1, 1)).squeeze(1) * seq_masks[-1]

        return joint_lh

    def _compute_likelihood_log(self, emission_scores, seq_masks):
        '''
        compute log part of likelihood
        '''

        seq_len = emission_scores.size(0)
        seq_masks = seq_masks.float()
        log_lh = self.st_transitions.view(1, -1) + emission_scores[0]

        for i in range(1, seq_len):
            broadcast_log_lh = log_lh.unsqueeze(2)
            broadcast_transition = self.transitions.unsqueeze(0)
            broadcast_emissions = emission_scores[i].unsqueeze(1)
            total_broadcast = broadcast_log_lh + broadcast_transition + broadcast_emissions
            total_broadcast = torch.logsumexp(total_broadcast, 1)
            log_lh = total_broadcast * seq_masks[i].unsqueeze(1) + log_lh * (1.-seq_masks[i]).unsqueeze(1)

        log_lh += self.ed_transitions.view(1, -1)
        return torch.logsumexp(log_lh, 1)



    def _viterbi_decode(self, emission_scores, seq_masks):
        '''
        apply Viterbi Algorithm

        returns
        sequence of best tags: list[list]
        '''
        best_tags_list = []

        seq_len = emission_scores.size(0)
        batch_size = emission_scores.size(1)
        lens_per = seq_masks.long().sum(dim=0)

        viterbi_score = []
        viterbi_path = []
        viterbi_score.append(self.st_transitions + emission_scores[0])

        for i in range(1, seq_len):
            broadcast_score = viterbi_score[i - 1].view(batch_size, -1, 1)
            broadcast_emission = emission_scores[i].view(batch_size, 1, -1)
            total_broadcast = broadcast_score + self.transitions + broadcast_emission
            best_score, best_path = total_broadcast.max(1)
            viterbi_score.append(best_score)
            viterbi_path.append(best_path)

        for i in range(batch_size):
            seq_end = lens_per[i] - 1
            _, best_last_tag = (viterbi_score[seq_end][i] + self.ed_transitions).max(0)
            best_tags = [best_last_tag.item()]

            for path in reversed(viterbi_path[:lens_per[i] - 1]):
                best_last_tag = path[i][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
