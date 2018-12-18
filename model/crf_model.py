# coding=utf-8
from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np

class CRF_layer(nn.Module):
    '''
    CRF model
    decode with Viterbi Algorithm
    '''
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.st_transitions = nn.Parameter(torch.empty(num_tags))
        self.ed_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

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





    def crf_decode(self, emission_scores, seq_masks):
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
