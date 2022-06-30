"""
REFERENCE : https://towardsdatascience.com/implementing-a-linear-chain-conditional-random-field-crf-in-pytorch-16b0b9c4b4ea

P(y|X) = \frac{exp(\sum^{l}_{k=1}U(x_{k}, y_{k}) + \sum^{l-1}_{k=1}T(y_{k}, y_{k+1}))} {Z(X)}

- emissions or unary scores (U) : scores representing how likely is yk given the input xk
- transition scores (T) : scores representing how likely is yk followed by yk+1
- partition function (Z) : normalization factor in order to get probability distribution over sequences

"""

import torch
from torch import nn


class Const:
    UNK_ID, UNK_TOKEN = 0, "<unk>"
    PAD_ID, PAD_TOKEN = 1, "<pad>"
    BOS_ID, BOS_TOKEN = 2, "<bos>"
    EOS_ID, EOS_TOKEN = 3, "<eos>"
    PAD_TAG_ID, PAD_TAG_TOKEN = 0, "<pad>"
    BOS_TAG_ID, BOS_TAG_TOKEN = 1, "<bos>"
    EOS_TAG_ID, EOS_TAG_TOKEN = 2, "<eos>"


class CRF(nn.Module):
    """
    we have to minimize LOSS FUNCTION L = -log(P(y|X)).
    In other words, we have to maximize P(y|X).
    This LOSS FUNCTION called negative log likelihood.
    """

    def __init__(self, nb_labels, bos_tag_id, eos_tag_id, pad_tag_id=None, batch_first=True):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.nb_labels = nb_labels
        self.BOS_TAG_ID = bos_tag_id
        self.EOS_TAG_ID = eos_tag_id
        self.PAD_TAG_ID = pad_tag_id
        self.batch_first = batch_first

        # transition scores
        self.transitions = nn.Parameter(torch.empty(self.nb_labels, self.nb_labels))
        self.init_weights()

    def init_weights(self):
        # initialize transitions from a random uniform distribution between -1.0 and 1.0
        nn.init.uniform_(self.transitions, -0.1, 0.1)

        # exp(-10000) tend to zero
        # no transitions allowed to beginning of sentence
        # no transitions allowed from end of sentence
        self.transitions.data[:, self.BOS_TAG_ID] = -10000.0
        self.transitions.data[self.EOS_TAG_ID, :] = -10000.0

        if self.PAD_TAG_ID is not None:
            # no transitions from padding
            self.transitions.data[self.PAD_TAG_ID, :] = -10000.0
            # no transitions to padding
            self.transitions.data[:, self.PAD_TAG_ID] = -10000.0
            # except if the end of sentence is reached
            # or we are already in a pad position
            self.transitions.data[self.PAD_TAG_ID, self.EOS_TAG_ID] = 0.0
            self.transitions.data[self.PAD_TAG_ID, self.PAD_TAG_ID] = 0.0

    def forward(self, emissions, tags, mask=None):
        """
        Compute the negative log-likelihood.
        :param emissions:
        :param tags:
        :param mask:
        :return:
        """
        nll = -self.log_likelihood(emissions, tags, mask=mask)
        return nll

    def log_likelihood(self, emissions, tags, mask=None):
        """
        Compute the probability of a sequence of tags y
        given a sequence of emissions scores X.

        Equation below is negative log likelihood.
        -log(P(y|X)) = -log(exp( \sum^{l}_{k=1}U(x_k+y_k) + \sum^{l-1}_{k=1}T(y_k,y_{k+1}) ) / Z(X))
                     = log(Z(X)) - log(exp( \sum^{l}_{k=1}U(x_k+y_k) + \sum^{l-1}_{k=1}T(y_k,y_{k+1}) ))
                     = log(Z(X)) - ( \sum^{l}_{k=1}U(x_k+y_k) + \sum^{l-1}_{k=1}T(y_k,y_{k+1}) )
                     = Zlog(X) - ( \sum^{l}_{k=1}U(x_k+y_k) + \sum^{l-1}_{k=1}T(y_k,y_{k+1}) )
                     = partition - ( scores )

        scores = sumU + sumT = e_scores + t_scores

        :param emissions: U
        if batch_first == True : (batch_size, seq_len, nb_labels)
        if batch_first == False : (seq_len, batch_size, nb_labels)
        :param tags:
        if batch_first == True : (batch_size, seq_len)
        if batch_first == False : (seq_len, batch_size)
        :param mask:
        if batch_first == True : (batch_size, seq_len)
        if batch_first == False : (seq_len, batch_size)

        :return:
        """
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)

        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float, device=self.device)

        scores = self._compute_scores(emissions, tags, mask=mask)
        partition = self._compute_log_partition(emissions, mask=mask)
        return torch.sum(scores - partition)

    def _compute_scores(self, emissions, tags, mask):
        """
        Compute the scores for a given batch of emissions with their tags.
        "scores" in torch.sum(scores - partition)

        Equation below is about scores (U + T)
        \sum^{l}_{k=1}U(x_k+y_k) + \sum^{l-1}_{k=1}T(y_k,y_{k+1})

        :param emissions: U
        if batch_first == True : (batch_size, seq_len, nb_labels)
        :param tags:
        if batch_first == True : (batch_size, seq_len)
        :param mask:
        if batch_first == True : (batch_size, seq_len)

        :return:
        """
        batch_size, seq_length = tags.shape
        scores = torch.zeros(batch_size, device=self.device)

        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(axis=1) - 1
        last_tags = tags.gather(dim=1, index=last_valid_idx.unsqueeze(1)).squeeze()

        # T
        # add the transition from BOS to the first tags for each batch
        t_scores = self.transitions[self.BOS_TAG_ID, first_tags]

        # U
        # add the [unary] emission scores for the first tags for each batch
        # for all batches, the first word, see the correspondent emissions
        # for the first tags (which is a list of ids):
        # emissions[:, 0, [tag_1, tag_2, ..., tag_nblabels]]
        e_scores = emissions[:, 0].gather(dim=1, index=first_tags.unsqueeze(1)).squeeze()

        scores += e_scores + t_scores

        for i in range(1, seq_length):

            is_valid = mask[:, i]

            previous_tags = tags[:, i-1]
            current_tags = tags[:, i]

            e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
            t_scores = self.transitions[previous_tags, current_tags]

            e_scores *= is_valid
            t_scores *= is_valid

            scores += e_scores + t_scores

        scores += self.transitions[last_tags, self.EOS_TAG_ID]

        return scores

    def _compute_log_partition(self, emissions, mask):
        """
        Compute the partition function in log-space using the forward-algorithm.

        :param emissions: U
        if batch_first == True : (batch_size, seq_len, nb_labels)
        :param mask:
        if batch_first == True : (batch_size, seq_len)

        :return:
        """
        batch_size, seq_length, nb_labels = emissions.shape

        # (1, nb_labels) + (batch_size, nb_labels) = (batch_size, nb_labels)
        alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]

        for i in range(1, seq_length):

            alpha_t = []

            for tag in range(nb_labels):

                # (batch_size, 1)
                # current tag of i'th sequence
                e_scores = emissions[:, i, tag]
                e_scores = e_scores.unsqueeze(1)

                # (1, nb_labels)
                # transition from something to current tag
                t_scores = self.transitions[:, tag]
                t_scores = t_scores.unsqueeze(0)

                # combine current scores with previous alpha
                # since alpha are in log space
                # we add them instead of multiplying
                # (batch_size, 1) + (1, nb_labels) + (batch_size, nb_labels) = (batch_size, nb_labels)
                scores = e_scores + t_scores + alphas

                # torch.logsumexp(a, 1) == torch.log(torch.sum(torch.exp(a), 1))
                alpha_t.append(torch.logsumexp(scores, dim=1))

            new_alpha = torch.stack(alpha_t).t()

            # set alphas if the mask is valid, otherwise keep the current alphas
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alpha + (1 - is_valid) * alphas

        last_transition = self.transitions[:, self.EOS_TAG_ID]
        end_scores = alphas + last_transition.unsqueeze(0)

        return torch.logsumexp(end_scores, dim=1)

    def decode(self, emissions, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float, device=self.device)

        scores, sequences = self._viterbi_decode(emissions, mask)

        return scores, sequences

    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_length, nb_labels = emissions.shape

        # (1, nb_labels) + (batch_size, nb_labels) = (batch_size, nb_labels)
        alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]

        backpointers = []

        for i in range(seq_length):

            alpha_t = []
            backpointers_t = []

            for tag in range(nb_labels):

                # (batch_size, 1)
                e_scores = emissions[:, i, tag]
                e_scores = e_scores.unsqueeze(1)

                # (1, nb_labels)
                t_scores = self.transitions[:, tag]
                t_scores = t_scores.unsqueeze(0)

                # (batch_size, 1) + (1, nb_labels) + (batch_size, nb_labels) = (batch_size, nb_labels)
                scores = e_scores + t_scores + alphas

                # (batch_size,), (batch_size)
                max_score, max_score_tag = torch.max(scores, dim=-1)  # torch.max returns values, indices

                alpha_t.append(max_score)

                backpointers_t.append(max_score_tag)

            # alpha_t = nb_labels * (batch_size,)
            # new_alphas = (batch_size, nb_labels)
            new_alphas = torch.stack(alpha_t).t()  # t() is transpose

            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas

            # backpointers_t = nb_labels * (batch_size,)
            # backpointers = seq_length * nb_labels * (batch_size,)
            backpointers.append(backpointers_t)

        # (nb_labels,)
        last_transition = self.transitions[:, self.EOS_TAG_ID]
        # (batch_size, nb_labels) + (1, nb_labels) = (batch_size, nb_labels)
        end_scores = alphas + last_transition.unsqueeze(0)

        # (batch_size,), (batch_size,)
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)

        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)

        for i in range(batch_size):

            sample_length = emission_lengths[i].item()  # 각 샘플의 시퀀스 길이
            sample_final_tag = max_final_tags[i].item()  # 각 샘플의 마지막 태그

            sample_backpointers = backpointers[: sample_length - 1]

            sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)

            best_sequences.append(sample_path)

        return max_final_scores, best_sequences

    def _find_best_path(self, sample_id, best_tag, backpointers):
        best_path = [best_tag]

        for backpointers_t in reversed(backpointers):
            best_tag = backpointers_t[best_tag][sample_id].item()
            best_path.insert(0, best_tag)

        return best_path
