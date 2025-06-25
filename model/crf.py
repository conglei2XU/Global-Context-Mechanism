import pdb
from typing import Dict, List
from typing import Tuple

import torch.nn as nn
import torch
from utils.constants import START_TAG, STOP_TAG, PAD

torch.autograd.set_detect_anomaly(True)
CUDA_LAUNCH_BLOCKING = 1.


def log_sum_exp_pytorch(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize * from_label * to_label].
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0], 1, vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))


def log_sum_exp(x):
    """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    max_score = x.max(-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()


IMPOSSIBLE = -1e4


class CRF(nn.Module):
    """General CRF module.
    The CRF module contain a inner Linear Layer which transform the input from features space to tag space.

    :param in_features: number of features for the input
    :param num_tag: number of tags. DO NOT include START, STOP tags, they are included internal.
    """

    def __init__(self, in_features, num_tags):
        super(CRF, self).__init__()

        self.num_tags = num_tags + 2
        self.start_idx = self.num_tags - 2
        self.stop_idx = self.num_tags - 1

        self.fc = nn.Linear(in_features, self.num_tags)

        # transition factor, Tij mean transition from j to i
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags), requires_grad=True)
        self.transitions.data[self.start_idx, :] = IMPOSSIBLE
        self.transitions.data[:, self.stop_idx] = IMPOSSIBLE

    def forward(self, features, masks):
        """decode tags

        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        features = self.fc(features)
        return self.__viterbi_decode(features, masks[:, :features.size(1)].float())

    def loss(self, features, ys, masks):
        """negative log likelihood loss
        B: batch size, L: sequence length, D: dimension

        :param features: [B, L, D]
        :param ys: tags, [B, L]
        :param masks: masks for padding, [B, L]
        :return: loss
        """
        features = self.fc(features)

        L = features.size(1)
        masks_ = masks[:, :L].float()

        forward_score = self.__forward_algorithm(features, masks_)
        gold_score = self.__score_sentence(features, ys[:, :L].long(), masks_)
        loss = (forward_score - gold_score).mean()
        return loss

    def __score_sentence(self, features, tags, masks):
        """Gives the score of a provided tag sequence

        :param features: [B, L, C]
        :param tags: [B, L]
        :param masks: [B, L]
        :return: [B] score in the log space
        """
        B, L, C = features.shape

        # emission score
        # (batch_size, seq_len, 1)
        emit_scores = features.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1)

        # transition score
        start_tag = torch.full((B, 1), self.start_idx, dtype=torch.long, device=tags.device)
        tags = torch.cat([start_tag, tags], dim=1)  # [B, L+1]
        # select pair (current_tags, next_tag) return (batch_size, seq_len -1, 1)
        # tags[:, :-1] select tags from 0 to the second last
        # tag[:, 1:] select tags form 1 to the last

        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]]

        # last transition score to STOP tag
        last_tag = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)  # [B]

        last_score = self.transitions[self.stop_idx, last_tag]
        # remove the emit_scores of the start tag
        # [(b, seq_len ) + (b, seq_len) ] * (b, seq_len)]
        score = ((trans_scores + emit_scores) * masks).sum(1) + last_score
        return score

    def __viterbi_decode(self, features, masks):
        """decode to tags using viterbi algorithm

        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        B, L, C = features.shape

        bps = torch.zeros(B, L, C, dtype=torch.long, device=features.device)  # back pointers

        # Initialize the viterbi variables in log space
        max_score = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        max_score[:, self.start_idx] = 0

        for t in range(L):
            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            emit_score_t = features[:, t]  # [B, C]

            # [B, 1, C] + [C, C]
            acc_score_t = max_score.unsqueeze(1) + self.transitions  # [B, C, C]
            # store the best path to current label list 0, 1, ..., c
            acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
            acc_score_t += emit_score_t
            max_score = acc_score_t * mask_t + max_score * (1 - mask_t)  # max_score or acc_score_t

        # Transition to STOP_TAG
        max_score += self.transitions[self.stop_idx]
        best_score, best_tag = max_score.max(dim=-1)

        # Follow the back pointers to decode the best path.
        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(B):
            best_tag_b = best_tag[b].item()
            seq_len = int(masks[b, :].sum().item())

            best_path = [best_tag_b]
            for bps_t in reversed(bps[b, :seq_len]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            # drop the last tag and reverse the left
            best_paths.append(best_path[-2::-1])

        return best_score, best_paths

    def __forward_algorithm(self, features, masks):
        """calculate the partition function with forward algorithm.
        TRICK: log_sum_exp([x1, x2, x3, x4, ...]) = log_sum_exp([log_sum_exp([x1, x2]), log_sum_exp([x3, x4]), ...])

        :param features: features. [B, L, C]
        :param masks: [B, L] masks
        :return:    [B], score in the log space
        """
        B, L, C = features.shape

        scores = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        scores[:, self.start_idx] = 0.
        trans = self.transitions.unsqueeze(0)  # [1, C, C]

        # Iterate through the sentence
        for t in range(L):
            emit_score_t = features[:, t].unsqueeze(2)  # [B, C, 1]
            score_t = scores.unsqueeze(1) + trans + emit_score_t  # [B, 1, C] + [1, C, C] + [B, C, 1] => [B, C, C]
            score_t = log_sum_exp(score_t)  # [B, C]

            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            scores = score_t * mask_t + scores * (1 - mask_t)
        scores = log_sum_exp(scores + self.transitions[self.stop_idx])
        return scores


class CRFPretrained(nn.Module):

    def __init__(self, label_size: int, label2idx: Dict[str, int], add_iobes_constraint: bool = False,
                 idx2labels: List[str] = None):
        super(CRFPretrained, self).__init__()

        self.label_size = label_size

        self.label2idx = label2idx
        self.idx2labels = idx2labels
        self.start_idx = self.label2idx[START_TAG]
        self.end_idx = self.label2idx[STOP_TAG]
        self.pad_idx = self.label2idx[PAD]

        # initialize the following transition (anything never cannot -> start. end never  cannot-> anything. Same thing for the padding label)
        init_transition = torch.randn(self.label_size, self.label_size)
        init_transition[:, self.start_idx] = -10000.0
        init_transition[self.end_idx, :] = -10000.0
        init_transition[:, self.pad_idx] = -10000.0
        init_transition[self.pad_idx, :] = -10000.0
        if add_iobes_constraint:
            self.add_constraint_for_iobes(init_transition)

        self.transition = nn.Parameter(init_transition)

    def add_constraint_for_iobes(self, transition: torch.Tensor):
        print("[Info] Adding IOBES constraints")
        ## add constraint:
        for prev_label in self.idx2labels:
            if prev_label == START_TAG or prev_label == STOP_TAG or prev_label == PAD:
                continue
            for next_label in self.idx2labels:
                if next_label == START_TAG or next_label == STOP_TAG or next_label == PAD:
                    continue
                if prev_label == "O" and (next_label.startswith("I-") or next_label.startswith("E-")):
                    transition[self.label2idx[prev_label], self.label2idx[next_label]] = -10000.0
                if prev_label.startswith("B-") or prev_label.startswith("I-"):
                    if next_label.startswith("O") or next_label.startswith("B-") or next_label.startswith("S-"):
                        transition[self.label2idx[prev_label], self.label2idx[next_label]] = -10000.0
                    elif prev_label[2:] != next_label[2:]:
                        transition[self.label2idx[prev_label], self.label2idx[next_label]] = -10000.0
                if prev_label.startswith("S-") or prev_label.startswith("E-"):
                    if next_label.startswith("I-") or next_label.startswith("E-"):
                        transition[self.label2idx[prev_label], self.label2idx[next_label]] = -10000.0
        ##constraint for start and end
        for label in self.idx2labels:
            if label.startswith("I-") or label.startswith("E-"):
                transition[self.start_idx, self.label2idx[label]] = -10000.0
            if label.startswith("I-") or label.startswith("B-"):
                transition[self.label2idx[label], self.end_idx] = -10000.0

    def forward(self, lstm_scores, word_seq_lens, tags, mask):
        """
        Calculate the negative log-likelihood
        :param lstm_scores:
        :param word_seq_lens:
        :param tags:
        :param mask:
        :return:
        """
        all_scores = self.calculate_all_scores(lstm_scores=lstm_scores)
        unlabed_score = self.forward_unlabeled(all_scores, word_seq_lens)
        labeled_score = self.forward_labeled(all_scores, word_seq_lens, tags, mask)
        return unlabed_score, labeled_score

    def get_marginal_score(self, lstm_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        marginal = self.forward_backward(lstm_scores=lstm_scores, word_seq_lens=word_seq_lens)
        return marginal

    def forward_unlabeled(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Calculate the scores with the forward algorithm. Basically calculating the normalization term
        :param all_scores: (batch_size x max_seq_len x num_labels x num_labels) from (lstm scores + transition scores).
        :param word_seq_lens: (batch_size)
        :return: The score for all the possible structures.
        """
        batch_size = all_scores.size(0)
        seq_len = all_scores.size(1)
        dev_num = all_scores.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        alpha = torch.zeros(batch_size, seq_len, self.label_size, device=curr_dev)

        alpha[:, 0, :] = all_scores[:, 0, self.start_idx,
                         :]  ## the first position of all labels = (the transition from start - > all labels) + current emission.

        for word_idx in range(1, seq_len):
            ## batch_size, self.label_size, self.label_size
            before_log_sum_exp = alpha[:, word_idx - 1, :].view(batch_size, self.label_size, 1).expand(batch_size,
                                                                                                       self.label_size,
                                                                                                       self.label_size) + all_scores[
                                                                                                                          :,
                                                                                                                          word_idx,
                                                                                                                          :,
                                                                                                                          :]
            # alpha[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)
            alpha[:, word_idx, :] = torch.logsumexp(before_log_sum_exp, dim=1)

        ### batch_size x label_size
        last_alpha = torch.gather(alpha, 1,
                                  word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size) - 1).view(
            batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha = log_sum_exp_pytorch(last_alpha.view(batch_size, self.label_size, 1)).view(batch_size)

        ## final score for the unlabeled network in this batch, with size: 1
        return torch.sum(last_alpha)

    def backward(self, lstm_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Backward algorithm. A benchmark implementation which is ready to use.
        :param lstm_scores: shape: (batch_size, sent_len, label_size) NOTE: the score from LSTMs, not `all_scores` (which add up the transtiion)
        :param word_seq_lens: shape: (batch_size,)
        :return: Backward variable
        """
        batch_size = lstm_scores.size(0)
        seq_len = lstm_scores.size(1)
        dev_num = lstm_scores.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        beta = torch.zeros(batch_size, seq_len, self.label_size, device=curr_dev)

        ## reverse the view of computing the score. we look from behind
        rev_score = self.transition.transpose(0, 1).view(1, 1, self.label_size, self.label_size).expand(batch_size,
                                                                                                        seq_len,
                                                                                                        self.label_size,
                                                                                                        self.label_size) + \
                    lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len,
                                                                                     self.label_size, self.label_size)

        ## The code below, reverse the score from [0 -> length]  to [length -> 0].  (NOTE: we need to avoid reversing the padding)
        perm_idx = torch.zeros(batch_size, seq_len, device=curr_dev)
        for batch_idx in range(batch_size):
            perm_idx[batch_idx][:word_seq_lens[batch_idx]] = torch.range(word_seq_lens[batch_idx] - 1, 0, -1)
        perm_idx = perm_idx.long()
        for i, length in enumerate(word_seq_lens):
            rev_score[i, :length] = rev_score[i, :length][perm_idx[i, :length]]

        ## backward operation
        beta[:, 0, :] = rev_score[:, 0, self.end_idx, :]
        for word_idx in range(1, seq_len):
            before_log_sum_exp = beta[:, word_idx - 1, :].view(batch_size, self.label_size, 1).expand(batch_size,
                                                                                                      self.label_size,
                                                                                                      self.label_size) + rev_score[
                                                                                                                         :,
                                                                                                                         word_idx,
                                                                                                                         :,
                                                                                                                         :]
            beta[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)

        ## Following code is used to check the backward beta implementation
        last_beta = torch.gather(beta, 1,
                                 word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size) - 1).view(
            batch_size, self.label_size)
        last_beta += self.transition.transpose(0, 1)[:, self.start_idx].view(1, self.label_size).expand(batch_size,
                                                                                                        self.label_size)
        last_beta = log_sum_exp_pytorch(last_beta.view(batch_size, self.label_size, 1)).view(batch_size)

        # This part if optionally, if you only use `last_beta`.
        # Otherwise, you need this to reverse back if you also need to use beta
        for i, length in enumerate(word_seq_lens):
            beta[i, :length] = beta[i, :length][perm_idx[i, :length]]

        return torch.sum(last_beta)

    def forward_backward(self, lstm_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Note: This function is not used unless you want to compute the marginal probability
        Forward-backward algorithm to compute the marginal probability (in log space)
        Basically, we follow the `backward` algorithm to obtain the backward scores.
        :param lstm_scores:   shape: (batch_size, sent_len, label_size) NOTE: the score from LSTMs, not `all_scores` (which add up the transtiion)
        :param word_seq_lens: shape: (batch_size,)
        :return: Marginal score. If you want probability, you need to use `torch.exp` to convert it into probability
        """
        batch_size = lstm_scores.size(0)
        seq_len = lstm_scores.size(1)
        dev_num = lstm_scores.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        alpha = torch.zeros(batch_size, seq_len, self.label_size, device=curr_dev)
        beta = torch.zeros(batch_size, seq_len, self.label_size, device=curr_dev)

        scores = self.transition.view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len,
                                                                                     self.label_size, self.label_size) + \
                 lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size,
                                                                                  self.label_size)
        ## reverse the view of computing the score. we look from behind
        rev_score = self.transition.transpose(0, 1).view(1, 1, self.label_size, self.label_size).expand(batch_size,
                                                                                                        seq_len,
                                                                                                        self.label_size,
                                                                                                        self.label_size) + \
                    lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len,
                                                                                     self.label_size, self.label_size)

        perm_idx = torch.zeros(batch_size, seq_len, device=curr_dev)
        for batch_idx in range(batch_size):
            perm_idx[batch_idx][:word_seq_lens[batch_idx]] = torch.range(word_seq_lens[batch_idx] - 1, 0, -1)
        perm_idx = perm_idx.long()
        for i, length in enumerate(word_seq_lens):
            rev_score[i, :length] = rev_score[i, :length][perm_idx[i, :length]]

        alpha[:, 0, :] = scores[:, 0, self.start_idx,
                         :]  ## the first position of all labels = (the transition from start - > all labels) + current emission.
        beta[:, 0, :] = rev_score[:, 0, self.end_idx, :]
        for word_idx in range(1, seq_len):
            before_log_sum_exp = alpha[:, word_idx - 1, :].view(batch_size, self.label_size, 1).expand(batch_size,
                                                                                                       self.label_size,
                                                                                                       self.label_size) + scores[
                                                                                                                          :,
                                                                                                                          word_idx,
                                                                                                                          :,
                                                                                                                          :]
            alpha[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)

            before_log_sum_exp = beta[:, word_idx - 1, :].view(batch_size, self.label_size, 1).expand(batch_size,
                                                                                                      self.label_size,
                                                                                                      self.label_size) + rev_score[
                                                                                                                         :,
                                                                                                                         word_idx,
                                                                                                                         :,
                                                                                                                         :]
            beta[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)

        ### batch_size x label_size
        last_alpha = torch.gather(alpha, 1,
                                  word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size) - 1).view(
            batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha = log_sum_exp_pytorch(last_alpha.view(batch_size, self.label_size, 1)).view(batch_size, 1, 1).expand(
            batch_size, seq_len, self.label_size)

        ## Because we need to use the beta variable later, we need to reverse back
        for i, length in enumerate(word_seq_lens):
            beta[i, :length] = beta[i, :length][perm_idx[i, :length]]

        # `alpha + beta - last_alpha` is the standard way to obtain the marginal
        # However, we have two emission scores overlap at each position, thus, we need to subtract one emission score
        return alpha + beta - last_alpha - lstm_scores

    def forward_labeled(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor, tags: torch.Tensor,
                        masks: torch.Tensor) -> torch.Tensor:
        '''
        Calculate the scores for the gold instances.
        :param all_scores: (batch, seq_len, label_size, label_size)
        :param word_seq_lens: (batch, seq_len)
        :param tags: (batch, seq_len)
        :param masks: batch, seq_len
        :return: sum of score for the gold sequences Shape: (batch_size)
        '''
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]

        ## all the scores to current labels: batch, seq_len, all_from_label?
        currentTagScores = torch.gather(all_scores, 3,
                                        tags.view(batchSize, sentLength, 1, 1).expand(batchSize, sentLength,
                                                                                      self.label_size, 1)).view(
            batchSize, -1, self.label_size)
        if sentLength != 1:
            tagTransScoresMiddle = torch.gather(currentTagScores[:, 1:, :], 2,
                                                tags[:, : sentLength - 1].view(batchSize, sentLength - 1, 1)).view(
                batchSize, -1)
        tagTransScoresBegin = currentTagScores[:, 0, self.start_idx]
        endTagIds = torch.gather(tags, 1, word_seq_lens.view(batchSize, 1) - 1)
        tagTransScoresEnd = torch.gather(
            self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size), 1,
            endTagIds).view(batchSize)
        score = torch.sum(tagTransScoresBegin) + torch.sum(tagTransScoresEnd)
        if sentLength != 1:
            score += torch.sum(tagTransScoresMiddle.masked_select(masks[:, 1:]))
        return score

    def calculate_all_scores(self, lstm_scores: torch.Tensor) -> torch.Tensor:
        """
        Calculate all scores by adding up the transition scores and emissions (from lstm).
        Basically, compute the scores for each edges between labels at adjacent positions.
        This score is later be used for forward-backward inference
        :param lstm_scores: emission scores.
        :return:
        """
        batch_size = lstm_scores.size(0)
        seq_len = lstm_scores.size(1)
        scores = self.transition.view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len,
                                                                                     self.label_size, self.label_size) + \
                 lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size,
                                                                                  self.label_size)
        return scores

    def decode(self, features, wordSeqLengths) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        all_scores = self.calculate_all_scores(features)
        bestScores, decodeIdx = self.viterbi_decode(all_scores, wordSeqLengths)
        return bestScores, decodeIdx

    def viterbi_decode(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Use viterbi to decode the instances given the scores and transition parameters
        :param all_scores: (batch_size x max_seq_len x num_labels)
        :param word_seq_lens: (batch_size)
        :return: the best scores as well as the predicted label ids.
               (batch_size) and (batch_size x max_seq_len)
        """
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]
        dev_num = all_scores.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        scoresRecord = torch.zeros([batchSize, sentLength, self.label_size], device=curr_dev)
        idxRecord = torch.zeros([batchSize, sentLength, self.label_size], dtype=torch.int64, device=curr_dev)
        mask = torch.ones_like(word_seq_lens, dtype=torch.int64, device=curr_dev)
        startIds = torch.full((batchSize, self.label_size), self.start_idx, dtype=torch.int64, device=curr_dev)
        decodeIdx = torch.LongTensor(batchSize, sentLength).to(curr_dev)

        scores = all_scores
        # scoresRecord[:, 0, :] = self.getInitAlphaWithBatchSize(batchSize).view(batchSize, self.label_size)
        scoresRecord[:, 0, :] = scores[:, 0, self.start_idx,
                                :]  ## represent the best current score from the start, is the best
        idxRecord[:, 0, :] = startIds
        for wordIdx in range(1, sentLength):
            ### scoresIdx: batch x from_label x to_label at current index.
            scoresIdx = scoresRecord[:, wordIdx - 1, :].view(batchSize, self.label_size, 1).expand(batchSize,
                                                                                                   self.label_size,
                                                                                                   self.label_size) + scores[
                                                                                                                      :,
                                                                                                                      wordIdx,
                                                                                                                      :,
                                                                                                                      :]
            idxRecord[:, wordIdx, :] = torch.argmax(scoresIdx, 1)  ## the best previous label idx to crrent labels
            scoresRecord[:, wordIdx, :] = torch.gather(scoresIdx, 1, idxRecord[:, wordIdx, :].view(batchSize, 1,
                                                                                                   self.label_size)).view(
                batchSize, self.label_size)

        lastScores = torch.gather(scoresRecord, 1,
                                  word_seq_lens.view(batchSize, 1, 1).expand(batchSize, 1, self.label_size) - 1).view(
            batchSize, self.label_size)  ##select position
        lastScores += self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size)
        decodeIdx[:, 0] = torch.argmax(lastScores, 1)
        bestScores = torch.gather(lastScores, 1, decodeIdx[:, 0].view(batchSize, 1))

        for distance2Last in range(sentLength - 1):
            lastNIdxRecord = torch.gather(idxRecord, 1, torch.where(word_seq_lens - distance2Last - 1 > 0,
                                                                    word_seq_lens - distance2Last - 1, mask).view(
                batchSize, 1, 1).expand(batchSize, 1, self.label_size)).view(batchSize, self.label_size)
            decodeIdx[:, distance2Last + 1] = torch.gather(lastNIdxRecord, 1,
                                                           decodeIdx[:, distance2Last].view(batchSize, 1)).view(
                batchSize)

        return bestScores, decodeIdx
