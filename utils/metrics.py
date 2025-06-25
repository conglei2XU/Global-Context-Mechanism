import pdb

import collections
from collections import Counter, defaultdict
from typing import Iterable, List

import numpy as np
import torch

class Span:
    """
    A class of `Span` where we use it during evaluation.
    We construct spans for the convenience of evaluation.
    """
    def __init__(self, left: int, right: int, type: str):
        """
        A span compose of left, right (inclusive) and its entity label.
        :param left:
        :param right: inclusive.
        :param type:
        """
        self.left = left
        self.right = right
        self.type = type

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right and self.type == other.type

    def __hash__(self):
        return hash((self.left, self.right, self.type))


class SEQMetric:
    def __init__(self, id2token, use_crf=False):
        self.id2token = id2token
        self.gold = []
        self.pred = []
        self.right = []
        self.use_crf = use_crf
        if isinstance(id2token, collections.abc.Mapping):
            self.mapping = True
        else:
            self.mapping = False


class NERMetric(SEQMetric):
    def __init__(self, id2token, use_crf=False):
        super(NERMetric, self).__init__(id2token, use_crf)

    def get_entity(self, labels, is_gold=True, gold=None, ignore_index=-100, device='cpu'):
        """
        prediction: numpy[batch_size, max_len] or tensor[batch_size, max_len]
        """
        chunks = []
        chunk = [-1, -1, -1]
        if device != 'cpu':
            labels = np.array(labels.cpu().reshape(-1, ))
        else:
            if not isinstance(labels, List):
                labels = labels.detach().cpu().numpy().reshape(-1, )
        # length = labels.shape[0]
        if is_gold:
            if not self.use_crf:
                indices = labels != -100
                labels = labels[indices]
                if not self.mapping:
                    labels = list(map(lambda x: self.id2token.id_to_token(x), labels))
                else:
                    labels = list(map(lambda x: self.id2token[x], labels))
            else:
                all_labels = []
                for idx, i in enumerate(labels):
                    if not self.mapping:
                        labels_ = list(map(lambda x: self.id2token.id_to_token(x), i))
                    else:
                        labels_ = list(map(lambda x: self.id2token[x], i))
                    all_labels.extend(labels_)
                labels = all_labels
        else:
            if not self.use_crf:
                gold = np.array(gold.cpu().reshape(-1, ))
                indices = gold != ignore_index
                labels = labels[indices]
                if not self.mapping:
                    labels = list(map(lambda x: self.id2token.id_to_token(x), labels))
                else:
                    labels = list(map(lambda x: self.id2token[x], labels))
            else:
                all_labels = []
                for idx, i in enumerate(labels):
                    if not self.mapping:
                        labels_ = list(map(lambda x: self.id2token.id_to_token(x), i))
                    else:
                        labels_ = list(map(lambda x: self.id2token[x], i))
                    all_labels.extend(labels_)
                labels = all_labels
        length = len(labels)
        for idx, value in enumerate(labels):
            if value.startswith('B-'):
                if chunk[0] != -1:
                    chunks.append(chunk)
                category = value.split('-')[-1]
                chunk = [category, idx, idx]

                if idx == length - 1:
                    chunks.append(chunk)
            elif value.startswith('I-'):
                category = value.split('-')[-1]
                pre_category = chunk[0]
                if category == pre_category:
                    chunk[-1] = idx
                else:
                    if isinstance(pre_category, str):
                        chunks.append(chunk)
                        chunk = [-1, -1, -1]
                if idx == length - 1:
                    chunks.append(chunk)

            else:
                if chunk[1] != -1:
                    chunks.append(chunk)
                    chunk = [-1, -1, -1]

        return chunks

    def calculate_f1(self, is_fined_grain=True):
        """
        :param is_fined_grain: denoted whether calculate fined grain f1 value
        :return:
        """
        names = ['precision', 'recall', 'f1']
        pred_num, right_num, gold_num = map(len, (self.pred, self.right, self.gold))
        # precision = right_num / gold_num if right_num else 0
        # recall = right_num / gold_num if gold_num else 0
        precision, recall, f1_value = f1_(pred_num, right_num, gold_num)
        f1 = dict(zip(names, [precision, recall, f1_value]))
        fined_grain_f1 = {}
        if is_fined_grain:
            pred_counter = Counter([i[0] for i in self.pred])
            right_counter = Counter(i[0] for i in self.right)
            gold_counter = Counter(i[0] for i in self.gold)
            for category, gold_num in gold_counter.items():
                pred_num = pred_counter.get(category, 0)
                right_num = right_counter.get(category, 0)
                precision, recall, f1_value = f1_(pred_num, right_num, gold_num)
                fined_grain_f1[category] = dict(zip(names, [precision, recall, f1_value]))
        return f1, fined_grain_f1

    def get_entity_batch(self, labels, gold=None, ignore_index=-100, device='cpu'):
        """
        prediction: numpy[batch_size, max_len] or tensor[batch_size, max_len]
        """

        if device != 'cpu':
            labels = np.array(labels.cpu().numpy())
            gold = gold.detach().numpy()
        else:
            if not isinstance(labels, List):
                labels = labels.detach().cpu().numpy()
                gold = gold.detach().cpu().numpy()
        batch_size = labels.shape[0]
        all_labels = []
        for i in range(batch_size):
            indices = gold[i] != ignore_index
            labels_seq = labels[i][indices]
            if not self.mapping:
                labels_ = list(map(lambda x: self.id2token.id_to_token(x), labels_seq))
            else:
                labels_ = list(map(lambda x: self.id2token[x], labels_seq))
            all_labels.append(labels_)
        return all_labels

    def __call__(self, prediction: Iterable, gold: torch.Tensor, ignore_index=-100):
        """

        :param prediction:
        :param gold: tensor[batch_size, max_len] or list[batch_size, max_len]
        :return:
        """
        pred_entities = self.get_entity(prediction, is_gold=False, gold=gold, ignore_index=ignore_index)
        gold_entities = self.get_entity(gold, ignore_index=ignore_index)
        self.pred.extend(pred_entities)
        self.gold.extend(gold_entities)
        self.right.extend([items for items in pred_entities if items in gold_entities])


class POSMetric(SEQMetric):
    def __init__(self, id2token, use_crf=False):
        super(POSMetric, self).__init__(id2token, use_crf=use_crf)

    def get_entity(self, labels, is_gold=True, gold=None, ignore_index=-100, device='cpu'):
        """
        prediction: numpy[batch_size, max_len] or tensor[batch_size, max_len]
        """
        if device != 'cpu':
            labels = np.array(labels.cpu().reshape(-1, ))
        else:
            if not isinstance(labels, List):
                labels = labels.detach().cpu().numpy().reshape(-1, )
        # length = labels.shape[0]
        if is_gold:
            if not self.use_crf:
                indices = labels != -100
                labels = labels[indices]
                if not self.mapping:
                    labels = list(map(lambda x: self.id2token.id_to_token(x), labels))
                else:
                    labels = list(map(lambda x: self.id2token[x], labels))
            else:
                all_labels = []
                for idx, i in enumerate(labels):
                    if not self.mapping:
                        labels_ = list(map(lambda x: self.id2token.id_to_token(x), i))
                    else:
                        labels_ = list(map(lambda x: self.id2token[x], i))
                    all_labels.extend(labels_)
                labels = all_labels
        else:
            if not self.use_crf:
                gold = np.array(gold.cpu().reshape(-1, ))
                indices = gold != ignore_index
                labels = labels[indices]
                if not self.mapping:
                    labels = list(map(lambda x: self.id2token.id_to_token(x), labels))
                else:
                    labels = list(map(lambda x: self.id2token[x], labels))
            else:
                all_labels = []
                for idx, i in enumerate(labels):
                    if not self.mapping:
                        labels_ = list(map(lambda x: self.id2token.id_to_token(x), i))
                    else:
                        labels_ = list(map(lambda x: self.id2token[x], i))
                    all_labels.extend(labels_)
                labels = all_labels

        return labels

    def calculate_f1(self, is_fined_grain=True):
        """
        :param is_fined_grain: denoted whether calculate fined grain f1 value
        :return:
        """
        names = ['precision', 'recall', 'f1']
        pred_num, right_num, gold_num = map(len, (self.pred, self.right, self.gold))
        # precision = right_num / gold_num if right_num else 0
        # recall = right_num / gold_num if gold_num else 0
        precision, recall, f1_value = f1_(pred_num, right_num, gold_num)
        f1 = dict(zip(names, [precision, recall, f1_value]))
        fined_grain_f1 = {}
        if is_fined_grain:
            pred_counter = Counter(self.pred)
            right_counter = Counter(self.right)
            gold_counter = Counter(self.gold)
            for category, gold_num in gold_counter.items():
                pred_num = pred_counter.get(category, 0)
                right_num = right_counter.get(category, 0)
                precision, recall, f1_value = f1_(pred_num, right_num, gold_num)
                fined_grain_f1[category] = dict(zip(names, [precision, recall, f1_value]))
        return f1, fined_grain_f1

    def __call__(self, prediction: Iterable, gold: torch.Tensor, ignore_index=-100):
        """

        :param prediction:
        :param gold: tensor[batch_size, max_len] or list[batch_size, max_len]
        :return:
        """
        pred_entities = self.get_entity(prediction, is_gold=False, gold=gold, ignore_index=ignore_index)
        gold_entities = self.get_entity(gold, ignore_index=ignore_index)
        right = []
        if len(pred_entities) == len(gold_entities):
            for pred, gold in zip(pred_entities, gold_entities):
                if pred == gold:
                    right.append(pred)
        else:
            raise ValueError(
                'pred length: %d do not match the length of gold %d' % (len(pred_entities), len(gold_entities)))
        self.right.extend(right)
        self.pred.extend(pred_entities)
        self.gold.extend(gold_entities)


class NERMetricPretrain(SEQMetric):
    """
    NER metric for pretrained model
    """
    def __init__(self, id2token, use_crf=False):
        self.p_dict, self.total_predict_dict, self.total_entity_dict = (
            Counter(), Counter(), Counter())
        super(NERMetricPretrain, self).__init__(id2token, use_crf)

    def get_entity(self, labels, gold, mask):
        """
        :param labels: torch.Tensor (batch_size, max_seq_len)
        :param gold: torch.Tensor  (batch_size, max_seq_len)
        :param mask: (batch_size) or (batch_size, max_seq_len)
        in CRF mode mask equals to sentence length (batch_size)
        """
        batch_p_dict = defaultdict(int)
        batch_total_entity_dict = defaultdict(int)
        batch_total_predict_dict = defaultdict(int)
        gold_batch_labels, pred_batch_labels = [], []
        if gold.device.type != 'cpu':
            gold = gold.cpu()
            mask = mask.cpu()
            labels = labels.cpu()
        if self.use_crf:
            word_seq_lens = mask.tolist()
            for idx in range(len(labels)):
                length = word_seq_lens[idx]
                output = gold[idx][:length].tolist()
                prediction = labels[idx][:length].tolist()
                prediction = prediction[::-1]
                gold_batch_labels.extend(output)
                pred_batch_labels.extend(prediction)
        else:
            gold_batch_labels = np.asarray(torch.masked_select(gold, mask.bool()))
            pred_batch_labels = np.asarray(torch.masked_select(labels, mask.bool()))
        if gold is not None:
            # convert token to ids
            if not self.mapping:
                gold_labels_word = list(map(lambda x: self.id2token.id_to_token(x), gold_batch_labels))
                pred_labels_word = list(map(lambda x: self.id2token.id_to_token(x), pred_batch_labels))
            else:
                gold_labels_word = list(map(lambda x: self.id2token[x], gold_batch_labels))
                pred_labels_word = list(map(lambda x: self.id2token[x], pred_batch_labels))
            # flatten batch labels in to a list

            assert len(pred_labels_word) == len(gold_labels_word), \
                (
                    f"pred label tokens {len(pred_labels_word)} dosen't same with number of gold label {len(gold_labels_word)}")
        # converting ids to tokens for pred label
        # get label indices by gold label
        # def get_chunks(batch_labels):
        #     chunks = []
        #     chunk = [-1, -1, -1]
        #     for idx, value in enumerate(batch_labels):
        #         # pdb.set_trace()
        #         if value.startswith('B-'):
        #             if chunk[0] != -1:
        #                 chunks.append(chunk)
        #             category = value.split('-')[-1]
        #             chunk = [category, idx, idx]
        #             if idx == total_tokens - 1:
        #                 chunks.append(chunk)
        #         elif value.startswith('I-'):
        #             category = value.split('-')[-1]
        #             pre_category = chunk[0]
        #             if category == pre_category:
        #                 chunk[-1] = idx
        #             else:
        #                 if isinstance(pre_category, str):
        #                     chunks.append(chunk)
        #                     chunk = [-1, -1, -1]
        #         else:
        #             if chunk[1] != -1:
        #                 chunks.append(chunk)
        #                 chunk = [-1, -1, -1]
        #     return chunks
        def get_spans(label_list):
            entity_dict = defaultdict(int)
            output_spans = set()
            chunk = [-1, -1, -1]
            for i in range(len(label_list)):
                if label_list[i].startswith("B-"):
                    if chunk[0] != -1:
                        output_spans.add(Span(chunk[0], chunk[1], chunk[2]))
                        entity_dict[chunk[2]] += 1
                    category = label_list[i].split('-')[-1]
                    chunk = [i, i, category]

                elif label_list[i].startswith("I-"):
                    category = label_list[i].split('-')[-1]
                    pre_category = chunk[2]
                    if category == pre_category:
                        chunk[1] = i
                    else:
                        if isinstance(pre_category, str):
                            output_spans.add(Span(chunk[0], chunk[1], chunk[2]))
                            entity_dict[chunk[2]] += 1
                            chunk = [-1, -1, -1]
                else:
                    if chunk[1] != -1:
                        output_spans.add(Span(chunk[0], chunk[1], chunk[2]))
                        entity_dict[chunk[2]] += 1
                        chunk = [-1, -1, -1]
            if chunk[2] != -1:
                output_spans.add(Span(chunk[0], chunk[1], chunk[2]))
                entity_dict[chunk[2]] += 1
            return output_spans, entity_dict
        gold_spans, gold_dict = get_spans(gold_labels_word)
        pred_spans, pred_dict = get_spans(pred_labels_word)
        for entity, entity_count in gold_dict.items():
            batch_total_entity_dict[entity] += entity_count
        for entity, entity_count in pred_dict.items():
            batch_total_predict_dict[entity] += entity_count
        correct_spans = pred_spans.intersection(gold_spans)
        for span in correct_spans:
            batch_p_dict[span.type] += 1
        # pdb.set_trace()
        return Counter(batch_p_dict), Counter(batch_total_predict_dict), Counter(batch_total_entity_dict)
        # pred_chunks = get_chunks(pred_labels_word)
        # gold_chunks = get_chunks(gold_labels_word)
        # return pred_chunks, gold_chunks

    def calculate_f1(self, is_fined_grain=True):
        """
        :param is_fined_grain: denoted whether calculate fined grain f1 value
        :return:
        """
        names = ['precision', 'recall', 'f1']
        # pdb.set_trace()
        pred_num, right_num, gold_num = map(len, (self.pred, self.right, self.gold))
        # precision = right_num / gold_num if right_num else 0
        # recall = right_num / gold_num if gold_num else 0

        fined_grain_f1 = {}
        if is_fined_grain:
            for key in self.total_entity_dict:
                precision_key, recall_key, fscore_key = f1_(self.total_predict_dict[key],
                                                            self.p_dict[key],
                                                            self.total_entity_dict[key])
                fined_grain_f1[key] = dict(zip(names, [precision_key, recall_key, fscore_key]))

        total_p = sum(list(self.p_dict.values()))
        total_predict = sum(list(self.total_predict_dict.values()))
        total_entity = sum(list(self.total_entity_dict.values()))
        precision, recall, f1_value = f1_(total_predict, total_p, total_entity)
        f1 = dict(zip(names, [precision, recall, f1_value]))

        return f1, fined_grain_f1

    def get_entity_batch(self, labels, gold=None, ignore_index=-100, device='cpu'):
        """
        prediction: numpy[batch_size, max_len] or tensor[batch_size, max_len]
        """

        if device != 'cpu':
            labels = np.array(labels.cpu().numpy())
            gold = gold.detach().numpy()
        else:
            if not isinstance(labels, List):
                labels = labels.detach().cpu().numpy()
                gold = gold.detach().cpu().numpy()
        batch_size = labels.shape[0]
        all_labels = []
        for i in range(batch_size):
            indices = gold[i] != ignore_index
            labels_seq = labels[i][indices]
            if not self.mapping:
                labels_ = list(map(lambda x: self.id2token.id_to_token(x), labels_seq))
            else:
                labels_ = list(map(lambda x: self.id2token[x], labels_seq))
            all_labels.append(labels_)
        return all_labels

    def __call__(self, prediction: Iterable, gold: torch.Tensor, mask: torch.Tensor):
        """

        :param prediction:
        :param gold: tensor[batch_size, max_len] or list[batch_size, max_len]
        :return:
        """
        batch_p, batch_predict, batch_total = self.get_entity(prediction, gold=gold, mask=mask)
        self.p_dict += batch_p
        self.total_predict_dict += batch_predict
        self.total_entity_dict += batch_total



def f1_(pred_num, right_num, gold_num):
    precision = right_num / pred_num if pred_num else 0
    recall = right_num / gold_num if gold_num else 0
    f1_value = (2 * precision * recall) / (precision + recall) if precision and recall else 0.
    return precision, recall, f1_value

# pred_ = [('你', 1, 2)]
# gold_ = [('你', 1, 2), ('我', 1, 2)]
# metric = SegMetric()
# metric(pred_, gold_)
# f1, fined_grain = metric.calculate_f1()
# print(f1, fined_grain)
