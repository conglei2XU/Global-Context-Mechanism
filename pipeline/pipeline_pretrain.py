# --*-- coding: utf-8 --*--
"""
hugging face bert utils to work with glue tasks
"""
import os
import pdb
from typing import Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import BatchEncoding
from transformers import Pipeline

from utils.constants import STOP_TAG, START_TAG, PAD


class InputExample:
    """
    input structures for transformer models
    """

    def __init__(self, guid: int, text_a: str, text_b: Optional[str] = None,
                 label: Optional[str] = None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __str__(self):
        if self.text_b:
            return "text_a: {} \t text_b: {}".format(self.text_a, self.text_b)
        return "text: {} \t".format(self.text_a)


class BaseExamples:
    """
    base class to construct input example for transformer models
    """

    def __init__(self, base_path=None, read_method=None):
        self.read_method = read_method
        self.base_dir = base_path
        self.train_examples = []
        self.dev_examples = []
        self.test_examples = []

    def get_train_examples(self, data_path) -> list:
        raise NotImplementedError

    def get_dev_examples(self, data_path) -> list:
        raise NotImplementedError

    def get_test_examples(self, data_path) -> list:
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError

    def construct_examples(self, data_path):
        raise NotImplementedError


class SeqExamples(BaseExamples):
    def __init__(self, base_path=None, read_method=None):
        super(SeqExamples, self).__init__(base_path, read_method)

    def get_train_examples(self, data_path=None) -> list:
        if data_path:
            source = data_path
        else:
            source = os.path.join(self.base_dir, 'train')
        self.train_examples = self.construct_examples(source)
        return self.train_examples

    def get_dev_examples(self, data_path=None) -> list:
        if data_path:
            source = data_path
        else:
            source = os.path.join(self.base_dir, 'valid')
        self.dev_examples = self.construct_examples(source)
        return self.dev_examples

    def get_test_examples(self, data_path=None) -> list:
        if data_path:
            source = data_path
        else:
            source = os.path.join(self.base_dir, 'test')
        self.test_examples = self.construct_examples(source)
        return self.test_examples

    def get_labels(self):
        label_list = []
        for example in self.train_examples:
            for label in example.label:
                if label not in label_list:
                    label_list.append(label)
        return label_list

    def construct_examples(self, data_path):
        examples = []
        # read_method return data in (sentence[list], label[list) format
        data_generator = self.read_method(data_path)
        guid = 0
        for sentence, labels in data_generator:
            examples.append(InputExample(guid=guid, text_a=sentence, label=labels))

        return examples


def convert_examples_to_records(examples: list, tokenizer, is_split_into_words=True) -> list:
    records = []
    for example in examples:
        record = tokenizer(example.text_a, is_split_into_words=is_split_into_words)
        label_ids = []
        pre_word_id = None
        for word_id in record.word_ids():
            if word_id is None:
                label_ids.append(-100)
            elif word_id != pre_word_id:
                label_ids.append(-100)
            else:
                label_ids.append(-100)
        record['labels'] = label_ids
        records.append(record)
    return records


def handle_subword(subword_ids, pad_idx, origin_label, label2idx):
    """
    got the origin word ids of subword and remove [PAD], [CLS], [SEP] tokens
    subword_ids List, contain origin word index for each wordpiece
    pad_idx int, pad_idx for labels
    origin_label List, series label for current data
    origin_text List, list of word
    label2idx Mapping, mapping label to its corresponding idx
    """
    prev_word_idx = -1
    inference_mask = []
    orig_to_tok_index, label_ids = [], []  # store the word index in subword token list
    for i, mapped_word_idx in enumerate(subword_ids):
        """
        Note: by default, we use the first wordpiece/subword token to represent the word
        If you want to do something else (e.g., use last wordpiece to represent), modify them here.
        """
        if mapped_word_idx is None:  ## cls and sep token

            label_ids.append(pad_idx)
            inference_mask.append(0)
            continue
        elif mapped_word_idx != prev_word_idx:
            inference_mask.append(1)
            orig_to_tok_index.append(i)
            prev_word_idx = mapped_word_idx
            try:
                cur_id = label2idx[origin_label[mapped_word_idx]]
                label_ids.append(cur_id)
            except KeyError:
                print(f'Missing {origin_label[mapped_word_idx]} in {label2idx}')
                exit(1)
        else:
            label_ids.append(pad_idx)
            inference_mask.append(0)
    return orig_to_tok_index, label_ids, inference_mask


class SeqDatasetCRF(Dataset):
    def __init__(self, dataset_path, read_method, tokenizer, label2idx=None, language='EN'):
        """
        Build torch Dataset for sequence labeling :
        1. build examples from dataset,
        2. build label2idx for dataset,
        3. convert examples to features
        """
        self.source = dataset_path
        self.reader = read_method
        self.tokenizer = tokenizer
        self.examples = self._construct_examples()
        if label2idx:
            self.label2idx = label2idx
            self.idx2label = {value: key for (key, value) in label2idx.items()}
        else:
            label_list = self.get_label()
            self.label2idx = defaultdict()
            self.label2idx.default_factory = self.label2idx.__len__
            self.idx2label = {}
            # for CRF mode we need to insert START_TAG, STOP_TAG and PAD into label list
            label_list.insert(0, START_TAG)
            label_list.append(PAD)
            label_list.append(STOP_TAG)
            for label in label_list:
                idx = self.label2idx[label]
                self.idx2label[idx] = label
        self.pad_idx = self.label2idx[PAD]
        self.language = language
        self.instance = self.convert_examples_to_features(self.examples, tokenizer)

    def __len__(self):
        return len(self.instance)

    def __getitem__(self, item):
        return self.instance[item]

    def get_label(self):
        labels = []
        for example in self.examples:
            for label in example.label:
                if label not in labels:
                    labels.append(label)
        return labels

    def _construct_examples(self):
        data_generator = self.reader(self.source)
        guid = 0
        examples = []
        for sentence, labels in data_generator:
            examples.append(InputExample(guid=guid, text_a=sentence, label=labels))
        return examples

    def convert_examples_to_features(self,
                                     examples: list,
                                     tokenizer,
                                     is_split_into_words: bool = True) -> list:
        instance = []
        for example in examples:
            encoded_wordpiece = tokenizer.encode_plus(example.text_a,
                                                      is_split_into_words=is_split_into_words,
                                                      )

            subword_idx2word_idx = encoded_wordpiece.word_ids(batch_index=0)
            orig_to_tok_index, label_ids, inference_mask = handle_subword(subword_ids=subword_idx2word_idx,
                                                                          pad_idx=self.pad_idx,
                                                                          origin_label=example.label,
                                                                          label2idx=self.label2idx)
            if self.language == 'CN' and len(orig_to_tok_index) != len(example.text_a):
                orig_to_tok_index = [idx for idx in range(len(example.text_a))]
                token_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(example.text_a) + [
                    tokenizer.sep_token_id]
                attention_mask = [1] * len(token_ids)
            else:
                attention_mask = encoded_wordpiece['attention_mask']
                token_ids = encoded_wordpiece['input_ids']
            assert len(orig_to_tok_index) == len(example.text_a)
            labels = example.label
            label_ids = [self.label2idx[label] for label in labels] if labels else [-100] * len(example.text_a)
            segment_ids = [0] * len(token_ids)
            instance.append({"input_ids": token_ids,
                             "attention_mask": attention_mask,
                             "orig_to_tok_index": orig_to_tok_index,
                             "token_type_ids": segment_ids,
                             "word_seq_len": len(orig_to_tok_index),
                             "labels": label_ids})

        return instance

    def collate_fn(self, batch_features):
        word_seq_len = [len(feature["orig_to_tok_index"]) for feature in batch_features]
        max_seq_len = max(word_seq_len)
        max_wordpiece_length = max([len(feature["input_ids"]) for feature in batch_features])
        # Tokenizer will add [CLS] and [SEP] at the beginning and ending for the input sentence.
        batch_input = []
        for i, feature in enumerate(batch_features):
            padding_length = max_wordpiece_length - len(feature["input_ids"])
            input_ids = feature["input_ids"] + [self.tokenizer.pad_token_id] * padding_length
            mask = feature["attention_mask"] + [0] * padding_length
            type_ids = feature["token_type_ids"] + [self.tokenizer.pad_token_type_id] * padding_length
            padding_word_len = max_seq_len - len(feature["orig_to_tok_index"])
            orig_to_tok_index = feature["orig_to_tok_index"] + [0] * padding_word_len
            label_ids = feature["labels"] + [0] * padding_word_len
            wordpiece_len = len(input_ids)

            batch_input.append({"input_ids": input_ids,
                                "attention_mask": mask,
                                "token_type_ids": type_ids,
                                "orig_to_tok_index": orig_to_tok_index,
                                "word_seq_len": feature["word_seq_len"],
                                "labels": label_ids,
                                "wordpiece_len": wordpiece_len})
        encoded_inputs = {key: [example[key] for example in batch_input] for key in batch_input[0].keys()}
        results = BatchEncoding(encoded_inputs, tensor_type='pt')
        return results


class SeqDataset(Dataset):
    def __init__(self, dataset_path, read_method, tokenizer, label2idx=None, language='EN'):
        self.source = dataset_path
        self.reader = read_method
        self.tokenizer = tokenizer
        examples = self._construct_examples(dataset_path)
        if label2idx:
            self.label2idx = label2idx
            self.idx2label = {value: key for (key, value) in label2idx.items()}
        else:
            label_list = self.get_label(examples)
            label_list.append(PAD)
            self.label2idx = defaultdict()
            self.label2idx.default_factory = self.label2idx.__len__
            self.idx2label = {}
            for label in label_list:
                idx = self.label2idx[label]
                self.idx2label[idx] = label
        self.pad_idx = self.label2idx[PAD]
        self.language = language

        # for CRF mode we need to insert START_TAG, STOP_TAG and PAD into label list
        self.instance = self.convert_examples_to_features(examples, tokenizer)

    def __len__(self):
        return len(self.instance)

    def __getitem__(self, item):
        return self.instance[item]

    def get_label(self, examples):
        labels = []
        for example in examples:
            for label in example.label:
                if label not in labels:
                    labels.append(label)
        return labels

    def _construct_examples(self, data_path):
        data_generator = self.reader(data_path)
        guid = 0
        examples = []
        for sentence, labels in data_generator:
            examples.append(InputExample(guid=guid, text_a=sentence, label=labels))
        return examples

    def convert_examples_to_features(self,
                                     examples: list,
                                     tokenizer,
                                     is_split_into_words: bool = True) -> list:
        instance = []
        for example in examples:
            # chinese tokenizer is hanzi level tokenizer we don't need to handle subword tokens
            encoded_wordpiece = tokenizer.encode_plus(example.text_a,
                                                      is_split_into_words=is_split_into_words,
                                                      max_length=512,
                                                      truncation=True,
                                                      add_special_tokens=True
                                                      )

            subword_idx2word_idx = encoded_wordpiece.word_ids(batch_index=0)
            orig_to_tok_index, label_ids, inference_mask = handle_subword(subword_ids=subword_idx2word_idx,
                                                                          pad_idx=self.pad_idx,
                                                                          origin_label=example.label,
                                                                          label2idx=self.label2idx)
            # for not use first token [CLS] and last token [SEP]
            # label_ids = label_ids[1:-1]
            # inference_mask = inference_mask[1:-1]

            if self.language == 'CN' and len(orig_to_tok_index) != len(example.text_a):
                token_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(example.text_a) + [
                    tokenizer.sep_token_id]
                attention_mask = [1] * len(token_ids)
                label_ids = [self.pad_idx] + [self.label2idx[label] for label in example.label] + [self.pad_idx]
                # label_ids = [self.pad_idx] + [self.label2idx[label] for label in example.label] + [self.pad_idx]
                inference_mask = [0] + [1] * len(example.label) + [0]
            else:
                attention_mask = encoded_wordpiece["attention_mask"]
                token_ids = encoded_wordpiece["input_ids"]
            instance.append({"input_ids": token_ids,
                             "attention_mask": attention_mask,
                             "labels": label_ids,
                             'inference_masks': inference_mask})

        return instance

    def collate_fn(self, batch_features):
        max_wordpiece_length = max([len(feature["input_ids"]) for feature in batch_features])
        # Tokenizer will add [CLS] and [SEP] at the beginning and ending for the input sentence.
        # batch_input = []
        for i, feature in enumerate(batch_features):
            padding_length = max_wordpiece_length - len(feature["input_ids"])
            input_ids = feature["input_ids"] + [self.tokenizer.pad_token_id] * padding_length
            mask = feature["attention_mask"] + [0] * padding_length
            inference_mask = feature["inference_masks"] + [0] * padding_length
            label_ids = feature["labels"] + [self.label2idx[PAD]] * padding_length
            wordpiece_length = len(input_ids)
            batch_features[i] = {"input_ids": input_ids,
                                 "attention_mask": mask,
                                 "labels": label_ids,
                                 "inference_masks": inference_mask,
                                 "wordpiece_len": wordpiece_length
                                 }
        encoded_inputs = {key: [example[key] for example in batch_features] for key in batch_features[0].keys()}
        results = BatchEncoding(encoded_inputs, tensor_type='pt')
        return results
