import json
from collections import Counter


def ner_reader(dataset_path):
    """
    read all words in dataset
    :param dataset_path: (str)
    :return: file generator
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        sentence = []
        labels = []
        for line in f:
            if line != '\n' and line != '\t\n':
                if '\t' in line:
                    pairs = line.split('\t')
                else:
                    pairs = line.split()
                sentence.append(pairs[0])
                labels.append(pairs[-1].strip())
            else:
                if sentence:
                    yield sentence, labels
                sentence, labels = [], []
        if sentence:
            yield sentence, labels


def pos_reader(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        sentence = []
        labels = []
        for line in f:
            if line != '\n' and line != '\t\n':
                if '\t' in line:
                    pairs = line.split('\t')
                else:
                    pairs = line.split()
                sentence.append(pairs[0])
                labels.append(pairs[1].strip())
            else:
                if sentence:
                    yield sentence, labels
                sentence, labels = [], []
        if sentence:
            yield sentence, labels


def ner_reader_cn(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        sentence = []
        labels = []
        for line in f:
            if line != '\n' and line != '\t\n':
                pairs = line.strip()
                sentence.append(pairs[0])
                labels.append(pairs[-1])
            else:
                if sentence:
                    yield sentence, labels
                sentence, labels = [], []
        if sentence:
            yield sentence, labels


def jsonl_reader(dataset_path):
    """
    read data form jsonl files with structure {
    'content': content,
    'label': label
    }
    return content[list of str] and label[list of label]
    """
    over_length_sent = 1
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_item = json.loads(line)
            sentence = list(data_item['content'])
            label = list(data_item['label'])
            yield sentence, label




if __name__ == "__main__":
    data_path = '../Dataset/NER/ChinaUnicomNER/train_BIO.jsonl'
    jsonl_reader(data_path)

