"""
This module convert dataset from BIO format to LLMs format for In-context learning
    {
        "text": "AL-AIN , United Arab Emirates 1996-12-06",
        "label": {
            "LOC": [
                "AL-AIN",
                "United Arab Emirates"
            ]
        }
    }
"""

import os
import json
import argparse
from collections import defaultdict

from utils.file_reader import ner_reader, ner_reader_cn


def init_args():
    argument = argparse.ArgumentParser()
    argument.add_argument('--dataset_source', type=str, default='Dataset')
    argument.add_argument('--target_dir', type=str, default='Dataset/LLM')
    argument.add_argument('--task_type', type=str, default='NER')
    argument.add_argument('--dataset_name', type=str, default='Conll2003')
    args = argument.parse_args()
    return args


def get_entity(labels, sentence):
    """
    labels: list of BIO labels
    sentence: original sentence
    output: convert list of BIO labels to entities
    """
    length = len(labels)
    chunk = [-1, -1, -1, '']
    chunks = []
    for idx, value in enumerate(labels):
        if value.startswith('B-'):
            if chunk[0] != -1:
                chunks.append(chunk)
            category = value.split('-')[-1]
            chunk = [category, idx, idx, sentence[idx]]

            if idx == length - 1:
                chunks.append(chunk)
        elif value.startswith('I-'):
            category = value.split('-')[-1]
            pre_category = chunk[0]
            if category == pre_category:
                chunk[-2] = idx
                chunk[-1] += sentence[idx]
            else:
                if isinstance(pre_category, str):
                    chunks.append(chunk)
                    chunk = [-1, -1, -1, '']
            if idx == length - 1:
                chunks.append(chunk)

        else:
            if chunk[1] != -1:
                chunks.append(chunk)
                chunk = [-1, -1, -1, '']
    return chunks


def main():
    args = init_args()
    source_path = os.path.join(args.dataset_source, args.task_type, args.dataset_name)
    target_path = source_path.replace(args.dataset_source, args.target_dir)
    if 'weibo' in source_path.lower():
        reader = ner_reader_cn
    else:
        reader = ner_reader
    print(f'Loading data from: {source_path}')
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    files = os.listdir(source_path)
    for file_item in files:
        file_name = file_item.split('.')[0]
        if file_name not in ['train', 'test', 'valid']:
            continue
        target_file_name = file_name + '.' + 'jsonl'
        target_file_path = os.path.join(target_path, target_file_name)
        source_file_path = os.path.join(source_path, file_item)
        print(f'converting {source_file_path} to {target_file_path}')
        with open(target_file_path, 'w') as fw:
            for sentence, label in reader(source_file_path):
                all_entities = defaultdict(list)
                entities = get_entity(label, sentence)
                for (entity_type, _, _, entity) in entities:
                    entity_cache = all_entities[entity_type]
                    if entity not in entity_cache:
                        entity_cache.append(entity)
                json_line = json.dumps({'text': sentence, 'label': all_entities})
                fw.write(json_line + '\n')
        print(f'{source_file_path} finished')
        print('\n')


if __name__ == "__main__":
    main()
