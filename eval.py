import re
import os
import json
from json.decoder import JSONDecodeError

from argparse import ArgumentParser


def init_args():
    argument = ArgumentParser()
    argument.add_argument('--result_dir', type=str, default='results/LLM')
    argument.add_argument('--dataset_dir', type=str, default='Dataset/LLM')
    # tasks specified configuration
    argument.add_argument('--task_type', type=str, default='NER')
    argument.add_argument('--dataset_name', type=str, default='Conll2003')
    args = argument.parse_args()
    return args


def llm_ner_metrics(gold_path, pred_path):
    with open(pred_path, 'r', encoding='utf-8') as f:
        raw_pred = json.load(f)
    gold = {}
    pred = {}
    with open(gold_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_item = json.loads(line.strip())
            origin_text = data_item['text']
            value = []
            for entity_type, entities in data_item['label'].items():
                if entities:
                    for entity in entities:
                        value.append((entity_type, entity))
            hash_key = hash(' '.join(origin_text))
            gold[hash_key] = value
    all_num, unmatch_num = 0, 0
    for pred_item in raw_pred:
        all_num += 1
        origin_text = pred_item['text']
        raw_results = pred_item['result']
        value = []

        json_pattern = r'"(\w+)":\W?(\[.*?\])'
        hash_key = hash(' '.join(origin_text))
        try:
            # raw_results = raw_results.replace("'", '"')
            # raw_results = re.sub(r'(\w+)"s', r"\1's", raw_results)
            # raw_results = re.sub(r'(\w+)"( \w)', r"\1'\2", raw_results)
            # raw_results = re.sub(r'(\w+) "S', r"\1 'S", raw_results)
            # raw_results = re.sub(r'(\w+)"(\w)', r"\1'\2", raw_results)
            entity_mapping = json.loads(raw_results.strip())
        except JSONDecodeError as e:
            entity_mapping = {}
            matches = re.findall(json_pattern, raw_results)
            if matches:
                for (entity_type, entities) in matches:
                    entity_mapping[entity_type] = eval(entities)
            else:
                unmatch_num += 1
                pred[hash_key] = []
                continue
        if isinstance(entity_mapping, str):
            entity_mapping = json.loads(entity_mapping)
        for entity_type, entities in entity_mapping.items():
            if entities:
                for entity in entities:
                    value.append((entity_type, entity))
        pred[hash_key] = value
    print(f'the unmatched precentage of LLM ans : {unmatch_num/all_num}')
    nb_true, nb_pred, nb_label = 0, 0, 0
    for hash_key, pred_ans in pred.items():
        gold_ans = gold[hash_key]
        nb_pred += len(set(pred_ans))
        nb_label += len(gold_ans)
        assert len(set(gold_ans)) == len(gold_ans)
        nb_true += len(set(pred_ans) & set(gold_ans))

    precision = nb_true / nb_pred
    recall = nb_true / nb_label
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def main():
    args = init_args()
    gold_path = os.path.join(args.dataset_dir, args.task_type, args.dataset_name, 'test.jsonl')
    pred_path = os.path.join(args.result_dir,args.task_type, args.dataset_name, 'ans.json')
    f1 = llm_ner_metrics(gold_path=gold_path, pred_path=pred_path)
    print(f1)


if __name__ == "__main__":
    main()
