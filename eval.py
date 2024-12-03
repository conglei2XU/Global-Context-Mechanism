import re
import os
import json
from json.decoder import JSONDecodeError

from argparse import ArgumentParser


def init_args():
    argument = ArgumentParser()
    argument.add_argument('--result_dir', type=str, default='results/LLM-Inference')
    argument.add_argument('--dataset_dir', type=str, default='Dataset/LLM-Inference')
    # tasks specified configuration
    argument.add_argument('--task_type', type=str, default='NER')
    argument.add_argument('--dataset_name', type=str, default='wnut2017')
    argument.add_argument('--prompt_type', type=str, default='common')
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
        raw_result = pred_item['result']
        value = []
        # json_pattern_first = r'\{.*?\}'
        # json_pattern_first = r'Output:(.*?)'

        hash_key = hash(' '.join(origin_text))
        entity_mapping = {}
        if raw_result is None:
            print("llm'answer is invalid")
            pred[hash_key] = []
            continue
        else:
            raw_result = raw_result.replace('\n', '')
        match_flag, entity_mapping = parser_llms_ans(raw_result=raw_result)
        unmatch_num += match_flag
        if isinstance(entity_mapping, str):
            entity_mapping = json.loads(entity_mapping)
        for entity_type, entities in entity_mapping.items():
            if entities:
                for entity in entities:
                    value.append((entity_type, entity))
            pred[hash_key] = value
    print(f'the unmatched precentage of LLM-Inference ans : {unmatch_num / all_num}')
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


def parser_llms_ans(raw_result):
    entity_mapping = {}
    match_flag = False
    json_pattern_first = r'\{.*?\}'
    # json_pattern_first = r'Output:(.*?)'
    json_pattern_second = r'"(\w?)":\W?(\[.*?\])'
    json_pattern_backup = r"'(\w?)':\W?(\[.*?\])"
    try:
        json_ans = re.findall(json_pattern_first, raw_result)
        if json_ans:
            entity_mapping = json.loads(json_ans[-1].strip())
        else:
            print('json content parser error.')
    except JSONDecodeError:
        entity_mapping = {}
        first_matches = re.findall(json_pattern_second, json_ans[-1])
        backup_match = re.findall(json_pattern_backup, json_ans[-1])
        matches = first_matches if first_matches else backup_match
        if matches:
            match_flag = True
            for (entity_type, entities) in matches:
                if len(entities) > 100:
                    print(f'{entities} is too long')
                else:
                    left_bracket, right_bracket = entities.count('['), entities.count(']')
                    if left_bracket > right_bracket:
                        entities = entities.replace('[', '', 1)

                    entity_mapping[entity_type] = eval(entities)
        else:
            print('{entity_type: [entities] format results parser error')
            print(f'{raw_result}')

    return match_flag, entity_mapping


def main():
    args = init_args()
    gold_path = os.path.join(args.dataset_dir, args.task_type, args.dataset_name, 'test.jsonl')
    pred_path = os.path.join(args.result_dir, args.task_type, args.prompt_type, args.dataset_name,
                             'ans_limitations.json')
    f1 = llm_ner_metrics(gold_path=gold_path, pred_path=pred_path)
    print(f1)


if __name__ == "__main__":
    main()
