import os
import json
import random
import requests
import argparse
import tqdm
from collections import defaultdict

from config import prompt_common, prompt_entityinfo, system_prompt, model_config


def init_args():
    argument = argparse.ArgumentParser()
    argument.add_argument('--dataset_dir', type=str, default='Dataset/LLM')
    argument.add_argument('--result_dir', type=str, default='results/LLM')
    # tasks specified configuration
    argument.add_argument('--task_type', type=str, default='NER')
    argument.add_argument('--dataset_name', type=str, default='Conll2003')

    argument.add_argument('--num_shots', type=int, default=3)
    argument.add_argument('--prompt_type', type=str, choices=['common', 'entityinfo'], default='common')
    argument.add_argument('--model_name', type=str, default='llama3.1')

    args = argument.parse_args()
    return args


class PromptGenerator:
    def __init__(self, dataset, prompt_name, dataset_name):
        entities_type = set()
        all_data = []
        type2name = defaultdict(set)

        with open(dataset, 'r') as f:
            for line in f:
                data_item = json.loads(line.strip())
                labels = list(data_item['label'].keys())
                for entity_type, entity_name in data_item['label'].items():
                    type2name[entity_type].update(entity_name)
                entities_type.update(labels)
                all_data.append(data_item)
        self.all_data = all_data
        self.entities_type = entities_type
        self.type2name = type2name
        self.prompt_name = prompt_name
        self.dataset_name = dataset_name

    def generate(self, num_shot):
        prompt_content = ''
        demo_samples = random.sample(self.all_data, num_shot)
        if self.prompt_name == 'common':
            prompt_first = prompt_common[self.dataset_name][0]
            prompt_second = prompt_common[self.dataset_name][1]
            format_placeholder = ''
            for entity in self.entities_type:
                format_placeholder += f'- {entity}\n'
            prompt_first = prompt_first.format(format_placeholder=format_placeholder)

            for sample in demo_samples:
                prompt_content += 'Input: ' + ' '.join(sample['text']) + '\nOutput: ' + str(sample['label']).replace(
                    "'", '"') + '\n'
        else:
            point_entity = {}
            prompt_first = prompt_entityinfo[self.dataset_name][0]
            prompt_second = prompt_entityinfo[self.dataset_name][1]
            for entity_type, entities in self.type2name.items():
                point_entity[entity_type] = random.sample(list(entities), num_shot)
            if self.dataset_name == 'Conll2003':
                per = ", ".join(['"' + x + '"' for x in point_entity['PER']])
                org = ", ".join(['"' + x + '"' for x in point_entity['ORG']])
                loc = ", ".join(['"' + x + '"' for x in point_entity['LOC']])
                misc = ", ".join(['"' + x + '"' for x in point_entity['MISC']])
                prompt_first = prompt_first.format(PER=per, ORG=org, LOC=loc, MISC=misc)
            for sample in demo_samples:
                prompt_content += 'Input: ' + ' '.join(sample['text']) + '\nOutput: ' + str(sample['label']).replace(
                    "'",
                    '"') + '\n'
        prompt = prompt_first + prompt_content + prompt_second + '\nInput: '

        return prompt



def get_llm_ans(prompt, model):
    model_name = model_config[model][0]
    model_api = model_config[model][1]
    data = {'model': model_name}
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]
    data['messages'] = messages
    llm_ans = requests.post(url=model_api, data=json.dumps(data))
    try:
        resp_content = json.loads(llm_ans.text)['choices'][0]['message']['content']
        return resp_content
    except Exception as msg:
        print(f"llm error response {msg}")
        return None


def main():
    args = init_args()
    dataset = os.path.join(args.dataset_dir, args.task_type, args.dataset_name, 'train.jsonl')
    test_data = os.path.join(args.dataset_dir, args.task_type, args.dataset_name, 'test.jsonl')
    ans_save_dir = os.path.join(args.result_dir, args.task_type, args.dataset_name)
    if not os.path.exists(ans_save_dir):
        os.makedirs(ans_save_dir)

    prompt_generator = PromptGenerator(dataset=dataset, prompt_name=args.prompt_type, dataset_name=args.dataset_name)

    all_ans = []
    with open(test_data, 'r') as f:
        all_lines = f.readlines()
        for line in tqdm.tqdm(all_lines):
            prompt = prompt_generator.generate(args.num_shots)
            test_item = json.loads(line.strip())
            sent = ' '.join(test_item['text'])
            ans = get_llm_ans(prompt + sent, args.model_name)
            ans_item = {'result': ans,
                        'text': test_item['text']}
            all_ans.append(ans_item)
    print(f'writing prediction into {ans_save_dir}')
    with open(os.path.join(ans_save_dir, 'ans.json'), 'w', encoding='utf-8') as f:
        json.dump(all_ans, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
