"""
config for using LLMs to conduct NER:
prompt style;
LLMs URL;
instruction style for SFT
"""
# OpenAI compatible server config
model_config = {'llama3.1':
                    ['/home/bml/mnt/conglei/chatbot/Models/LLMs/LLM-Research/Meta-Llama-3.1-8B-Instruct',
                     'http://10.188.48.146:8088/naturalLanguageProcessing/llama31-chat/v1/chat/completions']
                }

prompt_common = {
    'Conll2003': [
        (
            'Please list all named entities of the following entity types in the input sentence:\n'
            '{format_placeholder}'
            'Here are some examples:\n'
        ),
        (
            'You should only output a json format as {"entity type": [entity]}. like {"ORG": ["EU"], '
            '"MISC": ["German", "British"]} \n'
        )
    ],
    'wnut2017': [
        (
            'Please list all named entities of the following entity types in the input sentence:\n'
            '{format_placeholder}'
            'Here are some examples:\n'
        ),
        (
            'You should only output a json format as {"entity type": [entity]}. like {"group": ["Watch"], '
            '"product": ["What"], "work": ["What"], "location": [], "person": [], "corporation": ["Else is Making '
            'News"]}'
        )
    ],
}

prompt_entityinfo = {
    'Conll2003': [
        (
            'Please list all named entities of the following entity types in the input sentence:\n'
            '- PER: e.g. {PER}\n'
            '- ORG: e.g. {ORG}\n'
            '- LOC: e.g. {LOC}\n'
            '- MISC: e.g. {MISC}\n'
        ),

        (
            'You should output your results in the format {"type": [entity]} as a json, like {"ORG": ["EU"], "MISC": ["German", "British"]} \n'
        )
    ],
    'wnut2017': [
        (
            'Please list all named entities of the following entity types in the input sentence:\n'
            '- person: e.g. {person}\n'
            '- location: e.g. {location}\n'
            '- corporation: e.g. {corporation}\n'
            '- product: e.g. {product}\n'
            '- creative-work: e.g. {creative_work}\n'
            '- group: e.g. {group}\n'
        ),
        (
            'You should output your results in the format {"type": [entity]} as a json. like {"group": ["Watch"], '
            '"product": ["What"], "work": ["What"], "location": [], "person": [], "corporation": ["Else is Making'
            'News"]}'
        )

    ]}

system_prompt = 'You are a professional linguist, your job is extract structural information from text.'
