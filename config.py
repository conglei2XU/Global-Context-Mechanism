"""
config for using LLMs to conduct NER:
prompt style;
LLMs URL;
instruction style for SFT

"""
# OpenAI compatible server config
model_config = {'llama3.1':
    [
        # '/home/bml/mnt/conglei/chatbot/Models/LLMs/LLM-Research/Meta-Llama-3.1-8B-Instruct',
        # 'Meta-Llama-3.1-8B-Instruct',
        '/home/bml/storage/mnt/v-ehn3x2i8d9bd43e3/org/conglei/chatbot/Models/LLMs/LLM-Research/Meta-Llama-3.1-8B-Instruct',
        'http://10.125.128.9:39391/naturalLanguageProcessing/llama31-chat/v1/chat/completions'
        # 'http://127.0.0.1:8000/v1/chat/completions'
    ]

}

prompt_common = {
    'Conll2003': [
        (
            'Please list all named entities of the following entity types in the input sentence:\n'
            '{format_placeholder}'
            'Here are some examples:\n'
        ),
        (
            'You should output your results in the format {"type": [entity]} as a json.'
        )
    ],
    'wnut2017': [
        (
            'Please list all named entities of the following entity types in the input sentence:\n'
            '{format_placeholder}'
            'Here are some examples:\n'
        ),
        (
            'You should output your results in the format {"type": [entity]} as a json'
        )
    ],
    'laptop14': [
        (
            'Please classify the aspect words of a laptop from the comments in the following types:\n'
            '{format_placeholder}'
            'Here are some examples:\n'
        ),
        (
            'Output your results in the following JSON format \n'
            '{"type": [entity]}'
        )
    ]
    ,
    'rest': [
        (
            'Please classify the aspect words of a restaurant from the comments in the following types:\n'
            '{format_placeholder}'
            'Here are some examples:\n'
        ),
        (
            'Output your results in the following JSON format \n'
            '{"type": [entity]}'
        )
    ]
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
            'You should output your results in the format {"type": [entity]} as a json.'
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
            'Output your results in the following JSON format:\n'
            '{"type": [entity]}'
        )

    ],
    'laptop14': [
        (
            'Please classify the aspect words of a laptop from the comments in the following types:\n'
            '- NEG: e.g. {negative}\n'
            '- POS: e.g. {positive}\n'
            '- NEU: e.g. {neutral}\n'
            'Here are some examples:\n'
        ),
        (
            'Output your results in the following JSON format:\n'
            '{"type": [entity]}'
        )

    ],
    'rest': [
        (
            'Please classify the aspect words of a restaurant from the comments in the following types:\n'
            '- NEG: e.g. {negative}\n'
            '- POS: e.g. {positive}\n'
            '- NEU: e.g. {neutral}'
            'Here are some examples:\n'
        ),
        (
            'Output your results in the following JSON format:\n'
            '{"type": [entity]}'
        )

    ]
}

prompt_finetuning = {
    'Conll2003': [(' Please list all named entities of the following entity types in the input sentence:\n'
                   '- PER \n'
                   '- ORG \n'
                   '- LOC \n'
                   '- MISC \n'
                   'You should output your results in the format {"type": [entity]} as a json.\n '),
                  'Input: %s \n Output: '
                  ],
    'wnut2017': [('Please list all named entities of the following entity types in the input sentence:\n'
                  '- person \n'
                  '- location \n'
                  '- corporation \n'
                  '- product \n'
                  '- creative-work \n'
                  '- group \n'
                  'You should output your results in the format {"type": [entity]} as a json.\n '),
                 'Input: %s \n Output: '
                 ],
    'laptop14':
        [
            (
                'Please classify the described objects in a comment of'
                ' a computer in the following types:\n'
                '- NEG: Denotes negative attitude towards aspects of a computer \n'
                '- POS: Denotes positive attitude towards aspects of a computer \n'
                '- NEU: Denotes neutral attitude towards aspects of a computer \n'
                'You should output your results in the format {"type": [entity]} as a json.\n '),
            'Input: %s \n Output: '
        ],
    'rest':
        [
            (
                'Please classify the described objects in a comment of'
                ' a restaurant in the following types:\n'
                '- NEG: Denotes negative attitude towards aspects of a restaurant \n'
                '- POS: Denotes positive attitude towards aspects of a restaurant \n'
                '- NEU: Denotes neutral attitude towards aspects of a restaurant \n'
                'You should output your results in the format {"type": [entity]} as a json.\n'),
            'Input: %s \n Output: '
        ]
}

SYSTEM_PROMPT = {'NER': 'You are a professional linguist, your job is extract structural information from text.',
                 'absa': 'You are a sentiment analysis model. Your task is to '
                         'analyze the sentiment of the given objects in sentence and classify'
                         ' it into one of the following categories: Positive, Negative, or Neutral.'}

Selection = {'Conll2003': [20, 1, 13],
             'wnut2017': [219, 227, 241],
             'laptop14': [64, 104, 132],
             'rest14': [0, 11, 108],
             'rest15': [9, 12, 34],
             'rest16': [13, 72, 34]}

Definition = {
    'laptop14': {
        'NEG': 'Denotes negative attitude towards aspects of a computer (e.g., %s)',
        'NEU': 'Denotes neutral attitude towards aspects of a computer (e.g., %s})',
        'POS': 'Denotes positive attitude towards aspects of a computer (e.g., %s)'
    },
    'rest': {
        'NEG': 'Denotes negative attitude towards aspects of a restaurant (e.g., %s)',
        'NEU': 'Denotes neutral attitude towards aspects of a restaurant (e.g., %s)',
        'POS': 'Denotes positive attitude towards aspects of a restaurant (e.g., %s)'
    }
}
