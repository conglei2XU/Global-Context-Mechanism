import os
import json
import random
from dataclasses import dataclass, field
from typing import Optional
import argparse

import jieba
import torch
import torch.nn as nn
from rouge_chinese import Rouge
import numpy as np
from loguru import logger
import bitsandbytes as bnb
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import HfArgumentParser
from trl import SFTTrainer, get_kbit_device_map, SFTConfig

from PipeLine.llms_pipeline import jsonl_data_gen


@dataclass
class CustomizedArguments:
    """
    一些自定义参数
    """
    train_file: str = field(metadata={"help": "训练集。如果task_type=pretrain，请指定文件夹，将扫描其下面的所有jsonl文件"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    eval_file: Optional[str] = field(default="", metadata={"help": "验证集"})
    max_prompt_length: int = field(default=512, metadata={"help": "dpo时，prompt的最大长度"})
    beta: float = field(default=0.1, metadata={"help": "The beta factor in DPO loss"})
    tokenize_num_workers: int = field(default=10, metadata={"help": "预训练时tokenize的线程数量"})
    task_type: str = field(default="sft", metadata={"help": "预训练任务：[pretrain, sft]"})
    train_mode: str = field(default="lora", metadata={"help": "训练方式：[full, qlora]"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def init_args():
    argument = argparse.ArgumentParser()
    argument.add_argument('--train_args_file', type=str, default='config/llama3.1_lora.json')
    argument.add_argument('--local_rank', type=int, default=0)
    args_cmd = argument.parse_args()
    hug_args_parser = HfArgumentParser((CustomizedArguments, SFTConfig))
    args, training_args = hug_args_parser.parse_json_file(json_file=args_cmd.train_args_file)
    setattr(training_args, 'packing', True)
    logger.info("train_args:{}".format(training_args))
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(os.path.join(training_args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(training_args))
    # 加载训练配置文件
    with open(args_cmd.train_args_file, "r") as f:
        train_args = json.load(f)
    # 保存训练参数到输出目录
    with open(os.path.join(training_args.output_dir, 'train_args.json'), "w") as f:
        json.dump(train_args, f, indent=4)
    # 设置随机种子
    set_seed(training_args.seed)
    return args, training_args


def find_all_linear_names(model, train_mode):
    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    logger.info(f'LoRA target module names: {lora_module_names}')
    return lora_module_names


def init_train_components(args, train_args):
    assert train_args.bf16 or train_args.fp16, 'bf16 or fp16 should be True'
    logger.info(f'Loading model from base model: {args.model_name_or_path}')
    logger.info(f'Train model with {args.train_mode}')
    torch_dtype = torch.float16 if train_args.fp16 else torch.bfloat16
    if args.train_mode == 'qlora':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if train_args.fp16 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None
    if quantization_config:
        model_kwargs = dict(
            trust_remote_code=True,
            # attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if train_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
    else:
        model_kwargs = dict(
            trust_remote_code=True,
            # attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if train_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.train_mode == 'qlora':
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train_args.gradient_checkpointing)
    if args.train_mode == 'lora' and args.task_type in ['pretrain', 'sft']:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    if args.train_mode == 'full':
        pref_config = None
    else:
        target_modules = find_all_linear_names(model, args.train_mode)
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type='CAUSAL_LM'
        )
    if args.train_mode in ['lora', 'qlora'] and args.task_type in ['pretrain', 'sft']:
        model = get_peft_model(model, peft_config)
        logger.info(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
        model.print_trainable_parameters()
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))

    return {
        'model': model,
        'peft_config': peft_config,
        'tokenizer': model_tokenizer
    }


def load_dataset(args, train_args, tokenizer):
    train_file = args.train_file
    eval_file = args.eval_file
    train_dataset_generator = jsonl_data_gen(train_file)
    train_dataset = Dataset.from_generator(train_dataset_generator)
    eval_dataset_generator = jsonl_data_gen(eval_file)
    eval_dataset = Dataset.from_generator(eval_dataset_generator)
    return train_dataset, eval_dataset


def main():
    args, train_arguments = init_args()
    components = init_train_components(args, train_arguments)
    components['tokenizer'].pad_token = components['tokenizer'].eos_token
    train_dataset, eval_dataset = load_dataset(args, train_arguments, components['tokenizer'])
    tokenizer = components['tokenizer']

    def compute_metric(eval_data):
        preds, golds = eval_data
        print(preds.size(), golds.size())
        if isinstance(preds, tuple):
            preds = preds[0]
        pred_tokens = tokenizer.batch_decode(preds, skip_special_tokens=True)
        golds = np.where(golds != -100, golds, tokenizer.pad_token_id)
        gold_tokens = tokenizer.batch_decode(golds, skip_special_tokens=True)
        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
        }
        for pred_item, gold_item in zip(pred_tokens, gold_tokens):
            pred_sent = list(jieba.cut(pred_item))
            gold_sent = list(jieba.cut(gold_item))
            rogue = Rouge()
            pred_sent_rogue = ' '.join(pred_sent)
            gold_sent_rogue = ' '.join(gold_sent)
            if not pred_sent_rogue:
                pred_sent_rogue = '-'
            scores = rogue.get_scores(pred_sent_rogue, gold_sent_rogue)
            desire_result = scores[0]
            for k, v in desire_result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    trainer = SFTTrainer(
        model=components['model'],
        args=train_arguments,
        tokenizer=components['tokenizer'],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metric
    )
    if train_arguments.do_train:
        # 开始训练
        logger.info("*** starting training ***")
        train_result = trainer.train()
        # 保存最好的checkpoint
        final_save_path = os.path.join(train_arguments.output_dir)
        trainer.save_model(final_save_path)  # Saves the tokenizer too
        # 保存训练指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    if train_arguments.do_eval:
        logger.info("*** starting evaluating ***")
        metrics = trainer.evaluate(metric_key_prefix="eval", top_p=0.7, max_length=512, temperature=0.95)
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)


if __name__ == "__main__":
    main()
