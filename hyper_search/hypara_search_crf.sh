#!/usr/bin/env bash
export PATH=/opt/conda/envs/LLMs_torch_3.10/bin:$PATH
task_type=$1
dataset=$2
model=$3
device=$4

batch_size_set=(16 30)
learning_rate_base_set=(1e-5 2e-5)
learning_rate_tagger_set=(5e-3 1e-3)
learning_rate_context_set=(1e-4 5e-4 5e-3)
dropout=0.1
for batch_size in "${batch_size_set[@]}"
do
  for learning_rate_base in "${learning_rate_base_set[@]}"
  do
    for learning_rate_tagger in "${learning_rate_tagger_set[@]}"
    do
      for learning_rate_crf in "${learning_rate_context_set[@]}"
      do
        python main.py --task_type "${task_type}" --dataset_name "${dataset}" --mode pretrained --model_name  "${model}" --batch_size 16 --use_tagger True --learning_rate "${learning_rate_base}" --learning_rate_tagger "${learning_rate_tagger}" --learning_rate_crf "${learning_rate_crf}" --num_layers 1 --fix_pretrained False --use_crf True --batch_size "${batch_size}" --use_context False --device "${device}"
      done
    done
  done
done