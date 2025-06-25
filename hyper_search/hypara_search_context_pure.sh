#!/usr/bin/env bash
export PATH=/opt/conda/envs/LLMs_torch_3.10/bin:$PATH
task_type=$1
dataset=$2
model=$3
context=$4
device=$5

batch_size_set=(16 30)
learning_rate_base_set=(1e-5 2e-5)
# learning_rate_tagger_set=(1e-5)
learning_rate_context_set=(1e-4 5e-4 5e-3 1e-3)
dropout=0.1
for batch_size in "${batch_size_set[@]}"
do
  for learning_rate_base in "${learning_rate_base_set[@]}"
  do
    for learning_rate_context in "${learning_rate_context_set[@]}"
    do
      python main.py --batch_size "${batch_size}" --task_type "${task_type}" --dataset_name "${dataset}" --model_name "${model}" --learning_rate "${learning_rate_base}" --num_epoch 20 --use_context True --use_tagger False --use_crf False --device "${device}" --fix_pretrained False --no_improve 8 --num_layers 1 --context_mechanism "${context}" --learning_rate_context "${learning_rate_context}" --dropout "${dropout}"
    done
  done
done