#!/usr/bin/env bash

export PATH=/opt/conda/envs/LLMs_torch_3.10/bin:$PATH
task_type=$1
dataset=$2
model=$3
batch_size=32
learning_rate_base=5e-5
learning_rate_tagger=5e-4
learning_rate_context_set=(1e-5 5e-5 1e-4 5e-4 8e-4 1e-3 5e-3 1e-2)
dropout_set=(0.3 0.1 0.0)
for learning_rate_context in "${learning_rate_context_set[@]}"
do
  for dropout in "${dropout_set[@]}"
  do
    echo "${learning_rate_context}"
    python main.py --batch_size "${batch_size}" --task_type "${task_type}" --dataset_name "${dataset}" --model_name "${model}" \
              --learning_rate "${learning_rate_base}" --learning_rate_tagger "${learning_rate_tagger}" \
              --use_context True --learning_rate_context "${learning_rate_context}" --dropout_rate "${dropout}" \
              --context_mechanism global
  done
done