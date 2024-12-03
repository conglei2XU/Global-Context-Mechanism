#!/usr/bin/env bash
export PATH=/opt/conda/envs/LLMs_torch_3.10/bin:$PATH
task_type=$1
dataset=$2
use_context=$3
model=$4
context=$5

PYTHON="/opt/conda/envs/LLMs_torch_3.10/bin/python"
batch_size_set=(16 32)
learning_rate_base_set=(1e-5 5e-5)
# learning_rate_tagger_set=(1e-5)
learning_rate_tagger_set=(1e-4 5e-4 1e-3 5e-3)
learning_rate_context_set=(1e-4 5e-4 1e-3 5e-3)
dropout=0.1
for batch_size in "${batch_size_set[@]}"
do
  for learning_rate_base in "${learning_rate_base_set[@]}"
  do
    for learning_rate_tagger in "${learning_rate_tagger_set[@]}"
    do
      if [ "$use_context" = true ]; then
        for learning_rate_context in "${learning_rate_context_set[@]}"
        do
          if [ "$context" = "global" ]; then

            python main.py --batch_size "${batch_size}" --task_type "${task_type}" --dataset_name "${dataset}" --model_name "${model}" \
              --learning_rate "${learning_rate_base}" --learning_rate_tagger "${learning_rate_tagger}" \
              --use_context True --learning_rate_context "${learning_rate_context}" --dropout_rate "${dropout}" \
              --context_mechanism global
          else
            python main.py --batch_size "${batch_size}" --task_type "${task_type}" --dataset_name "${dataset}" --model_name "${model}" \
          --learning_rate "${learning_rate_base}" --learning_rate_tagger "${learning_rate_tagger}" \
          --use_context True --learning_rate_context "${learning_rate_context}" --context_mechanism self-attention
          fi
        done
      else
        python main.py --batch_size "${batch_size}" --task_type "${task_type}" --dataset_name "${dataset}" --model_name "${model}" \
              --learning_rate "${learning_rate_base}" --learning_rate_tagger "${learning_rate_tagger}" \
              --use_context False
      fi
    done
  done
done