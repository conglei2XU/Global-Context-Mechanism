#!/usr/bin/env bash
export PATH=/opt/conda/envs/LLMs_torch_3.10/bin:$PATH
task_type=$1
dataset=$2
model=$3
context=$4
device=$5

batch_size_set=(16 30)
learning_rate_base_set=(1e-5 2e-5)
learning_rate_tagger_set=(5e-3 1e-3)
learning_rate_context_set=(1e-4 5e-4 5e-3)
# dropout=0.3 for rest16
dropout=0.1
for batch_size in "${batch_size_set[@]}"
do
  for learning_rate_base in "${learning_rate_base_set[@]}"
  do
    for learning_rate_tagger in "${learning_rate_tagger_set[@]}"
    do
      for learning_rate_context in "${learning_rate_context_set[@]}"
      do
        python main.py --task_type "${task_type}" --dataset_name "${dataset}" --mode pretrained --model_name  "${model}" --batch_size 16 --use_tagger True --use_context True --learning_rate "${learning_rate_base}" --learning_rate_tagger "${learning_rate_tagger}" --learning_rate_context "${learning_rate_context}" --num_layers 1 --fix_pretrained False --context_mechanism "${context}" --use_crf False --batch_size "${batch_size}" --device "${device}" --dropout "${dropout}"
      done
    done
  done
done