#!/usr/bin/env bash
export PATH=/opt/conda/envs/LLMs_torch_3.10/bin:$PATH
task_type=$1
dataset=$2
model=$3
device=$4
PYTHON="/opt/conda/envs/LLMs_torch_3.10/bin/python"
batch_size_set=(16 30)
learning_rate_base_set=(1e-5 8e-5)
# learning_rate_tagger_set=(1e-5)
learning_rate_tagger_set=(5e-4 5e-3 1e-3)
dropout=0.1
for batch_size in "${batch_size_set[@]}"
do
  for learning_rate_base in "${learning_rate_base_set[@]}"
  do
    for learning_rate_tagger in "${learning_rate_tagger_set[@]}"
    do
      python main.py --batch_size "${batch_size}" --task_type "${task_type}" --dataset_name "${dataset}" --model_name "${model}" \
              --learning_rate "${learning_rate_base}" --learning_rate_tagger "${learning_rate_tagger}" --num_epoch 20 \
              --use_context False --use_tagger True --use_crf False --device "${device}" --fix_pretrained False --no_improve 8 --num_layers 1
    done
  done
done