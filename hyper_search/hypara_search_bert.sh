#!/usr/bin/env bash
task_type=$1
dataset=$2
model=$3
device=$4
export PATH=/opt/conda/envs/LLMs_torch_3.10/bin:$PATH
# PYTHON="/opt/conda/envs/vllm_deploy/bin/python"
batch_size_set=(16 30)
learning_rate=(1e-5 2e-5  8e-5)
weight_decay=(1e-6)

for batch_size in "${batch_size_set[@]}"
do
  for learning_rate in "${learning_rate[@]}"
  do
    for weight_decay in "${weight_decay[@]}"
    do
      python main.py --batch_size "${batch_size}" --task_type "${task_type}" --dataset_name "${dataset}" --model_name "${model}" \
              --learning_rate "${learning_rate}" --use_context False --use_tagger False --weight_decay "${weight_decay}" --num_epoch 15 --device "${device}"
    done
  done
done