#!/usr/bin/env bash
# export LD_LIBRARY_PATH=/usr/local/cuda-12.1/compat
export PATH=/opt/conda/envs/LLMs_torch_3.10/bin:$PATH
task_type=$1
dataset=$2
model=$3
device=$4

# PYTHON="/opt/conda/envs/LLMs_torch_3.10/bin/python"
batch_size=16
learning_rate_base=2e-5
# learning_rate_tagger_set=(1e-5)
learning_rate_tagger=5e-3
learning_rate_crf_set=(8e-4 1e-3 5e-3)
dropout=0.3

for learning_rate_crf in "${learning_rate_crf_set[@]}"
do
  python main.py --batch_size "${batch_size}" --task_type "${task_type}" --dataset_name "${dataset}" --model_name "${model}" \
              --learning_rate "${learning_rate_base}" --learning_rate_tagger "${learning_rate_tagger}" \
            --use_tagger True --use_crf True --learning_rate_crf "${learning_rate_crf}" --use_context False --device "${device}" --no_improve 8 --num_epoch 20
done
