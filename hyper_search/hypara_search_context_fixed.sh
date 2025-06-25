#!/usr/bin/env bash
# export LD_LIBRARY_PATH=/usr/local/cuda-12.1/compat
export PATH=/opt/conda/envs/LLMs_torch_3.10/bin:$PATH
task_type=$1
dataset=$2
model=$3
context=$4
device=$5

# PYTHON="/opt/conda/envs/LLMs_torch_3.10/bin/python"
batch_size=16
learning_rate_base=1e-5
# learning_rate_tagger_set=(1e-5)
learning_rate_tagger=5e-2
learning_rate_context_set=(1e-4 8e-4 1e-3 5e-3)
# learning_rate_context_set=(5e-4)
dropout=0.1

for learning_rate_context in "${learning_rate_context_set[@]}"
do
  python main.py --batch_size "${batch_size}" --task_type "${task_type}" --dataset_name "${dataset}" --model_name "${model}" \
              --learning_rate "${learning_rate_base}" --learning_rate_tagger "${learning_rate_tagger}" \
            --use_tagger True --use_crf False --learning_rate_context "${learning_rate_context}" --use_context True --num_epoch 20 --dropout_rate "${dropout}" --context_mechanism "${context}" --device "${device}" --no_improve 8
done
