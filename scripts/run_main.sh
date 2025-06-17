#!/bin/bash

# for l in {1..31}; do
#     python main.py --model_path meta-llama/Meta-Llama-3-8B \
#         --dataset_name cais/mmlu \
#         --batch_size 8 \
#         --max_length 1024 \
#         --layers_to_skip $l \
#         --dataset_size 4000 \
#         --dataset_column question \
#         --dataset_subset "all" \
#         --split auxiliary_train
# done

model_list=("VirtualFormer-3.2-8L", "VirtualSkip-3.2-8L", "Vanilla-3.2-8L")

for model in "${model_list[@]}"; do
    python main.py --model_path huzama/${model} \
        --model_name ${model} \
        --revision pico-epoch_0 \
        --dataset_size 1280 \
        --dataset_name pico-lm/pretokenized-dolma \
        --batch_size 8 \
        --max_length 1024 \
        --split train
done