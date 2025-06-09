#!/bin/bash


model_list=("Vanilla-3.2-8L" "VirtualSkip-3.2-8L")

# To run this script, use: bash run_evaluate.sh
for model in "${model_list[@]}"; do
    python evaluate.py --model_path "huzama/${model}" \
        --revision pico-epoch_0 \
        --dataset_name cais/mmlu \
        --batch_size 8 \
        --max_length 1024 \
        --dataset_subset "all" \
        --split test
done