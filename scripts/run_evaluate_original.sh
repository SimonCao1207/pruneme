#!/bin/bash


model_list=("Vanilla-3.2-8L" "VirtualSkip-3.2-8L")

# To run this script, use: bash run_evaluate.sh
# for model in "${model_list[@]}"; do
#     python evaluate.py --model_path "huzama/${model}" \
#         --revision pico-epoch_0 \
#         --dataset_name cais/mmlu \
#         --batch_size 8 \
#         --max_length 1024 \
#         --dataset_subset "all" \
#         --split test
# done

for model in "${model_list[@]}"; do
    python evaluate.py --model_path "huzama/${model}" \
        --model_name Vanilla-3.2-8L \
        --revision pico-epoch_0 \
        --dataset_size 1280 \
        --dataset_name pico-lm/pretokenized-dolma \
        --batch_size 8 \
        --max_length 1024 \
        --split train
done