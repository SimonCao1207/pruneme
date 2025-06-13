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

python main.py --model_path huzama/Vanilla-3.2-8L \
    --model_name Vanilla-3.2-8L \
    --revision pico-epoch_0 \
    --dataset_size 1280 \
    --dataset_name pico-lm/pretokenized-dolma \
    --batch_size 8 \
    --max_length 1024 \
    --split train