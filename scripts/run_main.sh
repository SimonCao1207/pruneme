#!/bin/bash

for l in {1..31}; do
    python main.py --model_path meta-llama/Meta-Llama-3-8B \
        --dataset_name cais/mmlu \
        --batch_size 8 \
        --max_length 1024 \
        --num_layers_to_skip $l \
        --dataset_size 4000 \
        --dataset_column question \
        --dataset_subset "all" \
        --split auxiliary_train
done