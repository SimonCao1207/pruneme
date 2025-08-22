#!/bin/bash

python src/evaluate.py \
    --model_path huzama/Full-3.2-16L \
    --model_name Full-3.2-16L \
    --method random \
    --config configs/random.yaml \
    --num_layers_to_skip 4