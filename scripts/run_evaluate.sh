#!/bin/bash

for l in {1..7}; do
    python src/evaluate.py \
        --model_name "Vanilla" \
        --method similarity-based \
        --num_layers_to_skip $l
done