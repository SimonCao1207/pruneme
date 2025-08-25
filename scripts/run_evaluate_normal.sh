#!/bin/bash

for i in 6 8
do
    python src/evaluate.py \
        --model_name "Full" \
        --method normal \
        --num_layers_to_skip $i
done