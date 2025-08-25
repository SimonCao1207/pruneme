#!/bin/bash

for i in 2
do
    python src/evaluate.py \
        --model_name "Full" \
        --method similarity-based \
        --num_layers_to_skip $i
done