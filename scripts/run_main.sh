#!/bin/bash

for l in {1..7}; do
    python src/main.py \
        --num_layers_to_skip $l \
        --method similarity-based
done