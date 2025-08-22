#!/bin/bash

for l in {1..15}; do
    python src/main.py \
        --num_layers_to_skip $l \
        --method prune-one
done