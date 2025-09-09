#!/bin/bash

python src/evaluate.py \
    --model_name "Full" \
    --method prune-multiple \
    --prune_layers 1 2 3 4 5 6 7 8