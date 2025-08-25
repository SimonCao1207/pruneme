#!/bin/bash

python src/evaluate.py \
    --model_name "Full" \
    --method prune-multiple \
    --prune_layers 13