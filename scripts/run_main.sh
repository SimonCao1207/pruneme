#!/bin/bash
for i in 1 2 4 6 8
do
python src/main.py \
    --num_layers_to_skip $i \
    --method similarity-based
done