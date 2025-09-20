#!/bin/bash

for dataset_name in pico
do
    echo "Running Similarity-based on $dataset_name"
    python src/similarity.py --dataset_name $dataset_name --method similarity-based
done
