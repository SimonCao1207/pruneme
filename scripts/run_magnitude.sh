#!/bin/bash

for dataset_name in wikitext
do
    echo "Running Magnitude on $dataset_name"
    python src/taylor.py --dataset_name $dataset_name --method magnitude --batch_size 8
done