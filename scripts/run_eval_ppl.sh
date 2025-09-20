#!/bin/bash

for dataset_name in pico wikitext
do
    for num_skips in 0 1 2 4 6 8
    do
        echo "Running GAP on $dataset_name with $num_skips layers skipped"
        python src/evaluate.py \
            --method gap \
            --dataset_name $dataset_name \
            --num_layers_to_skip $num_skips \
            --batch_size 8 

        echo "Running Similarity on $dataset_name with $num_skips layers skipped"
        python src/evaluate.py \
            --method similarity-based \
            --dataset_name $dataset_name \
            --num_layers_to_skip $num_skips \
            --batch_size 8

        echo "Running Taylor on $dataset_name with $num_skips layers skipped"
        python src/evaluate.py \
            --method taylor \
            --dataset_name $dataset_name \
            --num_layers_to_skip $num_skips \
            --batch_size 8

        echo "Running Magnitude on $dataset_name with $num_skips layers skipped"
        python src/evaluate.py \
            --method magnitude \
            --dataset_name $dataset_name \
            --num_layers_to_skip $num_skips \
            --batch_size 8
    done
done