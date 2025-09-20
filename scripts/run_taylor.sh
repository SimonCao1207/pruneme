#!/bin/bash

for dataset_name in hellaswag winogrande arc_easy openbook_qa copa social_iqa
do
    echo "Running Taylor on $dataset_name"
    python src/taylor.py --dataset_name $dataset_name --method taylor
done