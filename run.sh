#!/bin/bash

python main.py --model_path meta-llama/Meta-Llama-3-8B \
                      --dataset_name cais/mmlu \
                      --batch_size 8 \
                      --max_length 1024 \
                      --layers_to_skip 1 \
                      --dataset_size 4000 \
                      --dataset_column question \
                      --dataset_subset "all" \
                      --split auxiliary_train \