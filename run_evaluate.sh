#!/bin/bash

python evaluate.py --model_path meta-llama/Meta-Llama-3-8B \
                      --dataset_name cais/mmlu \
                      --batch_size 8 \
                      --max_length 1024 \
                      --dataset_subset "all" \
                      --split test \