#!/bin/bash

model_list=("Vanilla-3.2-8L" "VirtualSkip-3.2-8L")

# Parse --method argument
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --method) method="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$method" ]; then
    echo "Error: --method argument is required."
    exit 1
fi

# To run this script, use: bash run_evaluate_prunned.sh --method <METHOD>
for model in "${model_list[@]}"; do
    for i in {1..7}; do
        python evaluate.py --model_path "merged/${model}/${method}/${i}" \
            --model_name "${model}" \
            --method "$method" \
            --layers_to_skip $i \
            --dataset_name cais/mmlu \
            --batch_size 8 \
            --max_length 1024 \
            --dataset_subset "all" \
            --split test
    done
done