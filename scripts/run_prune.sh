for l in {1..7}; do
    python src/prune.py \
        --num_layers_to_skip $l \
        --method similarity-based
done