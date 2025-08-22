for l in {1..15}; do
    python src/prune.py \
        --config configs/random.yaml \
        --num_layers_to_skip $l \
        --method random
done