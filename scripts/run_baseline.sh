for dataset_name in pico
do
    echo "Running GAP on $dataset_name with 0 layers skipped"
    python src/evaluate.py \
        --method gap \
        --dataset_name $dataset_name \
        --num_layers_to_skip 0 \
        --batch_size 8
done