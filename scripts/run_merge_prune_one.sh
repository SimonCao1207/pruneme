model_list=("Vanilla-3.2-8L" "VirtualSkip-3.2-8L")
for model in "${model_list[@]}"; do
    for prune_layer in {0..7}; do
        python merge.py \
            --model_path "huzama/${model}" \
            --layers_to_skip 1 \
            --prune_layer $prune_layer \
            --revision pico-epoch_0 \
            --method prune-one \
            --num_layers 8
    done
done