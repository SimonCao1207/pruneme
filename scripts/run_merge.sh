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

model_list=("Vanilla-3.2-8L" "VirtualSkip-3.2-8L")
for model in "${model_list[@]}"; do
    for layers_to_skip in {1..7}; do
        python merge.py \
            --model_path "huzama/${model}" \
            --layers_to_skip $layers_to_skip \
            --revision pico-epoch_0 \
            --method "$method" \
            --num_layers 8
    done
done