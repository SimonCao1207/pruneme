import argparse
import os

import pandas as pd
import torch
import yaml

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

OUTPUT_PATH = "./merged"  # folder to store the result in
LORA_MERGE_CACHE = "/tmp"  # change if you want to keep these for some reason
CONFIG_YML = "./slice.yaml"  # merge configuration file
COPY_TOKENIZER = True
LAZY_UNPICKLE = False  # experimental low-memory model loader
LOW_CPU_MEMORY = False  # enable if you somehow have more VRAM than RAM+swap


def get_prune_layers(args):
    model_name = os.path.basename(args.model_path)
    num_dropped_layers = args.layers_to_skip
    file_name = f"outputs/{model_name}/prune_{num_dropped_layers}_layers.csv"
    df = pd.read_csv(file_name)
    min_row = df.loc[df["average_distance"].idxmin()]
    return {
        "block_start": int(min_row["block_start"]),
        "block_end": int(min_row["block_end"]),
        "average_distance": float(min_row["average_distance"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="huzama/Vanilla-3.2-8L",
        type=str,
        help="Path to the model.",
    )
    parser.add_argument("--num_layers", default=8, type=int, help="Number of layers")
    parser.add_argument(
        "--layers_to_skip", default=1, type=int, help="Number of layers to skip."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="pico-epoch_0",
        help="Model revision (optional).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="normal",
        choices=["normal", "similarity-based", "prune-one"],
        help="Method to determine layers to prune. ",
    )
    parser.add_argument(
        "--prune_layer",
        default=None,
        type=int,
        help="Which layer to prune (for prune-one method).",
    )
    args = parser.parse_args()

    if args.method == "normal":
        first_block = [0, args.num_layers - args.layers_to_skip]
        second_block = [
            args.num_layers - args.layers_to_skip,
            args.num_layers - args.layers_to_skip,
        ]  # empty block
    elif args.method == "similarity-based":
        # layer_range is half open in mergekit (https://github.com/arcee-ai/mergekit/issues/206)
        # Skip layers block_start to block_end-1
        prune_layers_info = get_prune_layers(args)
        first_block = [0, prune_layers_info["block_start"]]
        second_block = [prune_layers_info["block_end"], args.num_layers]
    elif args.method == "prune-one":
        layer = args.prune_layer
        assert layer is not None
        first_block = [0, layer]
        second_block = [layer + 1, args.num_layers]

    blocks = [first_block, second_block]

    with open(CONFIG_YML, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
        for i, slice_item in enumerate(config.get("slices", [])):
            for model_info in slice_item.get("sources", []):
                model_info["model"]["model"]["path"] = args.model_path
                model_info["model"]["model"]["revision"] = args.revision
                model_info["layer_range"] = blocks[i]
        print(config)
        merge_config = MergeConfiguration.model_validate(config)
        print(merge_config)

    file_name = (
        str(args.prune_layer)
        if args.method == "prune-one"
        else str(args.layers_to_skip)
    )

    run_merge(
        merge_config,
        out_path=os.path.join(
            OUTPUT_PATH,
            os.path.basename(args.model_path),
            args.method,
            file_name,
        ),
        options=MergeOptions(
            lora_merge_cache=LORA_MERGE_CACHE,
            cuda=torch.cuda.is_available(),
            copy_tokenizer=COPY_TOKENIZER,
            lazy_unpickle=LAZY_UNPICKLE,
            low_cpu_memory=LOW_CPU_MEMORY,
        ),
    )
    print("Done!")
