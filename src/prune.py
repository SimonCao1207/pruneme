import copy
import os
import random

import pandas as pd
import torch
import yaml

from config import Config, load_cfg
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge
from utils import print_config


def set_random(seed=42):
    random.seed(seed)


def get_prune_layers(config: Config):
    model_name = os.path.basename(config.model_path)
    num_layers_to_skip = config.num_layers_to_skip
    file_name = f"outputs/{model_name}/prune_{num_layers_to_skip}_layers.csv"
    df = pd.read_csv(file_name)
    min_idx = df["average_distance"].idxmin()
    min_row = df.loc[min_idx]
    return {
        "block_start": int(min_row.at["block_start"]),
        "block_end": int(min_row.at["block_end"]),
        "average_distance": float(min_row.at["average_distance"]),
    }


def get_prune_blocks(config: Config):
    blocks = []
    if config.method == "normal":
        first_block = [0, config.num_layers - config.num_layers_to_skip]
        second_block = [
            config.num_layers - config.num_layers_to_skip,
            config.num_layers - config.num_layers_to_skip,
        ]  # empty block
        blocks.extend([first_block, second_block])
    elif config.method == "similarity-based":
        # layer_range is half open in mergekit (https://github.com/arcee-ai/mergekit/issues/206)
        # Skip layers block_start to block_end-1
        prune_layers_info = get_prune_layers(config)
        first_block = [0, prune_layers_info["block_start"]]
        second_block = [prune_layers_info["block_end"], config.num_layers]
        blocks.extend([first_block, second_block])
    elif config.method == "prune-one":
        layer = config.prune_layer
        assert layer is not None
        first_block = [0, layer]
        second_block = [layer + 1, config.num_layers]
        blocks.extend([first_block, second_block])
    elif config.method == "random":
        num_layers_to_skip = config.num_layers_to_skip
        layers = sorted(random.sample(range(1, config.num_layers - 1), k=num_layers_to_skip))
        print("Dropped layers", layers)
        prev_layer = 0
        for layer in layers:
            if prev_layer != layer:
                blocks.append([prev_layer, layer])
            prev_layer = layer + 1
        if prev_layer != config.num_layers:
            blocks.append([prev_layer, config.num_layers])
    elif config.method == "gradient-based": 
        pass
        

    return blocks


def customize_mergekit_config(config: Config, default_mergekit_config_path: str):
    blocks = get_prune_blocks(config)
    print("Blocks list: ", blocks)
    with open(default_mergekit_config_path, encoding="utf-8") as fp:
        config_data = yaml.safe_load(fp)
        if len(config_data["slices"]) < len(blocks):
            original_slices = config_data["slices"][0]
            needed_copies = len(blocks) - len(config_data["slices"])
            config_data["slices"].extend([copy.deepcopy(original_slices) for i in range(needed_copies)])

        assert len(config_data["slices"]) == len(blocks) or len(blocks) < 2, (
            "Number of slices must match number of blocks"
        )

        for i in range(len(blocks)):
            slice_item = config_data["slices"][i]
            for model_info in slice_item.get("sources", []):
                model_info["model"]["model"]["path"] = str(config.model_path)
                model_info["model"]["model"]["revision"] = config.revision
                model_info["layer_range"] = blocks[i]

    for item in config_data["slices"]:
        print(item.get("sources")[0]["layer_range"])

    return MergeConfiguration.model_validate(config_data)


def _get_output_file_name(config: Config):
    if config.method == "prune-one":
        assert config.prune_layer is not None
        return str(config.prune_layer)
    elif config.method == "random":
        return f"prune_{config.num_layers_to_skip}_seed_{config.seed}"
    else:
        return str(config.num_layers_to_skip)


if __name__ == "__main__":
    main_config = load_cfg()
    print_config(main_config)
    set_random(main_config.seed)
    default_mergekit_config_path = "configs/slice.yaml"
    customized_merge_config = customize_mergekit_config(main_config, default_mergekit_config_path)

    file_name = _get_output_file_name(main_config)
    run_merge(
        customized_merge_config,
        out_path=os.path.join(
            "merged",
            os.path.basename(main_config.model_path),
            main_config.method,
            file_name,
        ),
        options=MergeOptions(
            lora_merge_cache="/tmp",
            cuda=torch.cuda.is_available(),
            copy_tokenizer=True,
            lazy_unpickle=False,
            low_cpu_memory=False,
        ),
    )
