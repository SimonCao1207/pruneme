# config.py
import argparse
from dataclasses import dataclass

from omegaconf import OmegaConf


@dataclass
class Config:
    # Model config
    model_path: str = "huzama/Vanilla-3.2-8L"
    model_name: str | None = None
    num_layers: int = 8

    # Dataset config
    dataset_name: str = "pico-lm/pretokenized-dolma"
    batch_size: int = 8
    max_length: int = 2048
    dataset_column: str | None = None
    dataset_size: int | None = None
    dataset_subset: str = "eval"
    split: str = "train"
    device: str = "cuda"
    revision: str = "main"

    # Merge config
    method: str = "normal"  # normal | similarity-based | prune-one
    num_layers_to_skip: int = 1
    prune_layer: int | None = None  # Only used if method is "prune-one"


def get_arg_parser():
    # Overwrite default config with command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--dataset_column", type=str)
    parser.add_argument("--dataset_size", type=int)
    parser.add_argument("--dataset_subset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--revision", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--num_layers_to_skip", type=int)
    parser.add_argument("--prune_layer", type=int)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser


def load_cfg():
    parser = get_arg_parser()
    args = parser.parse_args()

    yaml_cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_dotlist([f"{k}={v}" for k, v in vars(args).items() if k != "config" and v is not None])

    merged_cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
    cfg_dict = OmegaConf.to_container(merged_cfg, resolve=True)
    return Config(**cfg_dict)
