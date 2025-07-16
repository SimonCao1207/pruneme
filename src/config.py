# config.py
from dataclasses import dataclass
from pathlib import Path

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


def load_cfg(yaml_path: str | Path) -> Config:
    """Read the YAML file and convert it into a typed Config object."""
    cfg_dict = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
    return Config(**cfg_dict)
