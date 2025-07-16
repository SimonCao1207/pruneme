# config.py
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class Config:
    model_path: Path
    dataset_name: str
    batch_size: int
    max_length: int
    model_name: str | None = None
    dataset_column: str | None = None
    dataset_size: int | None = None
    dataset_subset: str = "eval"
    split: str = "train"
    device: str = "cuda"
    revision: str = "main"


@dataclass
class MergeConfig(Config):
    method: str = "normal"  # normal | similarity-based | prune-one
    prune_layer: int | None = None


def load_cfg(yaml_path: str | Path) -> Config:
    """Read the YAML file and convert it into a typed Config object."""
    cfg_dict = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
    return Config(**cfg_dict)
