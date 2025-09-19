import gzip
import json
import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

import datasets
import importlib_resources
import torch
import torch.nn.functional as F
from importlib_resources.abc import Traversable
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import Config
from src.model import LlamaForCausalLM

T = TypeVar("T")


logging.basicConfig(level=logging.INFO)


def print_config(config: Config | None) -> None:
    """Print current configuration settings."""
    logging.info("=" * 50)
    logging.info("CURRENT CONFIGURATION")
    logging.info("=" * 50)

    for attr in dir(config):
        if not attr.startswith("_"):
            value = getattr(config, attr)
            if not callable(value):
                logging.info(f"{attr}: {value}")

    logging.info("=" * 50)


def move_to_device(o: T, device: torch.device) -> T:
    if isinstance(o, torch.Tensor):
        return o.to(device)  # type: ignore[return-value]
    elif isinstance(o, dict):
        return {k: move_to_device(v, device) for k, v in o.items()}  # type: ignore[return-value]
    elif isinstance(o, list):
        return [move_to_device(x, device) for x in o]  # type: ignore[return-value]
    elif isinstance(o, tuple):
        return tuple(move_to_device(x, device) for x in o)  # type: ignore[return-value]
    else:
        return o  # type: ignore


def load_model_and_tokenizer(config: Config):
    if config.model_path.startswith("huzama"):
        model = LlamaForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.float32,
            revision=config.revision,
        ).to(config.device)
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-hf")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.float32,
            revision=config.revision,
        ).to(config.device)
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def angular_distance(x_l, x_l_plus_n) -> torch.Tensor:
    """Compute the angular distance between layer output tokens."""
    cosine_similarity = compute_cosine_similarity(x_l, x_l_plus_n)
    return torch.acos(cosine_similarity.clamp(min=-1, max=1)) / torch.pi


def _get_data_traversable(data_rel_path: str) -> Traversable:
    return importlib_resources.files("olmo_data").joinpath(data_rel_path)


@contextmanager
def get_data_path(data_rel_path: str) -> Generator[Path, None, None]:
    try:
        with importlib_resources.as_file(_get_data_traversable(data_rel_path)) as path:
            yield path
    finally:
        pass


def load_oe_eval_requests(path: str, name: str | None = None, split: str | None = None):
    """
    Loads an oe-eval request file from `olmo_data/oe_eval_tasks`.
    TODO: Add support from loading from S3 instead?
    """
    dataset_rel_path = os.path.join("oe_eval_tasks", path)
    if name is not None:
        dataset_rel_path = os.path.join(dataset_rel_path, name)
    with get_data_path(dataset_rel_path) as dataset_path:
        if not dataset_path.is_dir():
            raise NotADirectoryError(f"OE Eval dataset not found in directory {dataset_rel_path}")
        data_file = dataset_path / "requests.jsonl.gz"
        if not data_file.is_file():
            data_file = dataset_path / "requests.jsonl"
        if not data_file.is_file():
            raise FileNotFoundError(
                f"OE Eval dataset file requests-{split}.jsonl(.gz) missing in directory {dataset_rel_path}"
            )
        requests = []
        if data_file.suffix == ".gz":
            with gzip.open(data_file, "r") as file:
                for line in file:
                    requests.append(json.loads(line.decode("utf-8").strip()))
        else:
            with open(data_file) as file:
                for line2 in file:
                    requests.append(json.loads(line2.strip()))
        config = None
        config_file = dataset_path / "config.json"
        if config_file.is_file():
            with open(config_file) as file:
                config = json.load(file)
        return config, requests


def load_hf_dataset(path: str, name: str | None, split: str):
    """
    Loads a HuggingFace dataset. The dataset is assumed to be saved using
    `save_hf_dataset_to_disk` and located in `olmo_data/hf_datasets`.
    """
    dataset_rel_path = os.path.join("hf_datasets", path, name or "none", split)
    with get_data_path(dataset_rel_path) as dataset_path:
        if not dataset_path.is_dir():
            raise NotADirectoryError(
                f"HF dataset {path} name {name} split {split} not found in directory {dataset_rel_path}"
            )
        return datasets.load_from_disk(str(dataset_path))


def compute_cosine_similarity(x_l, x_l_plus_n) -> torch.Tensor:
    """Compute the cosine similarity between layer output tokens."""
    if x_l.shape[-1] != x_l_plus_n.shape[-1]:
        return torch.tensor(0.0, device=x_l.device)
    x_l_norm = x_l / torch.norm(x_l, dim=-1, keepdim=True)
    x_l_plus_n_norm = x_l_plus_n / torch.norm(x_l_plus_n, dim=-1, keepdim=True)
    return (x_l_norm * x_l_plus_n_norm).sum(-1)


def cross_entropy_loss(
    logits,
    labels,
    ignore_index: int = -100,
    reduction: str = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
):
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    if not compute_z_loss:
        return loss, None

    z_squared = logits.logsumexp(-1).pow(2)
    if reduction == "mean":
        z_squared = (z_squared * (labels != ignore_index)).mean()
    elif reduction == "sum":
        z_squared = (z_squared * (labels != ignore_index)).sum()

    z_loss = z_loss_multiplier * z_squared

    return loss, z_loss
