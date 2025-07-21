import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import Config

logging.basicConfig(level=logging.INFO)


def print_config(config: Config) -> None:
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


def load_model_and_tokenizer(config: Config):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        revision=config.revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.revision)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def angular_distance(x_l, x_l_plus_n) -> torch.Tensor:
    """Compute the angular distance between layer output tokens."""
    cosine_similarity = compute_cosine_similarity(x_l, x_l_plus_n)
    return torch.acos(cosine_similarity.clamp(min=-1, max=1)) / torch.pi


def compute_cosine_similarity(x_l, x_l_plus_n) -> torch.Tensor:
    """Compute the cosine similarity between layer output tokens."""
    if x_l.shape[-1] != x_l_plus_n.shape[-1]:
        return torch.tensor(0.0, device=x_l.device)
    x_l_norm = x_l / torch.norm(x_l, dim=-1, keepdim=True)
    x_l_plus_n_norm = x_l_plus_n / torch.norm(x_l_plus_n, dim=-1, keepdim=True)
    return (x_l_norm * x_l_plus_n_norm).sum(-1)
