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
