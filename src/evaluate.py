import json
import math
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config, load_cfg
from data import get_dataloader

main_config = "configs/config.yaml"
torch.cuda.empty_cache()


def load_model_and_tokenizer(config: Config):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        revision=config.revision,
    ).to(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.revision)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def evaluate_pico(model, tokenizer, dataloader, config: Config):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating PICO"):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            if tokenizer.pad_token_id is not None:
                num_tokens = (input_ids != tokenizer.pad_token_id).sum().item()
            else:
                num_tokens = input_ids.numel()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    output_info = {
        "model_path": str(config.model_path),
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "average_loss": avg_loss,
    }

    if hasattr(config, "model_name") and config.model_name:
        model_name = config.model_name
    else:
        model_name = os.path.basename(os.path.dirname(config.model_path))

    print(f"Model path: {config.model_path}")
    if config.method == "prune-one":
        assert config.prune_layer is not None
        print(f"Pruning layer: {config.prune_layer}")
        output_info.update({"prune_layer": config.prune_layer})
        output_file = f"results/pico/prune-one/{model_name}_{config.prune_layer}.json"
    else:
        print(f"Number of layers to skip: {config.num_layers_to_skip}")
        output_info.update({"num_layers_to_skip": config.num_layers_to_skip})
        if config.num_layers_to_skip > 0:
            output_file = f"results/pico/{config.method}/{model_name}_{config.num_layers_to_skip}.json"
        else:
            output_file = f"results/pico/{model_name}.json"
    print(f"Perplexity: {perplexity:.4f}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(output_info, f, indent=2)

    return perplexity


def _get_pruned_model_path(config: Config) -> Path:
    if config.method == "prune-one":
        assert config.prune_layer is not None
        return Path(f"merged/{os.path.basename(config.model_path)}/prune-one/{config.prune_layer}")
    else:
        return Path(f"merged/{os.path.basename(config.model_path)}/{config.method}/{config.num_layers_to_skip}")


if __name__ == "__main__":
    config = load_cfg(main_config)
    config.model_path = _get_pruned_model_path(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(config)

    dataloader = get_dataloader(config)
    if config.dataset_name == "pico-lm/pretokenized-dolma":
        evaluate_pico(model, tokenizer, dataloader, config)
