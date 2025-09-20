import json
import logging
import math
import os

import pandas as pd
import torch
from tqdm import tqdm

from oe_downstream import EvaluateLM, build_evaluators
from src.config import Config, load_cfg
from src.data import get_dataloader
from src.utils import load_model_and_tokenizer, print_config

torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO)


def evaluate(model, tokenizer, dataloader, config: Config):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    prune_layers = get_prune_layers(config)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {config.dataset_name}"):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)

            if config.model_path.startswith("huzama"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    drop_layer_ids=prune_layers,
                    labels=input_ids,
                )
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            if tokenizer.pad_token_id is not None:
                num_tokens = (input_ids != tokenizer.pad_token_id).sum().item()
            else:
                num_tokens = input_ids.numel()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    output_info = {
        "model_path": str(config.model_path),
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "average_loss": avg_loss,
    }

    print(f"Model path: {config.model_path}")
    out_dir = f"results/{config.method}/{config.dataset_name}"
    output_info.update({"prune_layers": prune_layers})
    output_file = f"{out_dir}/{config.model_name.split('-')[0]}_{'_'.join(map(str, prune_layers))}.json"

    print(f"Perplexity: {perplexity:.4f}")

    save_results(output_file, output_info)

    return perplexity


def save_results(output_file, output_info):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_info, f, indent=2)
    logging.info(f"Results saved to {output_file}")


def get_prune_layers(config: Config) -> list[int]:
    dir_name = f"outputs/{config.model_name}/{config.method}/{config.dataset_name}"
    num_skips = config.num_layers_to_skip
    if config.method == "similarity-based":
        csv_file = f"{dir_name}/similarity_matrix.csv"
        df = pd.read_csv(csv_file, index_col=0)
        prune_layers = df.idxmax(axis=1).to_list()
        if num_skips < len(prune_layers) and num_skips > 0:
            prune_layers = prune_layers[:num_skips]
            prune_layers = [int(x) for x in prune_layers]
        else:
            assert False

    elif config.method == "taylor" or config.method == "magnitude":
        csv_file = f"{dir_name}/block_order_L1.csv"
        df = pd.read_csv(csv_file, header=None)
        prune_layers = df.iloc[0].dropna().astype(int).tolist()
        if num_skips < len(prune_layers) and num_skips > 0:
            prune_layers = prune_layers[:num_skips]
        else:
            assert True
    elif config.method == "prune-last":
        num_layers = config.num_layers
        prune_layers = list(range(num_layers))
        num_layer_to_skip = config.num_layers_to_skip
        prune_layers = prune_layers[: num_layers - num_layer_to_skip]
    elif config.method == "gap":
        prune_layers = [13, 12, 14, 11, 10, 8, 7, 9]
        prune_layers = prune_layers[:num_skips]
    return prune_layers


@torch.no_grad()
def evaluate_downstream(model, tokenizer, config: Config, device: torch.device, full=False):
    if full:
        prune_layers = []
    else:
        prune_layers = get_prune_layers(config)
        assert len(prune_layers) < config.num_layers and len(prune_layers) > 0
    logging.info(f"Pruning layers: {prune_layers}")
    evaluators = build_evaluators(config, tokenizer, device)
    eval_lm = EvaluateLM(model=model, evaluators=evaluators, device=device, prune_layers=prune_layers)
    eval_metrics = eval_lm.eval()
    logging.info(f"Downstream evaluation results: {eval_metrics}")
    out_dir = f"results/{config.method}/{config.dataset_name}"
    output_file = f"{out_dir}/{config.model_name.split('-')[0]}_{'_'.join(map(str, prune_layers))}.json"
    save_results(output_file, eval_metrics)


if __name__ == "__main__":
    config = load_cfg()
    print_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(config)
    model = torch.compile(model)
    dataloader = get_dataloader(config)
    if config.dataset_name in ["pico", "wikitext"]:
        evaluate(model, tokenizer, dataloader, config)
    else:
        evaluate_downstream(model, tokenizer, config, device)
