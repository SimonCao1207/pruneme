import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import load_cfg
from data import get_dataloader
from src.model import LlamaForCausalLM
from src.utils import load_model_and_tokenizer, print_config


def calculate_and_save_average_hidden_states(
    model: LlamaForCausalLM,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    output_path: str,
):
    """
    Calculates the average hidden state for each layer of the model over a dataset
    and saves the result to a file.
    """
    model.to(device)
    model.eval()
    num_layers = model.config.num_hidden_layers
    similarity_sums = {}
    token_counts = {}

    print(f"Starting hidden state accumulation on device: {device}")

    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader, desc="Averaging Hidden States")):
            input_ids = batch["input_ids"].to(device)
            if "attention_mask" in batch:
                attention_mask = batch["attention_mask"].to(device)
            else:
                attention_mask = None

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[1:]
            if attention_mask is None:
                valid_token_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=device)
            else:
                valid_token_mask = attention_mask.bool()
            num_valid_tokens = valid_token_mask.sum().item()

            for n in range(1, num_layers):
                for l in range(num_layers - n):
                    state1 = hidden_states[l][valid_token_mask]
                    state2 = hidden_states[l + n][valid_token_mask]

                    sims = F.cosine_similarity(state1, state2)

                    # Update accumulators
                    key = (l, n)
                    if key not in similarity_sums:
                        similarity_sums[key] = 0.0
                        token_counts[key] = 0
                    similarity_sums[key] += torch.sum(sims).item()
                    token_counts[key] += num_valid_tokens

    # --- Final Averaging and DataFrame Creation ---
    distance_matrix = {}  # {n: [dist_l0, dist_l1, ...]}
    for n in range(1, num_layers):
        distances_for_n = []
        for l in range(num_layers - n):
            key = (l, n)
            if key in token_counts and token_counts[key] > 0:
                avg_sim = similarity_sums[key] / token_counts[key]
                # Convert similarity to angular distance
                avg_sim = max(min(avg_sim, 1.0), -1.0)  # Clamp for stability
                angular_dist = np.arccos(avg_sim) / np.pi  # Normalize to [0, 1]
                distances_for_n.append(angular_dist)
            else:
                distances_for_n.append(np.nan)
        distance_matrix[n] = distances_for_n

    dist_df = pd.DataFrame.from_dict(distance_matrix, orient="index")
    dist_df.index.name = "Block Size (n)"
    dist_df.columns.name = "Initial Layer (l)"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dist_df.to_csv(output_path)
    print(f"Block-wise angular distance matrix saved to: {output_path}")

    return dist_df


if __name__ == "__main__":
    config = load_cfg()
    print_config(config)
    assert config.method == "similarity-based", "This script only supports 'similarity-based' method."
    model, tokenizer = load_model_and_tokenizer(config)
    dataloader = get_dataloader(config, tokenizer)
    output_dir = f"outputs/{config.model_name}/{config.method}/{config.dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/similarity_matrix.csv"
    calculate_and_save_average_hidden_states(model, dataloader, config.device, output_path)
