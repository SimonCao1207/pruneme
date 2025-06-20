import csv
import logging
import os
from parser import parse_args
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from data import get_dataloader
from evaluate import load_model_and_tokenizer

logging.basicConfig(level=logging.INFO)

# Set seed
torch.manual_seed(42)
np.random.seed(42)


def angular_distance(x_l, x_l_plus_n) -> torch.Tensor:
    """Compute the angular distance between layer output tokens."""
    cosine_similarity = compute_cosine_similarity(x_l, x_l_plus_n)
    return torch.acos(cosine_similarity.clamp(min=-1, max=1)) / torch.pi


def compute_cosine_similarity(x_l, x_l_plus_n) -> torch.Tensor:
    """Compute the cosine similarity between layer output tokens."""
    x_l_norm = x_l / torch.norm(x_l, dim=-1, keepdim=True)
    x_l_plus_n_norm = x_l_plus_n / torch.norm(x_l_plus_n, dim=-1, keepdim=True)
    return (x_l_norm * x_l_plus_n_norm).sum(-1)


def compute_block_distances(
    hidden_states: List[torch.Tensor], layers_to_skip: int
) -> List[float]:
    """Compute and return angular distances for each block of layers."""
    distances = []
    num_layers = len(hidden_states)
    for l in range(num_layers - layers_to_skip):
        block_distance = (
            angular_distance(hidden_states[l], hidden_states[l + layers_to_skip])
            .mean()
            .item()
        )
        distances.append(block_distance)
    return distances


def layer_importance(
    hidden_states: List[torch.Tensor], attention_mask: torch.Tensor
) -> List[float]:
    """
    Calculate the importance of each layer based on the mean angular distance
    between the hidden states of the last token before and after each layer.

    Returns a list of importances (higher distance = higher importance).
    """
    last_non_padded_hidden_states = get_last_non_padded_tokens(
        hidden_states, attention_mask
    )
    # Remove first layer (embedding layer)
    last_non_padded_hidden_states = last_non_padded_hidden_states[1:]
    num_layers = len(last_non_padded_hidden_states)
    importances = []
    for l in range(num_layers - 1):
        dist = (
            angular_distance(
                last_non_padded_hidden_states[l], last_non_padded_hidden_states[l + 1]
            )
            .mean()
            .item()
        )
        importances.append(dist)
    return importances


def get_last_non_padded_tokens(hidden_states, attention_mask) -> List[torch.Tensor]:
    """Get last non-padded tokens for each layer."""
    last_non_padded_hidden_states = []
    for layer in hidden_states:
        batch_size, _, _ = layer.size()
        batch_last_tokens = []
        for batch in range(batch_size):
            if attention_mask:
                last_non_pad_index = (
                    attention_mask[batch].nonzero(as_tuple=True)[0].max()
                )
            else:
                last_non_pad_index = layer[batch].size(0) - 1
            last_token = layer[batch, last_non_pad_index, :]
            batch_last_tokens.append(last_token.unsqueeze(0))
        last_non_padded_hidden_states.append(torch.cat(batch_last_tokens, dim=0))
    return last_non_padded_hidden_states


def compute_and_save_layer_distances(
    model,
    tokenizer,
    dataloader,
    args,
    compute_block_distances,
):
    num_layers = model.config.num_hidden_layers
    num_dropped_layers = args.layers_to_skip

    # Initialize a list to store distances for each block across the dataset
    all_distances = [[] for _ in range(num_layers - num_dropped_layers)]

    for batch in tqdm(dataloader, desc="Processing batches"):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=args.max_length,
            truncation=True,
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        attention_mask = inputs["attention_mask"]
        hidden_states = outputs.hidden_states
        last_non_padded_hidden_states = get_last_non_padded_tokens(
            hidden_states, attention_mask
        )

        # Remove the first element to account for the input layer not being considered a model hidden layer
        # This adjustment is necessary for analyses focusing on the model's internal transformations
        last_non_padded_hidden_states = last_non_padded_hidden_states[1:]

        # Ensure that the length of last_non_padded_hidden_states matches the number of model hidden layers
        assert len(last_non_padded_hidden_states) == model.config.num_hidden_layers, (
            "Length of last_non_padded_hidden_states  \
        does not match expected number of hidden layers."
        )

        # Compute distances and append to all_distances
        distances = compute_block_distances(
            last_non_padded_hidden_states, num_dropped_layers
        )
        for i, distance in enumerate(distances):
            all_distances[i].append(distance)

    # Calculate average distances for each block
    average_distances = [np.mean(block_distances) for block_distances in all_distances]

    min_distance = float("inf")
    min_distance_layer = 0

    model_name = os.path.basename(args.model_path)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(f"outputs/{model_name}", exist_ok=True)
    with open(
        f"outputs/{model_name}/prune_{num_dropped_layers}_layers.csv",
        "w",
        newline="",
    ) as csvfile:
        fieldnames = ["block_start", "block_end", "average_distance"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, avg_dist in enumerate(average_distances):
            writer.writerow(
                {
                    "block_start": i,
                    "block_end": i + num_dropped_layers,
                    "average_distance": avg_dist,
                }
            )

            if avg_dist < min_distance:
                min_distance = avg_dist
                min_distance_layer = i

    # Log the layer with the minimum average distance
    logging.info(
        f"Layer {min_distance_layer} to {min_distance_layer + num_dropped_layers} has the minimum average distance of {min_distance}."
    )
    logging.info(
        f"Consider prunning layer {min_distance_layer} to {min_distance_layer + num_dropped_layers - 1}"
    )
    logging.info(
        f"Layer distances written to {model_name}_drop_{num_dropped_layers}_layers.csv"
    )


def compute_and_save_layer_importance(
    model,
    tokenizer,
    dataloader,
    args,
):
    """Compute and save layer importance based on angular distances."""
    all_importances = []

    for batch in tqdm(dataloader, desc="Processing batches"):
        if args.dataset_name == "pico-lm/pretokenized-dolma":
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
        else:
            input_ids = tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                max_length=args.max_length,
                truncation=True,
            ).to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states

        importances = layer_importance(hidden_states, attention_mask)
        all_importances.append(importances)

    average_importances = np.mean(all_importances, axis=0)

    model_name = os.path.basename(args.model_path)
    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{model_name}_layer_importance.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Layer", "Importance"])
        for i, importance in enumerate(average_importances):
            writer.writerow([i + 2, importance])  # 1-based indexing for layers

    logging.info(f"Layer importance saved to {model_name}_layer_importance.csv")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model_and_tokenizer(args, device)
    model.eval()

    dataloader = get_dataloader(args)

    # compute_and_save_layer_distances(
    #     model,
    #     tokenizer,
    #     dataloader,
    #     args,
    #     compute_block_distances,
    # )
    compute_and_save_layer_importance(
        model,
        tokenizer,
        dataloader,
        args,
    )


if __name__ == "__main__":
    main(parse_args())
