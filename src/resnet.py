import argparse
import glob
import json
import logging
import os

import numpy as np
import torch
from datasets import ClassLabel, Dataset, Features, Image
from einops import reduce
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, ResNetForImageClassification

from utils import angular_distance

DEBUG = False


def get_local_dataset(image_root: str):
    image_paths = glob.glob(os.path.join(image_root, "*", "*.JPEG"))
    labels = sorted(os.listdir(image_root))
    label2id = {name: idx for idx, name in enumerate(labels)}

    data = {"image": [], "label": []}

    for path in image_paths:
        class_name = os.path.basename(os.path.dirname(path))
        data["image"].append({"path": path})
        data["label"].append(label2id[class_name])

    features = Features({"image": Image(), "label": ClassLabel(names=labels)})

    return Dataset.from_dict(data, features=features)


def transform(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    pixel_values = processor(images, return_tensors="pt")["pixel_values"]
    return {"pixel_values": pixel_values, "label": batch["label"]}


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def get_layer_representation(hidden_states: list[torch.Tensor]):
    """Extract layer representations from hidden states."""
    layer_representations = []
    for layer in hidden_states:
        assert len(layer.shape) == 4
        # For vision models, the shape is (batch_size, channels, height, width)
        pooled = reduce(layer, "b c h w -> b c", "mean")  # just simple global average pooling here
        layer_representations.append(pooled)
    return layer_representations


def get_resnet_layer_info(model):
    """
    Get detailed information about ResNet layers and their groupings.
    Returns layer groups that can be safely compared (same dimensions).
    """
    layers = []
    layer_info = []

    for stage_idx, stage in enumerate(model.resnet.encoder.stages):
        stage_layers = []
        for layer_idx, layer in enumerate(stage.layers):
            layers.append(layer)
            layer_info.append(
                {"stage": stage_idx, "layer_in_stage": layer_idx, "global_index": len(layers) - 1, "layer": layer}
            )
            stage_layers.append(len(layers) - 1)

        print(f"Stage {stage_idx}: layers {stage_layers[0]} to {stage_layers[-1]} ({len(stage_layers)} layers)")

    return layers, layer_info


def compute_safe_block_distances(
    hidden_states: list[Tensor], layer_info: list[dict], num_layers_to_skip: int
) -> list[float]:
    # return a list distances size num_layers - num_layers_to_skip

    num_layers = len(hidden_states)
    num_layers_to_skip = min(num_layers_to_skip, num_layers - 1)

    assert num_layers - num_layers_to_skip > 0, "Not enough layers to compute distances after skipping."
    block_distances = []

    for layer_idx in range(num_layers - num_layers_to_skip):
        target_idx = layer_idx + num_layers_to_skip

        current_stage = layer_info[layer_idx]["stage"]
        target_stage = layer_info[target_idx]["stage"]
        assert current_stage == target_stage, "Cannot compare layers from different stages."
        block_distance = angular_distance(hidden_states[layer_idx], hidden_states[target_idx]).mean().item()
        block_distances.append(block_distance)
        if DEBUG:
            logging.debug(
                f"Intra-stage: Layer {layer_idx} -> {target_idx} (Stage {current_stage}): {block_distance:.6f}"
            )
    return block_distances


def process_batch_distances(hidden_states: list[Tensor], layer_info: list[dict], num_layers_to_skip: int) -> dict:
    layer_representations = get_layer_representation(hidden_states)

    stages_data = {}
    for info in layer_info:
        stage_idx = info["stage"]
        if stage_idx not in stages_data:
            stages_data[stage_idx] = {"representations": [], "info": []}
        global_index = info["global_index"]
        stages_data[stage_idx]["representations"].append(layer_representations[global_index])
        stages_data[stage_idx]["info"].append(info)

    stage_distances = {}
    for stage_idx, stage_data in stages_data.items():
        distances = compute_safe_block_distances(stage_data["representations"], stage_data["info"], num_layers_to_skip)
        stage_distances[stage_idx] = distances
    return stage_distances


def aggregate_distances(all_stage_distances: dict) -> dict:
    """
    Aggregate distances for each stage.
    Returns a dictionary with stage indices as keys and average distances as values.
    """
    average_distances = {}
    for stage_idx, distances in all_stage_distances.items():
        if distances:
            average_distance = np.mean(distances, axis=0)
            average_distances[stage_idx] = average_distance
    return average_distances


def save_distances_to_file(distances: dict, filename: str):
    for stage_idx, stage_distances in distances.items():
        if isinstance(stage_distances, np.ndarray):
            distances[stage_idx] = stage_distances.tolist()
    with open(filename, "w") as f:
        json.dump(distances, f, indent=4)
    logging.info(f"Distances saved to {filename}")


if __name__ == "__main__":
    image_root = "./data/imagenet-1k/val"
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()
    num_layers_to_skip = args.n
    dataset = get_local_dataset(image_root)

    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)
    dataset.set_transform(transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)  # type: ignore
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(device)

    model.eval()

    layers, layer_info = get_resnet_layer_info(model)

    num_layers = len(layers)
    logging.info(f"Number of layers in the ResNet model: {num_layers}")

    num_stages = len(model.resnet.encoder.stages)  # 4 stages
    all_stage_distances = {stage_idx: [] for stage_idx in range(num_stages)}
    num_tests = 9

    for batch in tqdm(dataloader, desc="Processing batches"):
        if num_tests <= 0 and DEBUG:
            break
        pixel_values = batch["pixel_values"].to(model.device)

        with torch.no_grad():
            # Forward pass through embedder
            x = model.resnet.embedder(pixel_values)

            # Collect hidden states from each layer
            hidden_states = []
            for layer in layers:
                x = layer(x)
                hidden_states.append(x.clone())

        batch_stage_distances = process_batch_distances(
            hidden_states, layer_info, num_layers_to_skip
        )  # return a dict of stage_idx -> list of distances for that stage

        for stage_idx, distances in batch_stage_distances.items():
            all_stage_distances[stage_idx].append(distances)

        num_tests -= 1

    average_distances = aggregate_distances(all_stage_distances)

    for stage_idx, distances in average_distances.items():
        logging.info(f"Stage {stage_idx} average distances: {average_distances[stage_idx]}")

    save_path = f"results/resnet/{num_layers_to_skip}"
    os.makedirs("results/resnet", exist_ok=True)
    save_distances_to_file(average_distances, f"{save_path}.json")
