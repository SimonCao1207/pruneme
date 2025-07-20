import glob
import logging
import os

import torch
from datasets import ClassLabel, Dataset, Features, Image
from einops import reduce
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, ResNetForImageClassification

from main import compute_block_distances


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
        pooled = reduce(layer, "b c h w -> b c", "mean")  # Global average pooling
        layer_representations.append(pooled)
    return layer_representations


if __name__ == "__main__":
    image_root = "./data/imagenet-1k/val"
    num_layers_to_skip = 2
    dataset = get_local_dataset(image_root)

    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)
    dataset.set_transform(transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)  # type: ignore

    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    model.eval()

    layers = []
    for stage in model.resnet.encoder.stages:
        layers.extend(stage.layers)  # type: ignore

    num_layers = len(layers)
    logging.info(f"Number of layers in the ResNet model: {num_layers}")

    all_distances = [[] for _ in range(num_layers - num_layers_to_skip)]
    for batch in tqdm(dataloader, desc="Processing batches"):
        pixel_values = batch["pixel_values"].to(model.device)

        with torch.no_grad():
            # Forward pass through embedder
            x = model.resnet.embedder(pixel_values)

            # Collect hidden states from each layer
            hidden_states = []
            for layer in layers:
                x = layer(x)
                hidden_states.append(x.clone())

        layer_representations = get_layer_representation(hidden_states)
        print(f"Layer representations shape: {[layer.shape for layer in layer_representations]}")

        distances = compute_block_distances(layer_representations, num_layers_to_skip)
        for i, distance in enumerate(distances):
            all_distances[i].append(distance)

        logging.info(f"Sample distances for first {num_layers - num_layers_to_skip} layers: {all_distances[:5]}")
