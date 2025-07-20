import glob
import os

import torch
from datasets import ClassLabel, Dataset, Features, Image
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor


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


image_root = "./data/imagenet-1k/val"
dataset = get_local_dataset(image_root)

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)


def transform(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    pixel_values = processor(images, return_tensors="pt")["pixel_values"]
    return {"pixel_values": pixel_values, "label": batch["label"]}


dataset.set_transform(transform)


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)  # type: ignore

# Test a batch
for batch in dataloader:
    print(batch["pixel_values"].shape)
    print(batch["labels"])
    break
