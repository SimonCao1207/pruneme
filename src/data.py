import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

from config import Config


class PicoDataset(IterableDataset):
    def __init__(self, config: Config):
        self.args = config
        self.dataset = load_dataset(config.dataset_name, split=config.split, streaming=True)
        if config.dataset_size:
            self.dataset = self.dataset.take(config.dataset_size)

    def __iter__(self):
        yield from self.dataset


def collate_fn(batch, args):
    if args.dataset_name == "pico-lm/pretokenized-dolma":
        input_ids_list = [torch.tensor(x["input_ids"]) for x in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=0)
        return {
            "input_ids": input_ids,  # (batch_size, max_seq_len)
        }


def get_dataloader(config, collate_fn=collate_fn):
    if config.dataset_name == "pico-lm/pretokenized-dolma":
        torch_dataset = PicoDataset(config)
        dataloader = DataLoader(
            torch_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=lambda batch: collate_fn(batch, config),
            num_workers=0,
        )
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset_name}")
    return dataloader
