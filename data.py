import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, IterableDataset


class PicoDataset(IterableDataset):
    def __init__(self, args):
        self.args = args
        self.dataset = load_dataset(args.dataset_name, split=args.split, streaming=True)
        if args.dataset_size:
            self.dataset = self.dataset.take(args.dataset_size)

    def __iter__(self):
        for item in self.dataset:
            yield item


class MMLUDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.dataset = load_dataset(
            args.dataset_name, split=args.split, streaming=False
        )
        if args.dataset_size and hasattr(self.dataset, "select"):
            self.dataset = self.dataset.select(
                range(min(args.dataset_size, len(self.dataset)))
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]  # e.g., "A"
        prompt = question + "\n"
        for i, choice in zip(["A", "B", "C", "D"], choices):
            prompt += f"{i}. {choice}\n"
        prompt += "Answer:"
        return {"prompt": prompt, "answer": answer}


def collate_fn(batch, args):
    if args.dataset_name == "pico-lm/pretokenized-dolma":
        input_ids_list = [torch.tensor(x["input_ids"]) for x in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=0
        )
        return {
            "input_ids": input_ids,  # (batch_size, max_seq_len)
        }
    elif args.dataset_name == "cais/mmlu":
        return {
            "prompts": [x["prompt"] for x in batch],
            "answers": [x["answer"] for x in batch],
        }


def get_dataloader(args, collate_fn=collate_fn):
    if args.dataset_name == "pico-lm/pretokenized-dolma":
        torch_dataset = PicoDataset(args)
        dataloader = DataLoader(
            torch_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=lambda batch: collate_fn(batch, args),
            num_workers=0,
        )
    elif args.dataset_name == "cais/mmlu":
        torch_dataset = MMLUDataset(args)
        dataloader = DataLoader(
            torch_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=lambda batch: collate_fn(batch, args),
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    return dataloader
