import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from src.config import Config


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
    elif config.dataset_name == "wikitext":
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-hf")
        dataloader = _get_dataloader_wikitext(tokenizer, config.batch_size)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset_name}")
    return dataloader


def _get_dataloader_wikitext(tokenizer, batch_size=6, block_size=1024):
    val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    val_dataset = val_dataset.filter(lambda example: len(example["text"]) > 0)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)

    tokenized_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
    )
    print(f"Dataset processed into {len(lm_dataset)} blocks of size {block_size}.")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    val_loader = torch.utils.data.DataLoader(lm_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    return val_loader


if __name__ == "__main__":
    config = Config()
    config.dataset_name = "wikitext"
    config.batch_size = 16
    dataloader = get_dataloader(config)
    batch = next(iter(dataloader))
    print(batch["input_ids"].shape)
