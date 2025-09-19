import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from src.config import Config
from src.oe_downstream import OEEvalTask, label_to_task_map


class PicoDataset(IterableDataset):
    def __init__(self, config: Config):
        self.args = config
        self.dataset = load_dataset(config.dataset_name, split="train", streaming=True)
        val_size = 64 * 20
        self.dataset = self.dataset.take(val_size)

    def __iter__(self):
        yield from self.dataset


def collate_fn(batch, tokenizer):
    inputs = {}

    inputs["input_ids"] = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    inputs["attention_mask"] = (inputs["input_ids"] != tokenizer.pad_token_id).long()

    labels = inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    inputs["labels"] = labels

    inputs["input_ids"] = inputs["input_ids"][:, :-1]
    inputs["attention_mask"] = inputs["attention_mask"][:, :-1]
    inputs["labels"] = inputs["labels"][:, 1:]

    return inputs


def get_dataloader(config, tokenizer=None, collate_fn=collate_fn):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-hf")
    if config.dataset_name == "pico":
        torch_dataset = PicoDataset(config)
        dataloader = DataLoader(
            torch_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=lambda batch: collate_fn(batch, tokenizer),
            num_workers=0,
        )
    elif config.dataset_name == "wikitext":
        dataloader = _get_dataloader_wikitext(tokenizer, config.batch_size)
    else:
        task_kwargs = {}
        task_class = label_to_task_map[config.dataset_name]
        if isinstance(task_class, tuple):
            task_class, task_kwargs = task_class
        if task_class is OEEvalTask:
            ds_eval_dataset = task_class(tokenizer=tokenizer, model_ctx_len=config.max_length, **task_kwargs)
        else:
            ds_eval_dataset = task_class(tokenizer=tokenizer, **task_kwargs)
        dataloader = DataLoader(
            ds_eval_dataset,  # type: ignore
            batch_size=config.batch_size,
            collate_fn=ds_eval_dataset.collate_fn,
            num_workers=0,
        )
    return dataloader


def _get_dataloader_wikitext(tokenizer, batch_size=16, block_size=1024):
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
