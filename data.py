from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset


class CustomDataset(TorchDataset):
    def __init__(self, args):
        dataset = load_dataset(args.dataset_name, args.dataset_subset, split=args.split)
        if args.dataset_size and hasattr(dataset, "select"):
            dataset = dataset.select(range(min(args.dataset_size, len(dataset))))

        self.dataset = dataset
        self.args = args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.args.dataset_column:
            item = self.dataset[idx]
            return item[self.args.dataset_column]
        else:
            item = self.dataset[idx]
            question = item["question"]
            choices = item["choices"]
            answer = item["answer"]  # ground truth, e.g., "A"
            prompt = question + "\n"
            for i, choice in zip(["A", "B", "C", "D"], choices):
                prompt += f"{i}. {choice}\n"
            prompt += "Answer:"
            return {"prompt": prompt, "answer": answer}


def get_dataloader(args, collate_fn=None):
    torch_dataset = CustomDataset(args)
    dataloader = DataLoader(
        torch_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return dataloader
