import json
import os
from parser import parse_args

import torch
from tqdm import tqdm

from data import get_dataloader
from main import load_model_and_tokenizer


def collate_fn(batch):
    return {
        "prompts": [x["prompt"] for x in batch],
        "answers": [x["answer"] for x in batch],
    }


def evaluate(model, tokenizer, dataloader, args):
    correct = 0
    total = 0

    for batch in tqdm(dataloader):
        prompts = batch["prompts"]
        answers = batch["answers"]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        for k in inputs:
            inputs[k] = inputs[k].to(model.device)

        options = ["A", "B", "C", "D"]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        predictions = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        for pred, ans in zip(predictions, answers):
            pred_letter = pred.strip()[0].upper() if pred.strip() else ""
            try:
                pred_idx = options.index(pred_letter)
            except ValueError:
                pred_idx = -1
            if pred_idx == ans:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0

    print(f"Model path: {args.model_path}")
    print(f"Layers to skip: {args.layers_to_skip}")
    print(f"Accuracy: {accuracy:.2%}")

    os.makedirs("results", exist_ok=True)
    output_info = {
        "model_path": args.model_path,
        "layers_to_skip": args.layers_to_skip,
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
    }
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = os.path.basename(os.path.dirname(args.model_path))

    if args.layers_to_skip > 0:
        output_file = f"results/{args.method}/{model_name}_{args.layers_to_skip}.json"
    else:
        output_file = f"results/{model_name}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_info, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader = get_dataloader(args, collate_fn=collate_fn)
    model, tokenizer = load_model_and_tokenizer(args)
    evaluate(model, tokenizer, dataloader, args)
