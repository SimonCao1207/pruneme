import json
import math
import os
from parser import parse_args

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import get_dataloader

torch.cuda.empty_cache()


def load_model_and_tokenizer(args, device):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        revision=args.revision,
    )
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=args.revision)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def evaluate_mmlu(model, tokenizer, dataloader, args):
    model.eval()
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
        output_file = (
            f"results/mmlu/{args.method}/{model_name}_{args.layers_to_skip}.json"
        )
    else:
        output_file = f"results/mmlu/{model_name}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_info, f, indent=2)


def evaluate_pico(model, tokenizer, dataloader, args):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating PICO"):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)

            labels = input_ids.clone()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            if tokenizer.pad_token_id is not None:
                num_tokens = (labels != tokenizer.pad_token_id).sum().item()
            else:
                num_tokens = labels.numel()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    # Prepare output dictionary
    output_info = {
        "model_path": args.model_path,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "average_loss": avg_loss,
    }

    # Determine model name for filename
    if hasattr(args, "model_name") and args.model_name:
        model_name = args.model_name
    else:
        model_name = os.path.basename(os.path.dirname(args.model_path))

    print(f"Model path: {args.model_path}")
    if args.method == "prune-one":
        assert args.prune_layer is not None
        print(f"Pruning layer: {args.prune_layer}")
        output_info.update({"prune_layer": args.prune_layer})
        output_file = f"results/pico/prune-one/{model_name}_{args.prune_layer}.json"
    else:
        print(f"Layers to skip: {args.layers_to_skip}")
        output_info.update({"layers_to_skip": args.layers_to_skip})
        # Construct output file path similar to evaluate_mmlu
        if args.layers_to_skip > 0:
            output_file = (
                f"results/pico/{args.method}/{model_name}_{args.layers_to_skip}.json"
            )
        else:
            output_file = f"results/pico/{model_name}.json"
    print(f"Perplexity: {perplexity:.4f}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(output_info, f, indent=2)

    return perplexity


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(args, device)

    dataloader = get_dataloader(args)
    if args.dataset_name == "cais/mmlu":
        evaluate_mmlu(model, tokenizer, dataloader, args)
    elif args.dataset_name == "pico-lm/pretokenized-dolma":
        evaluate_pico(model, tokenizer, dataloader, args)
