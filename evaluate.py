from parser import parse_args

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import get_dataloader


def generate_text(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


def collate_fn(batch):
    return {
        "prompts": [x["prompt"] for x in batch],
        "answers": [x["answer"] for x in batch],
    }


def evaluate(model, tokenizer, dataloader):
    correct = 0
    total = 0

    for batch in tqdm(dataloader):
        prompts = batch["prompts"]
        answers = batch["answers"]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

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
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader = get_dataloader(args, collate_fn=collate_fn)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    evaluate(model, tokenizer, dataloader)
