import csv
import os

import torch
from tqdm import tqdm

from config import load_cfg
from data import get_dataloader
from utils import load_model_and_tokenizer, print_config

config = load_cfg()
print_config(config)
model, tokenizer = load_model_and_tokenizer(config)
dataloader = get_dataloader(config)
model.eval()

salience_dict = {}
target_params = {}
for k, param in model.named_parameters():
    if param.requires_grad and "weight" in k and "embed_tokens" not in k:
        target_params[k] = param
        salience_dict[k] = None  # Initialize with None for first allocation

for batch in tqdm(dataloader, desc=f"Processing batches {config.dataset_name}"):
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    for k, param in target_params.items():
        assert param.grad is not None
        salience = (param * param.grad).detach().cpu().float()
        if salience_dict[k] is None:
            salience_dict[k] = salience.clone()
        else:
            salience_dict[k] += salience
    model.zero_grad()

    torch.cuda.empty_cache()


def _get_layer_idx(k):
    # 'model.layers.i
    return ".".join(k.split(".")[:3])


model_name = config.model_name
os.makedirs(f"outputs/{model_name}", exist_ok=True)

result_csv_weight = f"outputs/{model_name}/weight_score.csv"
result_csv_block = f"outputs/{model_name}/block_score_all.csv"
result_csv_block_detail = f"outputs/{model_name}/block_score_all_detail.csv"
result_csv_block_sort = f"outputs/{model_name}/block_score_sorted.csv"
block_order_path = f"outputs/{model_name}/block_order.csv"
block_info = {}
with open(result_csv_weight, "w") as logfile:
    logwriter = csv.writer(logfile, delimiter=",")
    logwriter.writerow(["weight_name", "weight_score"])
    for k, param in model.named_parameters():
        if param.requires_grad and "weight" in k and "embed_tokens" not in k:
            layer_idx = _get_layer_idx(k)
            if "proj" in k or "lm_head" in k:  # output_dim x input_dim
                weight_imp = salience_dict[k].abs().pow(2).sum(1)  # [output_dim]
            elif "norm" in k:  # [output_dim]
                weight_imp = salience_dict[k].abs().pow(2)

            # Sum reduction
            weight_imp = weight_imp.sum(dim=0).item()
            logwriter.writerow([k, weight_imp])
            print([k, weight_imp])

            if layer_idx not in block_info.keys():
                block_info[layer_idx] = [weight_imp]
            else:
                block_info[layer_idx].append(weight_imp)

block_info_summary = {}
with open(result_csv_block, "w") as logfile, open(result_csv_block_detail, "w") as logfile_detail:
    logwriter = csv.writer(logfile, delimiter=",")
    logwriter.writerow(["block_name", "block_score"])
    logwriter_detail = csv.writer(logfile_detail, delimiter=",")
    logwriter_detail.writerow(["block_name", "all_weight_scores"])
    for k, v in block_info.items():
        print(k, v)
        logwriter_detail.writerow([k] + v)

        block_imp = torch.tensor(v).sum(dim=0).item()
        logwriter.writerow([k, block_imp])
        block_info_summary[k] = block_imp

for k in ["model.norm.weight", "lm_head.weight"]:
    if k in block_info_summary:
        del block_info_summary[k]

sorted_items = sorted(block_info_summary.items(), key=lambda x: x[1])
block_order = []
with open(result_csv_block_sort, "w") as logfile:
    logwriter = csv.writer(logfile, delimiter=",")
    logwriter.writerow(["rank", "block_name", "block_score", "block_index"])
    for rank, (key, value) in enumerate(sorted_items, start=1):
        logwriter.writerow([rank, key, value, key.split(".")[-1]])
        print([rank, key, value, key.split(".")[-1]])
        block_order.append(int(key.split(".")[-1]))

with open(block_order_path, "w") as logfile_order:
    logwriter_order = csv.writer(logfile_order, delimiter=",")
    logwriter_order.writerow(block_order)

print(f"=== block order removed: {block_order_path}")
print(block_order)
print(f"len: {len(block_order)}")
