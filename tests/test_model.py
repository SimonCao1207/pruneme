import torch
from transformers import AutoTokenizer

from src.model import LlamaForCausalLM


def test_forward():
    model_path = "huzama/Vanilla-3.2-8L"
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        revision="pico-epoch_0",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision="pico-epoch_0")

    input = tokenizer("Hello world", return_tensors="pt").to(model.device)
    outputs1 = model(input_ids=input["input_ids"], drop_layer_id=7)
    outputs2 = model(input_ids=input["input_ids"], drop_layer_id=6)
    assert not torch.equal(outputs1.logits, outputs2.logits)
