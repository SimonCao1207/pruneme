import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.config import Config
from src.data import get_dataloader
from src.evaluate import load_model_and_tokenizer
from src.model import LlamaForCausalLM

FIXTURES_PATH = (Path(__file__).resolve().parent) / "fixtures"


def _load_test_config():
    """Load test configuration from fixture file."""
    with open(FIXTURES_PATH / "Vanilla-3.2-8L_6.json") as f:
        meta_data = json.load(f)

    config = Config()
    config.model_path = Path(meta_data["model_path"])
    config.prune_layer = meta_data["prune_layer"]
    return config


def _load_reference_model(config):
    """Load reference model and return it."""
    reference_model = LlamaForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        revision=config.revision,
    )
    reference_model.eval()
    return reference_model


def _get_expected_outputs(reference_model, input_ids, attention_mask, prune_layer):
    """Get expected outputs from reference model with layer dropping."""
    with torch.no_grad():
        expected_outputs = reference_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            drop_layer_id=prune_layer,
        )
    return expected_outputs


def _cleanup_and_get_model(reference_model, config):
    """Clean up reference model and load the pruned model."""
    del reference_model
    torch.cuda.empty_cache()

    model, _ = load_model_and_tokenizer(config)
    model.eval()
    return model


def _assert_outputs_match(expected_outputs, outputs):
    """Assert that model outputs match expected outputs."""
    expected_outputs.logits = expected_outputs.logits.to(outputs.logits.device)
    assert torch.allclose(
        expected_outputs.logits,
        outputs.logits,
        atol=1e-4,
        rtol=1e-3,
    )


def test_prune_one_layer():
    config = _load_test_config()
    assert config.model_path == Path("merged/Vanilla-3.2-8L/prune-one/6")

    dataloader = get_dataloader(config)
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(config.device)
    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(config.device)

    reference_model = _load_reference_model(config)
    expected_outputs = _get_expected_outputs(reference_model, input_ids, attention_mask, config.prune_layer)

    model = _cleanup_and_get_model(reference_model, config)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    _assert_outputs_match(expected_outputs, outputs)


def test_prune_one_layer_simple():
    config = _load_test_config()
    assert config.model_path == Path("merged/Vanilla-3.2-8L/prune-one/6")

    test_text = "The quick brown fox jumps over the lazy dog."
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(config.device)
    attention_mask = inputs["attention_mask"].to(config.device)

    reference_model = _load_reference_model(config)
    expected_outputs = _get_expected_outputs(reference_model, input_ids, attention_mask, config.prune_layer)

    model = _cleanup_and_get_model(reference_model, config)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    _assert_outputs_match(expected_outputs, outputs)
