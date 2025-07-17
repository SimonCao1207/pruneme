import torch

from src.config import Config
from src.evaluate import load_model_and_tokenizer
from src.model import LlamaForCausalLM

TEST_TEXT = "The quick brown fox jumps over the lazy dog."
REFERENCE_MODEL_PATH = "huzama/Vanilla-3.2-8L"


def _load_test_config() -> Config:
    config = Config()
    config.model_path = "merged/Vanilla-3.2-8L/prune-one/6"
    config.prune_layer = 6
    config.revision = "pico-epoch_0"
    return config


def _prepare_inputs(tokenizer, text: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    return input_ids, attention_mask


def _load_reference_model(config: Config) -> LlamaForCausalLM:
    reference_model = LlamaForCausalLM.from_pretrained(
        str(config.model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        revision=config.revision,
    )
    reference_model.eval()
    return reference_model


def _get_expected_outputs(
    reference_model: LlamaForCausalLM, input_ids: torch.Tensor, attention_mask: torch.Tensor, prune_layer: int | None
):
    with torch.no_grad():
        return reference_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            drop_layer_id=prune_layer,
        )


def _assert_outputs_match(expected_outputs, outputs) -> None:
    expected_outputs.logits = expected_outputs.logits.to(outputs.logits.device)
    # Add debugging information
    diff = torch.abs(expected_outputs.logits - outputs.logits)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)

    assert torch.allclose(
        expected_outputs.logits,
        outputs.logits,
        atol=1e-4,
        rtol=1e-3,
    ), f"Model outputs do not match. Max diff: {max_diff}, Mean diff: {mean_diff}"


def _cleanup_model(model) -> None:
    del model
    torch.cuda.empty_cache()


def test_prune_one_layer_simple():
    """Test that pruned model outputs match reference model with dropped layer."""
    config = _load_test_config()

    # Pruned model
    model, tokenizer = load_model_and_tokenizer(config)
    model.eval()

    input_ids, attention_mask = _prepare_inputs(tokenizer, TEST_TEXT, config.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    _cleanup_model(model)

    # Reference model
    config.model_path = REFERENCE_MODEL_PATH
    reference_model = _load_reference_model(config)
    expected_outputs = _get_expected_outputs(reference_model, input_ids, attention_mask, config.prune_layer)

    _assert_outputs_match(expected_outputs, outputs)
