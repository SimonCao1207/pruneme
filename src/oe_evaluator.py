import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric, Metric

from oe_metric import ICLMetric
from utils import cross_entropy_loss, move_to_device

log = logging.getLogger(__name__)

__all__ = ["Evaluator"]


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


class EvaluatorType(StrEnum):
    downstream = "downstream"
    lm = "lm"


@dataclass
class Evaluator:
    label: str
    type: EvaluatorType
    eval_loader: DataLoader
    eval_metric: Metric | dict[str, Metric]
    subset_num_batches: int | None = None

    def reset_metrics(self) -> None:
        if isinstance(self.eval_metric, Metric):
            self.eval_metric.reset()
        else:
            for metric in self.eval_metric.values():
                metric.reset()

    def compute_metrics(self) -> dict[str, float]:
        if self.type == EvaluatorType.downstream:
            assert isinstance(self.eval_metric, ICLMetric)
            value = self.eval_metric.compute().item()
            key = f"eval/downstream/{self.label}_{self.eval_metric.metric_type}"
            if self.eval_metric.metric_type in ["ce_loss", "bpb"]:
                key = key.replace("/downstream/", f"/downstream_{self.eval_metric.metric_type}/")
            return {key: value}
        elif self.type == EvaluatorType.lm:
            # Metric(s) = cross entropy loss
            metrics: dict[str, Metric]
            if isinstance(self.eval_metric, Metric):
                metrics = {self.label: self.eval_metric}
            else:
                metrics = self.eval_metric
            out = {}
            for label in sorted(metrics.keys()):
                metric = metrics[label]
                assert isinstance(metric, MeanMetric)
                if metric.weight.item() == 0.0:  # type: ignore
                    # In this case we probably haven't called '.update()' on this metric yet,
                    # so we do so here with dummy values. Since we pass 0.0 in for weight this won't
                    # affect the final value.
                    # This can happen when the evaluator contains multiple tasks/datasets and we didn't
                    # get to this one within the current evaluation loop.
                    metric.update(0.0, 0.0)
                loss = metric.compute()
                if loss.isnan().item():
                    # This can happen when the evaluator contains multiple tasks/datasets and we didn't
                    # get to this one within the current evaluation loop.
                    continue
                else:
                    out[f"eval/{label}/CrossEntropyLoss"] = loss.item()
                    out[f"eval/{label}/Perplexity"] = torch.exp(loss).item()
            return out
        else:
            raise ValueError(f"Unexpected evaluator type '{self.type}'")

    def update_metrics(
        self,
        batch: dict[str, Any],
        ce_loss: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        if self.type == EvaluatorType.downstream:
            assert isinstance(self.eval_metric, ICLMetric)
            self.eval_metric.update(batch, logits)  # type: ignore
        elif self.type == EvaluatorType.lm:
            # Metric(s) = cross entropy loss
            for metadata, instance_loss in zip(batch["metadata"], ce_loss):
                if isinstance(self.eval_metric, dict):
                    metric = self.eval_metric[metadata["label"]]
                else:
                    metric = self.eval_metric
                metric.update(instance_loss)
        else:
            raise ValueError(f"Unexpected evaluator type '{self.type}'")


class EvaluateLM:
    def __init__(self, model: torch.nn.Module, evaluator: Evaluator, device: torch.device) -> None:
        self.model = model
        self.evaluator = evaluator
        self.loss_fn = cross_entropy_loss
        self.device = device
        self.console_log_interval = 1  # Set to 1 to log every eval step, 0 otherwise

    def get_labels(self, batch: dict[str, Any]) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, label_mask, attention_mask, instance_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
            batch.get("instance_mask"),
        )
        if label_mask is not None:
            labels.masked_fill_(~label_mask, -100)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, -100)
        if instance_mask is not None:
            labels.masked_fill_(~instance_mask.unsqueeze(-1), value=-100)
        return labels[..., 1:].contiguous()

    def model_forward(
        self, batch: dict[str, Any], loss_reduction: str = "mean", compute_z_loss: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        # shape: (batch_size, seq_len, vocab_size)
        logits = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            attention_bias=batch.get("attention_bias"),
            doc_lens=batch.get("doc_lens"),
            max_doc_lens=batch.get("max_doc_lens"),
        ).logits
        logits_for_loss = logits[..., :-1, :].contiguous()
        # shape: (batch_size * seq_len, vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
        # shape: (batch_size, seq_len)
        labels = self.get_labels(batch)
        # shape: (batch_size * seq_len,)
        labels = labels.view(-1)
        ce_loss, z_loss = self.loss_fn(
            logits_for_loss, labels, ignore_index=-100, reduction=loss_reduction, compute_z_loss=compute_z_loss
        )
        if loss_reduction == "none":
            # Reshape (batch_size * seq_len,) -> (batch_size, seq_len)
            ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1)
            if z_loss is not None:
                z_loss = z_loss.view(batch["input_ids"].shape[0], -1)
        return ce_loss, z_loss, logits

    def eval_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", enabled=True, dtype=torch.float32):
            ce_loss, _, logits = self.model_forward(batch, loss_reduction="none")
        return ce_loss.mean(dim=-1), logits

    def eval_step(self, batch: dict[str, Any], evaluator: Evaluator) -> None:
        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward pass.
        with torch.no_grad():  # NOTE: 'torch.inference_mode()' doesn't work with 'torch.compile()'.
            ce_loss, logits = self.eval_batch(batch)

        # Update metrics.
        evaluator.update_metrics(batch, ce_loss, logits)  # batch includes all keys that the downstream evaluation needs

    def log_metrics_to_console(self, prefix: str, metrics: dict[str, float]):
        def format_float(value: float) -> str:
            if value < 0.0001:
                return str(value)  # scientific notation
            elif value > 1000:
                return f"{int(value):,d}"
            elif value > 100:
                return f"{value:.1f}"
            elif value > 10:
                return f"{value:.2f}"
            elif value > 1:
                return f"{value:.3f}"
            else:
                return f"{value:.4f}"

        log.info(
            f"{prefix}\n"
            + "\n".join(
                [
                    f"    {name}={format_float(value)}"
                    for name, value in metrics.items()
                    if name == "optim/total_grad_norm"
                    or not name.startswith("optim/")  # there's too many optimizer metrics
                ]
            )
        )

    def eval(self):
        eval_metrics = {}
        log.info(f"Running evaluation for '{self.evaluator.label}'...")

        # Reset metrics.
        self.evaluator.reset_metrics()

        # Initialize data loader iterator.
        eval_batches = iter(self.evaluator.eval_loader)
        num_eval_batches = len(self.evaluator.eval_loader)

        # Run model over batches.
        for eval_step, eval_batch in enumerate(eval_batches):
            self.eval_step(eval_batch, self.evaluator)

            # Log to console.
            if eval_step + 1 == num_eval_batches or (eval_step + 1) % self.console_log_interval == 0:
                log.info(f"[eval_step={eval_step + 1}/{num_eval_batches}]")

        # Get final metrics.
        metrics = self.evaluator.compute_metrics()
        eval_metrics.update(metrics)
        self.log_metrics_to_console(f"{self.evaluator.label}", metrics)

        del eval_batches
        return eval_metrics
