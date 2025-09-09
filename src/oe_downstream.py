import abc
import logging
from collections.abc import Sequence
from typing import Any

import datasets
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer as Tokenizer

from oe_evaluator import EvaluateLM, Evaluator, EvaluatorType
from oe_metric import ICLMetric
from src.config import load_cfg
from utils import load_hf_dataset, load_model_and_tokenizer, load_oe_eval_requests, print_config

log = logging.getLogger(__name__)

# Map from oe-eval metrics to metrics used here
METRIC_FROM_OE_EVAL = {
    "acc_raw": "acc",
    "acc_per_char": "len_norm",
    "acc_uncond": "pmi_dc",
    "logits_per_byte": "bpb",
}


class ICLMultiChoiceTaskDataset(metaclass=abc.ABCMeta):
    """Only supports zero-shot for now."""

    metric_type: str

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: str | Sequence[str] | None = None,
        model_ctx_len: int = 2048,
        split="validation",
        metric_type=None,  # Override default metric type
        prompts=[None],  # List of prompt variants to use
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.model_ctx_len = model_ctx_len
        self.prompts = prompts
        self.current_prompt = None
        if metric_type is not None:
            self.metric_type = metric_type
        self.log_instances = 0  # Set to > 0 to log the first few instances as a sanity check

        self.samples: list[dict[str, Any]] = []
        dataset_names: Sequence[str | None]
        if isinstance(dataset_name, str) or dataset_name is None:
            dataset_names = [dataset_name]
        else:
            dataset_names = dataset_name

        dataset_list = []
        for ds_name in dataset_names:
            dataset = load_hf_dataset(self.dataset_path, ds_name, split)
            dataset_list.append(dataset)
        self.dataset = datasets.concatenate_datasets(dataset_list)

        # prep examples
        self.prep_examples()

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def prep_examples(self):
        """Append doc_ids to each example so that they are processed together in the metric"""
        doc_id = 0
        for doc in self.dataset:
            for prompt in self.prompts:
                self.current_prompt = prompt
                # from EAI harness
                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                continuations = self.doc_to_continuations(doc)
                label_id = self.doc_to_label(doc)
                doc_text = self.doc_to_text(doc)
                ctx = self.token_encode(doc_text)
                dc = self.token_encode(self.doc_to_domain_conditional(doc))
                if self.log_instances > 0:
                    self.log_instances -= 1
                    ds_name = self.dataset_name
                    if isinstance(ds_name, list):
                        ds_name = ds_name[0]
                    log.info(
                        f"Sample doc from ({self.dataset_path}, {ds_name}, {self.current_prompt}):"
                        + f"\ndoc_text: {doc_text}\ncontinuations: {continuations}"
                    )

                for cont_id, continuation_str in enumerate(continuations):
                    cont_str_len = len(continuation_str) - 1  # continuation contain leading blank
                    cont_byte_len = len(continuation_str[1:].encode("utf-8"))
                    continuation = self.token_encode(continuation_str)

                    # query, remove last token from continuation, truncate from left is longer than model ctx length
                    query = ctx + continuation[:-1]
                    query = query[-self.model_ctx_len :]
                    # this will be different from len(ctx) when truncated by model_ctx_len
                    actual_ctx_len = len(query) - len(continuation) + 1

                    # get domain conditional query
                    # we don't expect this to be longer than self.model_ctx_len and it won't make sense to truncate from left
                    dc_query = dc + continuation[:-1]

                    # form a sample
                    self.samples.append(
                        {
                            "doc_id": doc_id,
                            "cont_id": cont_id,
                            "ctx": ctx,
                            "continuation": continuation,
                            "ctx_len": actual_ctx_len,
                            "dc_len": len(dc),
                            "cont_len": len(
                                continuation
                            ),  # even if query has last token removed, LM will output same cont len
                            "cont_str_len": cont_str_len,
                            "cont_byte_len": cont_byte_len,
                            "query": query,  # remove last token from continuation
                            "dc_query": dc_query,
                            "label_id": label_id,
                        }
                    )

                doc_id += 1

    def pad_tokens_until_max(self, tokens, max_len=2048):
        """truncate from left if len(tokens) > model_ctx_len, max_len is not considered then
        queries are already truncated at max length of model_ctx_len
        this acts as additional check for all types of sequences in the batch
        """
        if len(tokens) > self.model_ctx_len:
            return tokens[-self.model_ctx_len :]
        else:
            # pad to max_len, but check again if this padding exceeded self.model_ctx_len
            # this time truncate from right side of the sequence because additional padding caused len(tokens) > self.model_ctx_len
            tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))

            if len(tokens) > self.model_ctx_len:
                tokens = tokens[: self.model_ctx_len]

            return tokens

    def collate_fn(self, data):
        # pad to max length
        # 'ctx', 'continuation', 'query' can all have variable length
        max_ctx_len = 0
        max_cont_len = 0
        max_query_len = 0
        max_dc_query_len = 0

        for sample in data:
            if len(sample["ctx"]) > max_ctx_len:
                max_ctx_len = len(sample["ctx"])

            if len(sample["continuation"]) > max_cont_len:
                max_cont_len = len(sample["continuation"])

            if len(sample["query"]) > max_query_len:
                max_query_len = len(sample["query"])

            if len(sample["dc_query"]) > max_dc_query_len:
                max_dc_query_len = len(sample["dc_query"])

        doc_ids = []
        cont_ids = []
        ctxs = []
        continuations = []
        ctx_lens = []
        dc_lens = []
        cont_lens = []
        cont_str_lens = []
        cont_byte_lens = []
        queries = []
        dc_queries = []
        label_ids = []

        # pad according to max_lengths
        for sample in data:
            doc_ids.append(sample["doc_id"])
            cont_ids.append(sample["cont_id"])

            ctxs.append(torch.LongTensor(self.pad_tokens_until_max(sample["ctx"], max_len=max_ctx_len)))
            continuations.append(
                torch.LongTensor(self.pad_tokens_until_max(sample["continuation"], max_len=max_cont_len))
            )

            ctx_lens.append(sample["ctx_len"])
            dc_lens.append(sample["dc_len"])
            cont_lens.append(sample["cont_len"])
            cont_str_lens.append(sample["cont_str_len"])
            cont_byte_lens.append(sample["cont_byte_len"])

            queries.append(torch.LongTensor(self.pad_tokens_until_max(sample["query"], max_len=max_query_len)))
            dc_queries.append(torch.LongTensor(self.pad_tokens_until_max(sample["dc_query"], max_len=max_dc_query_len)))

            label_ids.append(sample["label_id"])

        batch = {
            "doc_id": torch.LongTensor(doc_ids),
            "cont_id": torch.LongTensor(cont_ids),
            "ctx": torch.stack(ctxs),
            "continuation": torch.stack(continuations),
            "ctx_len": torch.LongTensor(ctx_lens),
            "dc_len": torch.LongTensor(dc_lens),
            "cont_len": torch.LongTensor(cont_lens),  # since query has last token removed from continuation
            "cont_str_len": torch.LongTensor(cont_str_lens),
            "cont_byte_len": torch.LongTensor(cont_byte_lens),
            "input_ids": torch.stack(queries),
            "dc_input_ids": torch.stack(dc_queries),
        }

        if not isinstance(label_ids, str):
            batch["label_id"] = torch.LongTensor(label_ids)

        return batch

    def token_encode(self, string: str) -> list[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def token_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    @abc.abstractmethod
    def doc_to_text(self, doc) -> str:
        """Match EAI eval harness
        returns a single context string
        """
        raise NotImplementedError

    @abc.abstractmethod
    def doc_to_continuations(self, doc) -> list[str]:
        """Match EAI eval harness
        returns a list of continuations
        """
        raise NotImplementedError

    @abc.abstractmethod
    def doc_to_label(self, doc) -> int:
        """Match EAI eval harness
        returns continuation id which corresponds to true label
        """
        raise NotImplementedError

    def doc_to_domain_conditional(self, doc) -> str:
        """Provide string for domain conditional normalization
        by default its blank string, continuation normalized by prob conditioned on a blank
        """
        del doc
        return " "


class OEEvalTask(ICLMultiChoiceTaskDataset):
    """Generic class for OE evaluation tasks"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: str | Sequence[str] | None = None,
        model_ctx_len: int = 2048,
        split=None,
        metric_type=None,
        prompts=[None],  # List of prompt variants to use
    ):
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.model_ctx_len = model_ctx_len
        self.log_instances = 0  # Set to > 0 to log the first few instances as a sanity check

        self.samples: list[dict[str, Any]] = []
        dataset_names: Sequence[str | None]
        if isinstance(dataset_name, str) or dataset_name is None:
            dataset_names = [dataset_name]
        else:
            dataset_names = dataset_name

        requests_list = []
        configs = []
        for ds_name in dataset_names:
            config, requests = load_oe_eval_requests(self.dataset_path, ds_name, split)
            requests_list.append(requests)
            configs.append(config)
        if metric_type is not None:
            self.metric_type = metric_type
        else:
            # Use metric type from associated task config
            for config in configs:
                if config is not None:
                    metric_type_raw = config["task_config"].get("primary_metric")
                    if metric_type_raw is not None:
                        # acc, len_norm, pmi_dc
                        metric_type = METRIC_FROM_OE_EVAL[metric_type_raw]
                        if self.metric_type is not None and self.metric_type != metric_type:
                            raise ValueError(f"Conflicting metric types: {self.metric_type} and {metric_type}")
                        self.metric_type = metric_type
        self.dataset = requests_list

        # prep examples
        self.prep_examples()

    def prep_examples(self):
        current_doc_id_offset = 0
        max_doc_id = 0
        for requests in self.dataset:
            current_doc_id_offset += max_doc_id
            max_doc_id = 0  # Max doc id seen in this dataset
            for request in requests:
                doc = request["doc"]
                doc_id = request["doc_id"]
                if doc_id >= 1000000:
                    # Hacky implementation of unconditional requests in oe-eval
                    # Not supported here for now
                    continue
                if doc_id > max_doc_id:
                    max_doc_id = doc_id
                assert request["request_type"] == "loglikelihood", (
                    f"Unsupported request type: {request['request_type']}"
                )

                # from EAI harness
                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                request_dict = request["request"]
                continuation_str = request_dict["continuation"]
                label_id = request["label"]
                cont_id = request["idx"]
                if self.metric_type in ["ce_loss", "bpb"]:
                    if label_id != cont_id and not isinstance(label_id, str):
                        # Skip non-target continuations for ce_loss and bpb
                        continue
                    else:
                        # Treat as instance with just one continuation
                        cont_id = 0
                        label_id = 0
                doc_text = request_dict["context"]
                ctx = self.token_encode(doc_text)
                dc = self.token_encode(self.doc_to_domain_conditional(doc))
                if self.log_instances > 0:
                    self.log_instances -= 1
                    ds_name = self.dataset_name
                    if isinstance(ds_name, list):
                        ds_name = ds_name[0]
                    log.info(
                        f"Sample doc from ({self.dataset_path}, {ds_name}):"
                        + f"\ndoc_text: {doc_text}\ncontinuation: {continuation_str}"
                    )
                cont_str_len = len(continuation_str) - 1  # continuation contain leading blank
                cont_byte_len = len(continuation_str[1:].encode("utf-8"))
                continuation = self.token_encode(continuation_str)

                # query, remove last token from continuation, truncate from left is longer than model ctx length
                query = ctx + continuation[:-1]
                query = query[-self.model_ctx_len :]
                # this will be different from len(ctx) when truncated by model_ctx_len
                actual_ctx_len = len(query) - len(continuation) + 1

                # get domain conditional query
                # we don't expect this to be longer than self.model_ctx_len and it won't make sense to truncate from left
                dc_query = dc + continuation[:-1]

                # form a sample
                self.samples.append(
                    {
                        "doc_id": doc_id + current_doc_id_offset,
                        "cont_id": cont_id,
                        "ctx": ctx,
                        "continuation": continuation,
                        "ctx_len": actual_ctx_len,
                        "dc_len": len(dc),
                        "cont_len": len(
                            continuation
                        ),  # even if query has last token removed, LM will output same cont len
                        "cont_str_len": cont_str_len,
                        "cont_byte_len": cont_byte_len,
                        "query": query,  # remove last token from continuation
                        "dc_query": dc_query,
                        "label_id": label_id,
                    }
                )

    def doc_to_text(self, doc) -> str:
        raise NotImplementedError

    def doc_to_continuations(self, doc) -> list[str]:
        raise NotImplementedError

    def doc_to_label(self, doc) -> int:
        raise NotImplementedError


label_to_task_map = {
    "boolq_mc_5shot": (OEEvalTask, {"dataset_path": "boolq", "dataset_name": "mc_5shot", "metric_type": "acc"}),
}

if __name__ == "__main__":
    config = load_cfg()
    config.dataset_name = "boolq_mc_5shot"
    print_config(config)
    model, tokenizer = load_model_and_tokenizer(config)
    task_class = label_to_task_map[config.dataset_name]
    if isinstance(task_class, tuple):
        task_class, task_kwargs = task_class
    ds_eval_dataset = task_class(tokenizer=tokenizer, model_ctx_len=config.max_length, **task_kwargs)  # type: ignore
    ds_eval_dataloader = DataLoader(
        ds_eval_dataset,  # type: ignore
        batch_size=config.batch_size,
        collate_fn=ds_eval_dataset.collate_fn,
        num_workers=0,
    )
    metric = ICLMetric(metric_type=ds_eval_dataset.metric_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluator = Evaluator(
        label=config.dataset_name,
        type=EvaluatorType.downstream,
        eval_loader=ds_eval_dataloader,
        eval_metric=metric.to(device),
    )
    eval_lm = EvaluateLM(model=model, evaluator=evaluator, device=device)
    eval_metrics = eval_lm.eval()
    print(eval_metrics)
