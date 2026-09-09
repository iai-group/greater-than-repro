"""Importable core of reproduce_greater_than.py: model loading, dataset/dataloader
wrapping, the year-prob-diff metric, and thin wrappers over the eap library for
circuit discovery and evaluation.

Behavior is kept identical to reproduce_greater_than.py (the reviewed, working
RQ1 script) — this module just makes the same logic importable and adds the
extra hooks (invertible metric, DataFrame-backed dataloader, node/graph
accessors) needed by the RQ2 rigor battery and later cross-model runs.
"""
import tempfile
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute


# ---- model loading ----
# Checkpoints large enough that fp32 weights alone would exhaust an 80GB A100
# (pythia-12b is ~24GB in half precision, ~48GB in fp32, before EAP-IG's
# activation caches). bfloat16 rather than float16: attribution backprops
# through the whole graph and fp16 gradients underflow on the small
# probability differences this metric produces.
HALF_PRECISION_MODELS = {"pythia-6.9b", "pythia-12b", "Qwen/Qwen2.5-7B"}


def load_model(name: str, device: str = "cuda", dtype=None) -> HookedTransformer:
    """Loads a HookedTransformer with the hooks EAP-IG's edge graph requires.

    Args:
        name: TransformerLens model name, e.g. "pythia-12b".
        device: Torch device string.
        dtype: Weight dtype. Defaults to torch.bfloat16 for the checkpoints in
            HALF_PRECISION_MODELS and to TransformerLens's own default
            (float32) otherwise, so the numbers already reported for the
            smaller models are unchanged.

    Returns:
        The loaded model, with split-QKV inputs, attention results, and MLP
        input hooks enabled.
    """
    if dtype is None and name in HALF_PRECISION_MODELS:
        dtype = torch.bfloat16
    kwargs = {"dtype": dtype} if dtype is not None else {}
    model = HookedTransformer.from_pretrained(name, device=device, **kwargs)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    return model


# ---- dataset / dataloader (DataFrame-backed, same collate as reproduce_greater_than.py) ----
def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    return list(clean), list(corrupted), list(labels)


class DFDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return row["clean"], row["corrupted"], row["label"]


def df_to_dataloader(df, batch_size: int = 100) -> DataLoader:
    return DataLoader(DFDataset(df), batch_size=batch_size, collate_fn=collate_EAP)


# ---- metric ----
def get_logit_positions(logits, input_length):
    idx = torch.arange(logits.size(0), device=logits.device)
    return logits[idx, input_length - 1]


def year_prob_diff(tokenizer, invert: bool = False):
    """Probability-difference metric from EAP-IG/greater_than.ipynb.

    invert=False (default): sum p(year > YY) - sum p(year <= YY)  [greater-than]
    invert=True: sum p(year <= YY) - sum p(year > YY)  [less-than; swaps the two sums]
    """
    # add_special_tokens=False: with a BOS-prepending tokenizer (TL Pythia/Qwen),
    # input_ids[0] would be BOS for every year, silently corrupting the metric.
    year_indices = torch.tensor(
        [tokenizer(f"{year:02d}", add_special_tokens=False).input_ids[0] for year in range(100)]
    )

    def prob_diff(logits, clean_logits, input_length, labels, mean=True, loss=False):
        logits = get_logit_positions(logits, input_length)
        probs = torch.softmax(logits, dim=-1)[:, year_indices.to(logits.device)]
        if invert:
            results = torch.stack([p[: y + 1].sum() - p[y + 1 :].sum() for p, y in zip(probs, labels)])
        else:
            results = torch.stack([p[y + 1 :].sum() - p[: y + 1].sum() for p, y in zip(probs, labels)])
        if loss:
            results = -results
        return results.mean() if mean else results

    return prob_diff


# ---- discovery ----
def discover(model, dataloader, metric, method: str = "EAP-IG-inputs", topn: int = 200, ig_steps: int = 5):
    """Build a Graph from `model`, attribute edges via `method`, apply top-n pruning,
    and evaluate circuit performance. Returns (graph, perf_float)."""
    torch.manual_seed(0)
    g = Graph.from_model(model)
    kw = {"ig_steps": ig_steps} if "IG" in method else {}
    attribute(model, g, dataloader, partial(metric, loss=True, mean=True), method=method, **kw)
    g.apply_topn(topn, True)
    perf = evaluate_graph(model, g, dataloader, partial(metric, loss=False, mean=False)).mean().item()
    return g, perf


# ---- evaluation helpers ----
def evaluate_baseline_values(model, dataloader, metric):
    """Per-item full-model metric values (no reduction)."""
    return evaluate_baseline(model, dataloader, partial(metric, loss=False, mean=False))


def evaluate_graph_mean(model, graph, dataloader, metric) -> float:
    """Mean circuit performance under `metric`."""
    return evaluate_graph(model, graph, dataloader, partial(metric, loss=False, mean=False)).mean().item()


def graph_nodes(graph: Graph) -> set:
    """in_graph node names, excluding the special 'input'/'logits' nodes — same
    normalization as gtlib.ground_truth.load_discovered_nodes."""
    return {name for name, node in graph.nodes.items()
            if name not in ("input", "logits") and bool(node.in_graph)}


def graph_json(graph: Graph, path: str = None) -> str:
    """Write graph.to_json to `path` (or a fresh temp file) and return the path."""
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".json", prefix="graph_")
        import os
        os.close(fd)
    graph.to_json(str(path))
    return str(path)
