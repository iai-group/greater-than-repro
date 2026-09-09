"""Reproduce the GPT-2 greater-than circuit in TransformerLens via EAP-IG.

Faithful script port of EAP-IG/greater_than.ipynb (Hanna, Liu, Variengien 2023,
arXiv:2305.00586). Baseline (full-model) prob-diff should be ~0.814; the EAP-IG
circuit should recover ~0.745 (~91%) with ~29 nodes / 150 edges.

Metric = probability difference = sum p(year > YY) - sum p(year <= YY), at the
final token, over the 00..99 two-digit year vocabulary.

Kept CLI-parameterizable (model / batch-size / methods) so the SAME script drives
the RQ3 cross-model runs (Pythia, Llama-3-8B, ...) later.
"""
import argparse
import json
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute


# ---- dataset + metric (verbatim logic from EAP-IG greater_than.ipynb) ----
def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    return list(clean), list(corrupted), list(labels)


class EAPDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return row["clean"], row["corrupted"], row["label"]

    def to_dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)


def get_logit_positions(logits, input_length):
    idx = torch.arange(logits.size(0), device=logits.device)
    return logits[idx, input_length - 1]


def get_prob_diff(tokenizer):
    year_indices = torch.tensor([tokenizer(f"{year:02d}").input_ids[0] for year in range(100)])

    def prob_diff(logits, clean_logits, input_length, labels, mean=True, loss=False):
        logits = get_logit_positions(logits, input_length)
        # .to(logits.device): defensive — advanced-indexing needs same-device index
        probs = torch.softmax(logits, dim=-1)[:, year_indices.to(logits.device)]
        results = torch.stack([p[y + 1 :].sum() - p[: y + 1].sum() for p, y in zip(probs, labels)])
        if loss:
            results = -results
        return results.mean() if mean else results

    return prob_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-small")
    ap.add_argument("--data", default=str(Path(__file__).parent / "greater_than_data.csv"))
    ap.add_argument("--batch-size", type=int, default=120)
    ap.add_argument("--topn", type=int, default=200)
    ap.add_argument("--ig-steps", type=int, default=5)
    ap.add_argument("--methods", nargs="+", default=["EAP-IG-inputs", "EAP", "clean-corrupted"])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model}", flush=True)
    if device == "cpu":
        print("WARNING: no CUDA — this will be slow.", flush=True)

    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    ds = EAPDataset(args.data)
    dataloader = ds.to_dataloader(args.batch_size)
    prob_diff = get_prob_diff(model.tokenizer)

    baseline = evaluate_baseline(model, dataloader, partial(prob_diff, loss=False, mean=False)).mean().item()
    print(f"BASELINE (full model) prob-diff = {baseline:.4f}   [paper ~0.814]", flush=True)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {"model": args.model, "baseline": baseline, "circuits": {}}

    for method in args.methods:
        g = Graph.from_model(model)
        kw = {"ig_steps": args.ig_steps} if "IG" in method else {}
        attribute(model, g, dataloader, partial(prob_diff, loss=True, mean=True), method=method, **kw)
        g.apply_topn(args.topn, True)
        perf = evaluate_graph(model, g, dataloader, partial(prob_diff, loss=False, mean=False)).mean().item()
        nodes, edges = g.count_included_nodes(), g.count_included_edges()
        recovery = perf / baseline if baseline else float("nan")
        print(
            f"{method:>16}: circuit prob-diff={perf:.4f}  recovery={recovery:6.1%}  "
            f"nodes={nodes} edges={edges}",
            flush=True,
        )
        g.to_json(str(outdir / f"graph_{args.model}_{method}.json"))
        summary["circuits"][method] = {
            "perf": perf,
            "recovery": recovery,
            "nodes": int(nodes),
            "edges": int(edges),
        }

    out_json = outdir / f"summary_{args.model}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
