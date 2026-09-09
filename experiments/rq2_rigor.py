"""RQ2: variance across templates/ranges, method agreement, random-circuit null,
less-than failure mode. Writes results/rq2_rigor.json."""
import argparse
import json
import sys
from functools import partial
from pathlib import Path

# Run as `python experiments/rq2_rigor.py`: sys.path[0] is then experiments/,
# not the project root, so gtlib (at the root) isn't importable without this
# (same convention as tests/conftest.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtlib import runner, stats, ground_truth
from gtlib.datasets import make_year_dataset, TEMPLATES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    model = runner.load_model("gpt2-small")
    metric = runner.year_prob_diff(model.tokenizer)
    out = {"templates": {}, "methods": {}, "random_null": None, "less_than": None}

    # (a) per-template baseline + circuit, with bootstrap CIs over items
    for tid in TEMPLATES:
        df = make_year_dataset(tid, n=args.n, seed=args.seed)
        dl = runner.df_to_dataloader(df)
        base_vals = runner.evaluate_baseline_values(model, dl, metric)  # per-item tensor
        g, perf = runner.discover(model, dl, metric)
        overlap = ground_truth.node_overlap(
            runner.graph_nodes(g), ground_truth.CORE_NODES)
        out["templates"][tid] = {
            "baseline": stats.bootstrap_ci(base_vals.tolist(), seed=args.seed),
            "circuit_perf": perf,
            "overlap_core": overlap,
        }
        print(f"[template={tid}] baseline_mean={out['templates'][tid]['baseline']['mean']:.4f} "
              f"circuit_perf={perf:.4f} overlap={overlap}", flush=True)

    # (b) method agreement on the canonical template
    df = make_year_dataset("war", n=args.n, seed=args.seed)
    dl = runner.df_to_dataloader(df)
    for method in ["EAP-IG-inputs", "EAP", "clean-corrupted"]:
        g, perf = runner.discover(model, dl, metric, method=method)
        out["methods"][method] = {
            "perf": perf,
            "nodes": sorted(runner.graph_nodes(g)),
        }
        print(f"[method={method}] perf={perf:.4f} n_nodes={len(out['methods'][method]['nodes'])}", flush=True)

    # (c) random-circuit null vs the EAP-IG circuit (reuse the EAP-IG-inputs graph from (b))
    g, perf = runner.discover(model, dl, metric)
    graph_path = runner.graph_json(g, path="results/_rq2_null_graph.json")
    null = stats.random_circuit_scores(model, graph_path, dl,
                                        partial(metric, loss=False, mean=False),
                                        n_random=25, seed=args.seed)
    out["random_null"] = {"observed": perf, "null": null,
                           "effect": stats.effect_size(perf, null)}
    print(f"[random_null] observed={perf:.4f} effect={out['random_null']['effect']}", flush=True)

    # (d) less-than failure mode: same circuit evaluated under inverted metric
    lt_metric = runner.year_prob_diff(model.tokenizer, invert=True)
    df_lt = make_year_dataset("war", n=args.n, seed=args.seed, invert=True)
    dl_lt = runner.df_to_dataloader(df_lt)
    out["less_than"] = {
        "full_model": float(runner.evaluate_baseline_values(model, dl_lt, lt_metric).mean()),
        "circuit": runner.evaluate_graph_mean(model, g, dl_lt, lt_metric),
    }
    print(f"[less_than] full_model={out['less_than']['full_model']:.4f} "
          f"circuit={out['less_than']['circuit']:.4f}", flush=True)

    Path("results").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("results/rq2_rigor.json", "w"), indent=2)
    print("WROTE results/rq2_rigor.json", flush=True)


if __name__ == "__main__":
    main()
