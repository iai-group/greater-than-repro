"""E1b: the Qwen NLI circuit's faithfulness at normalized edge budgets.

The submitted paper reports the Qwen2.5-7B entailment circuit as
anti-faithful at "every budget tested" (-1.02 at 200 edges through -0.77 at
1600). But Qwen's graph has 935,747 edges, so 1600 edges is 0.17% of it,
against the 0.616% that gpt2-small's canonical 200-edge circuit represents.
The budget criticism that reviewers yDuj and CxwV raised against our RQ3
table therefore applies to this claim too, and we apply it to ourselves here
rather than leave the asymmetry in place.

Conditions (dataset, seed, ranges, metric, GQA ungrouping, batch-size
fallback) are imported or copied from rq4_nli.discover_qwen_nli so the
numbers extend that sweep rather than forming a separate one. No
re-attribution: the cached graph carries its edge scores.

Writes results/rq4_qwen_normalized.json.
"""
import argparse
import json
import sys
from pathlib import Path

# Run as `python experiments/rq4_qwen_normalized.py`: sys.path[0] is then
# experiments/, not the project root, so gtlib isn't importable without this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq3_normalized import FRACTIONS, MATCHED_FRACTION, budgets_for, total_edges
from rq4_nli import (NLI_N, NLI_SEED, NLI_X_RANGE, NLI_Y_RANGE,
                     QWEN_BATCH_SIZES, QWEN_NAME)

GRAPH_PATH = "results/graph_Qwen_Qwen2.5-7B_nli_EAP-IG-inputs.json"


def run(seed: int, fractions=None) -> dict:
    """Sweeps fraction-matched budgets for the cached Qwen NLI circuit.

    Args:
        seed: Dataset sampling seed; defaults to rq4_nli's NLI_SEED so the
            probe matches the published sweep.
        fractions: Fractions to evaluate; defaults to the full grid. Passing a
            subset lets a crashed budget be re-run without redoing the rest,
            and merges into any existing result file.

    Returns:
        The results dict written to results/rq4_qwen_normalized.json.
    """
    import torch
    from eap.evaluate import evaluate_graph
    from eap.graph import Graph
    from functools import partial

    from gtlib import runner
    from gtlib.nli import make_nli_dataset, nli_logit_diff

    n_edges = total_edges(GRAPH_PATH)
    budgets = budgets_for(n_edges, fractions)

    # Merge into any prior partial run so a re-run of one budget keeps the rest.
    # absolute_budgets must merge too, or a subset re-run leaves the result
    # file unable to say which edge count each retained faithfulness used.
    prior = {}
    prior_budgets = {}
    prior_path = Path("results/rq4_qwen_normalized.json")
    if prior_path.exists():
        with open(prior_path) as f:
            cached = json.load(f)
        prior = cached.get("fraction_sweep", {})
        prior_budgets = cached.get("absolute_budgets", {})
    budgets_record = {**prior_budgets, **budgets}
    print(f"[qwen-nli-norm] total_edges={n_edges} budgets={budgets}",
          flush=True)

    model = None
    try:
        model = runner.load_model(QWEN_NAME)
        # eap asserts ungroup_grouped_query_attention for GQA models, and it
        # must be set before Graph.from_json so hook shapes match. Same
        # reasoning as rq4_nli.discover_qwen_nli.
        if model.cfg.n_key_value_heads is not None:
            model.cfg.ungroup_grouped_query_attention = True

        df = make_nli_dataset(
            NLI_N, seed=seed, tokenizer=model.tokenizer,
            require_single_token_numbers=False,
            x_range=NLI_X_RANGE, y_range=NLI_Y_RANGE,
        )
        metric = nli_logit_diff(model.tokenizer)

        def eval_with_fallback(graph=None):
            """Evaluates baseline or circuit, halving batch size on OOM."""
            for bs in QWEN_BATCH_SIZES:
                try:
                    dl = runner.df_to_dataloader(df, batch_size=bs)
                    if graph is None:
                        return float(runner.evaluate_baseline_values(
                            model, dl, metric).mean()), bs
                    return evaluate_graph(
                        model, graph, dl,
                        partial(metric, loss=False, mean=False)
                    ).mean().item(), bs
                except torch.OutOfMemoryError:
                    print(f"[qwen-nli-norm] OOM at batch_size={bs}, "
                          "retrying smaller", flush=True)
                    torch.cuda.empty_cache()
                    continue
            raise RuntimeError("OOM at every batch size")

        baseline, bs = eval_with_fallback()
        print(f"[qwen-nli-norm] baseline={baseline:.4f} (batch {bs})",
              flush=True)

        g = Graph.from_json(GRAPH_PATH)
        fraction_sweep = dict(prior)
        for frac, topn in budgets.items():
            g.apply_topn(topn, True)
            perf, bs = eval_with_fallback(g)
            faith = perf / baseline if baseline else float("nan")
            fraction_sweep[frac] = faith
            print(f"[qwen-nli-norm] frac={float(frac):.5f} n={topn} "
                  f"perf={perf:.4f} faithfulness={faith:.4f}", flush=True)

            Path("results").mkdir(parents=True, exist_ok=True)
            with open("results/rq4_qwen_normalized.json", "w") as f:
                json.dump({"model": QWEN_NAME,
                           "task": "synthetic comparison NLI",
                           "total_edges": n_edges,
                           "matched_fraction": MATCHED_FRACTION,
                           "fractions": FRACTIONS,
                           "baseline": baseline,
                           "absolute_budgets": budgets_record,
                           "fraction_sweep": fraction_sweep}, f, indent=2)

        return {"model": QWEN_NAME, "total_edges": n_edges,
                "fraction_sweep": fraction_sweep}
    finally:
        if model is not None:
            del model
            torch.cuda.empty_cache()


def main():
    """Parses arguments and writes results/rq4_qwen_normalized.json."""
    ap = argparse.ArgumentParser(
        description="E1b: normalized budgets for the Qwen NLI circuit."
    )
    ap.add_argument("--seed", type=int, default=NLI_SEED)
    ap.add_argument("--fractions", type=float, nargs="+", default=None,
                    help="Subset of the fraction grid to evaluate; merges "
                         "into any existing result file.")
    args = ap.parse_args()

    run(seed=args.seed, fractions=args.fractions)
    print("WROTE results/rq4_qwen_normalized.json", flush=True)


if __name__ == "__main__":
    main()
