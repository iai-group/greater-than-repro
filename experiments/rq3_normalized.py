"""E1: faithfulness at edge budgets normalized by total graph size (RQ3).

The submitted paper compared a fixed 200-edge budget across a 40x parameter
range. Measured from the cached graphs, 200 edges is 0.616% of gpt2-small's
32,491 edges but only 0.013% of pythia-2.8b's 1,591,857, so the comparison
was roughly 12x unfair at the top of the range (reviewers yDuj and CxwV both
raised this). This script re-evaluates every model at matched *fractions* of
its own graph.

No re-attribution is needed. Graph.to_json persists each edge's score and
Graph.from_json restores it (EAP-IG/src/eap/graph.py:696,736), and
apply_topn mutates only in_graph/nodes_in_graph, never scores, so a cached
graph can be re-thresholded at any budget.

Writes results/rq3_normalized.json.
"""
import argparse
import json
import sys
from pathlib import Path

# Run as `python experiments/rq3_normalized.py`: sys.path[0] is then
# experiments/, not the project root, so gtlib isn't importable without this
# (same convention as rq2_rigor.py and rq3_crossmodel.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# gpt2-small's canonical 200-edge circuit over its 32,491 total edges. This is
# the fraction-matched comparison point against the submitted paper.
MATCHED_FRACTION = 200 / 32491

FRACTIONS = sorted([0.0003, 0.0006, 0.0015, 0.003, MATCHED_FRACTION, 0.012])

MODELS = [
    "gpt2-small",
    "gpt2-medium",
    "pythia-160m",
    "pythia-410m",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
    "pythia-12b",
]

# Evaluation batch sizes that fit an A100 80GB, mirroring
# rq3_crossmodel.BATCH_SIZES.
BATCH_SIZES = {"pythia-2.8b": 25, "pythia-6.9b": 8, "pythia-12b": 4}


def _sanitize(name: str) -> str:
    """Returns `name` with slashes replaced, matching the cached-graph naming."""
    return name.replace("/", "_")


def graph_path_for(name: str) -> str:
    """Returns the cached EAP-IG graph path for model `name`."""
    return f"results/graph_{_sanitize(name)}_EAP-IG-inputs.json"


def total_edges(graph_json_path: str) -> int:
    """Counts every edge in a cached graph, in or out of the circuit.

    Args:
        graph_json_path: Path to a graph written by Graph.to_json.

    Returns:
        The total number of edges in the model's computational graph.
    """
    with open(graph_json_path) as f:
        return len(json.load(f)["edges"])


def budgets_for(n_edges: int, fractions=None) -> dict:
    """Converts edge fractions into absolute top-n budgets.

    Args:
        n_edges: The model's total edge count.
        fractions: Fractions of the graph to keep; defaults to FRACTIONS.

    Returns:
        A mapping from each fraction, formatted as a string, to the absolute
        edge budget it corresponds to, floored at 1.
    """
    if fractions is None:
        fractions = FRACTIONS
    return {str(f): max(1, round(f * n_edges)) for f in fractions}


def run(seed: int, n: int) -> dict:
    """Sweeps fraction-matched edge budgets for every model with a cached graph.

    Args:
        seed: Dataset sampling seed, matching the RQ3 discovery run.
        n: Number of year-span items to evaluate on.

    Returns:
        The results dict written to results/rq3_normalized.json.
    """
    import torch
    from eap.graph import Graph

    from gtlib import runner
    from gtlib.datasets import make_year_dataset

    per_model = {}
    for name in MODELS:
        gpath = graph_path_for(name)
        if not Path(gpath).exists():
            print(f"[{name}] SKIP: no cached graph at {gpath}", flush=True)
            continue

        with open(gpath) as f:
            cached = json.load(f)
        n_edges = len(cached["edges"])
        n_nodes = len(cached["nodes"])
        del cached
        budgets = budgets_for(n_edges)
        print(f"[{name}] total_edges={n_edges} budgets={budgets}", flush=True)

        model = None
        try:
            model = runner.load_model(name)
            batch_size = BATCH_SIZES.get(name, 100)
            df = make_year_dataset("war", n=n, seed=seed,
                                   tokenizer=model.tokenizer)
            metric = runner.year_prob_diff(model.tokenizer)
            dl = runner.df_to_dataloader(df, batch_size=batch_size)

            baseline = float(
                runner.evaluate_baseline_values(model, dl, metric).mean()
            )
            print(f"[{name}] baseline={baseline:.4f}", flush=True)

            g = Graph.from_json(gpath)
            fraction_sweep = {}
            for frac, topn in budgets.items():
                g.apply_topn(topn, True)
                perf = runner.evaluate_graph_mean(model, g, dl, metric)
                faith = perf / baseline if baseline else float("nan")
                fraction_sweep[frac] = faith
                print(f"[{name}] frac={float(frac):.5f} n={topn} "
                      f"perf={perf:.4f} faithfulness={faith:.4f}", flush=True)

            per_model[name] = {
                "total_edges": n_edges,
                "n_nodes": n_nodes,
                "baseline": baseline,
                "absolute_budgets": budgets,
                "fraction_sweep": fraction_sweep,
            }
        except Exception as exc:
            per_model[name] = {"error": str(exc), "total_edges": n_edges}
            print(f"[{name}] ERROR: {exc}", flush=True)
        finally:
            if model is not None:
                del model
                torch.cuda.empty_cache()

        # Written after every model so a late OOM does not lose earlier work.
        Path("results").mkdir(parents=True, exist_ok=True)
        with open("results/rq3_normalized.json", "w") as f:
            json.dump({"fractions": FRACTIONS,
                       "matched_fraction": MATCHED_FRACTION,
                       "models": per_model}, f, indent=2)

    return {"fractions": FRACTIONS, "matched_fraction": MATCHED_FRACTION,
            "models": per_model}


def main():
    """Parses arguments and writes results/rq3_normalized.json."""
    ap = argparse.ArgumentParser(
        description="E1: fraction-matched edge-budget sweep."
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    run(seed=args.seed, n=args.n)
    print("WROTE results/rq3_normalized.json", flush=True)


if __name__ == "__main__":
    main()
