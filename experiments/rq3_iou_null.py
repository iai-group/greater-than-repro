"""E4: a null distribution for the depth-bucketed IoU of RQ3 (CPU only).

Reviewer CxwV observed that with late-MLP-heavy circuits and coarse depth
buckets, a bucketed IoU of 0.71--1.00 against gpt2-small may be near-trivial,
and asked for the bucket granularity and a random late-layer baseline. This
script supplies both: for each model it samples size-matched random node sets
drawn only from late layers, the regime the real circuits occupy, and reports
where the observed IoU falls in the resulting null.

Buckets are round(layer_fraction, 1) keyed by component type, i.e. K=11
deciles held constant across every model depth, matching
rq3_crossmodel.run_discover exactly.

Writes results/rq3_iou_null.json.
"""
import argparse
import json
import random
import sys
from pathlib import Path

# Run as `python experiments/rq3_iou_null.py`: sys.path[0] is then
# experiments/, not the project root, so gtlib isn't importable without this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# round(fraction, 1) over [0, 1] yields 0.0, 0.1, ..., 1.0.
N_BUCKETS = 11

REFERENCE_MODEL = "gpt2-small"

# The layer-fraction threshold that rq3_crossmodel and the paper both use to
# call a component "late".
LATE_THRESHOLD = 0.5


def bucket_nodes(layer_fractions: dict) -> set:
    """Buckets node names into type-keyed depth deciles.

    Args:
        layer_fractions: Mapping from node name to layer fraction in [0, 1].

    Returns:
        A set of bucket labels such as "m@0.9" or "a@0.7".
    """
    bucketed = set()
    for node_name, frac in layer_fractions.items():
        kind = "m" if node_name.startswith("m") else "a"
        bucketed.add(f"{kind}@{round(frac, 1)}")
    return bucketed


def late_layer_candidates(n_layers: int, n_heads: int) -> list:
    """Lists every node at layer fraction >= LATE_THRESHOLD.

    Args:
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads per layer.

    Returns:
        Node names, each an MLP ("m9") or an attention head ("a9.h1").
    """
    out = []
    for layer in range(n_layers):
        frac = layer / (n_layers - 1) if n_layers > 1 else 0.0
        if frac < LATE_THRESHOLD:
            continue
        out.append(f"m{layer}")
        for head in range(n_heads):
            out.append(f"a{layer}.h{head}")
    return out


def _fractions_for(nodes, n_layers: int) -> dict:
    """Returns layer fractions for `nodes` in a model with `n_layers` layers."""
    fractions = {}
    for name in nodes:
        if name.startswith("m"):
            layer = int(name[1:])
        elif name.startswith("a"):
            layer = int(name[1:].split(".")[0])
        else:
            continue
        fractions[name] = layer / (n_layers - 1) if n_layers > 1 else 0.0
    return fractions


def run(n_samples: int, seed: int) -> dict:
    """Computes the late-layer IoU null for every model in rq3_circuits.json.

    Args:
        n_samples: Random node sets to draw per model.
        seed: Seed for the sampler.

    Returns:
        The results dict written to results/rq3_iou_null.json.
    """
    import numpy as np

    from gtlib.ground_truth import node_overlap

    with open("results/rq3_circuits.json") as f:
        circuits = json.load(f)["models"]

    ref_buckets = bucket_nodes(circuits[REFERENCE_MODEL]["layer_fractions"])

    per_model = {}
    for name, info in circuits.items():
        if name == REFERENCE_MODEL or "layer_fractions" not in info:
            continue
        n_layers = info["n_layers"]
        observed_iou = node_overlap(
            bucket_nodes(info["layer_fractions"]), ref_buckets
        )["iou"]

        # n_heads is not stored in rq3_circuits.json, so recover it from the
        # cached graph's cfg rather than assuming a family convention.
        with open(info["graph_path"]) as f:
            n_heads = json.load(f)["cfg"]["n_heads"]

        candidates = late_layer_candidates(n_layers, n_heads)
        size = min(len(info["nodes"]), len(candidates))
        rng = random.Random(seed)

        null = []
        for _ in range(n_samples):
            drawn = rng.sample(candidates, size)
            drawn_buckets = bucket_nodes(_fractions_for(drawn, n_layers))
            null.append(node_overlap(drawn_buckets, ref_buckets)["iou"])

        arr = np.asarray(null, dtype=float)
        per_model[name] = {
            "observed_iou": float(observed_iou),
            "circuit_size": len(info["nodes"]),
            "sampled_size": size,
            "n_late_candidates": len(candidates),
            "null_mean": float(arr.mean()),
            "null_sd": float(arr.std(ddof=1)),
            "null_p95": float(np.quantile(arr, 0.95)),
            "percentile": float((arr < observed_iou).mean() * 100),
        }
        print(f"[{name}] observed={observed_iou:.3f} "
              f"null={arr.mean():.3f}+-{arr.std(ddof=1):.3f} "
              f"p95={np.quantile(arr, 0.95):.3f} "
              f"pct={per_model[name]['percentile']:.1f}", flush=True)

    return {
        "n_buckets": N_BUCKETS,
        "n_samples": n_samples,
        "late_threshold": LATE_THRESHOLD,
        "reference_model": REFERENCE_MODEL,
        "models": per_model,
    }


def main():
    """Parses arguments and writes results/rq3_iou_null.json."""
    ap = argparse.ArgumentParser(
        description="E4: late-layer null for depth-bucketed IoU."
    )
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = run(n_samples=args.n_samples, seed=args.seed)
    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/rq3_iou_null.json", "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE results/rq3_iou_null.json", flush=True)


if __name__ == "__main__":
    main()
