"""E5: edge-level ablation for the RQ4 necessity probe.

Reviewer yDuj asked why the submitted necessity probe ablates late-MLP nodes
when the discovered circuit contains the *edges into* those MLPs, not the MLPs
themselves. This script ablates those edges instead.

Every condition (dataset, seed, two-shot prefix, batch size, metric, accuracy
definition, late-MLP threshold) is imported from rq4_nli rather than restated,
so these numbers are directly comparable to the node-ablation numbers already
in the paper. Only what is ablated differs.

Three conditions are reported:
  circuit_edges  the discovered circuit's own edges into late MLPs, which is
                 the condition the reviewer asked for.
  all_edges      every edge into those same MLPs, the direct edge-level
                 analogue of the published node ablation.
  random_control size-matched random edges drawn from the same population
                 (edges whose child is any MLP).

Writes results/rq4_edge_ablation.json.
"""
import argparse
import json
import random
import sys
from pathlib import Path

# Run as `python experiments/rq4_edge_ablation.py`: sys.path[0] is then
# experiments/, not the project root, so gtlib isn't importable without this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq4_nli import (NLI_N, NLI_SEED, NLI_X_RANGE, NLI_Y_RANGE,
                     PYTHIA_EVAL_BATCH_SIZE, PYTHIA_LATE_MLP_THRESHOLD,
                     PYTHIA_NAME, TWO_SHOT_PREFIX)

GRAPH_PATH = "results/graph_pythia-2.8b_EAP-IG-inputs.json"
CONTROL_SEED = 42
N_CONTROLS = 5


def late_mlp_names(graph_path: str, threshold: float) -> set:
    """Returns late-MLP node names from the RQ3 circuit record.

    Reads layer fractions from results/rq3_circuits.json rather than
    recomputing them, so the ablated set matches the published node ablation.

    Args:
        graph_path: Unused placeholder kept for call-site symmetry.
        threshold: Layer-fraction cutoff above which an MLP counts as late.

    Returns:
        Node names such as {"m20", "m21"}.
    """
    with open("results/rq3_circuits.json") as f:
        fractions = json.load(f)["models"][PYTHIA_NAME]["layer_fractions"]
    return {name for name, frac in fractions.items()
            if name.startswith("m") and frac >= threshold}


def edges_into(graph_json_path: str, targets: set, in_circuit_only: bool):
    """Lists edge names whose child is one of `targets`.

    Edge names are "<parent>-><child>", with attention children carrying a
    <q>/<k>/<v> suffix, so an edge into MLP m9 ends with "->m9"
    (EAP-IG/src/eap/graph.py, Graph.to_json).

    Args:
        graph_json_path: Path to a graph written by Graph.to_json.
        targets: Child node names to match.
        in_circuit_only: If True, keep only edges already in the circuit.

    Returns:
        A list of matching edge names.
    """
    with open(graph_json_path) as f:
        edges = json.load(f)["edges"]
    out = []
    for name, val in edges.items():
        if name.split("->")[-1] not in targets:
            continue
        if in_circuit_only and not val.get("in_graph"):
            continue
        out.append(name)
    return out


def all_mlp_edges(graph_json_path: str, n_layers: int) -> list:
    """Lists every edge whose child is any MLP, the control population."""
    mlps = {f"m{layer}" for layer in range(n_layers)}
    with open(graph_json_path) as f:
        edges = json.load(f)["edges"]
    return [name for name in edges if name.split("->")[-1] in mlps]


def _accuracy_with_edges_ablated(model, dataloader, metric, ablated):
    """Returns accuracy with `ablated` edges removed from an otherwise full graph.

    An edge that is not in_graph has its input replaced by the corrupted
    activation, so removing every edge into an MLP is the edge-level analogue
    of zero-ablating that MLP.

    Args:
        model: The loaded HookedTransformer.
        dataloader: Clean/corrupted probe batches.
        metric: nli_logit_diff, whose per-item sign encodes correctness.
        ablated: Edge names to remove.

    Returns:
        The fraction of items answered correctly.
    """
    from functools import partial

    from eap.evaluate import evaluate_graph
    from eap.graph import Graph

    g = Graph.from_json(GRAPH_PATH)
    for edge in g.edges.values():
        edge.in_graph = True
    for node_name, node in g.nodes.items():
        # LogitNode.in_graph is a read-only property that always returns True
        # (EAP-IG/src/eap/graph.py:104-109), so assigning to it raises.
        if node_name == "logits":
            continue
        node.in_graph = True
    for name in ablated:
        g.edges[name].in_graph = False

    values = evaluate_graph(
        model, g, dataloader, partial(metric, loss=False, mean=False)
    ).tolist()
    # Same accuracy definition as rq4_nli.py:54.
    return sum(1 for v in values if v > 0) / len(values)


def run(seed: int) -> dict:
    """Compares circuit-edge ablation against size-matched random controls.

    Args:
        seed: Dataset sampling seed; defaults to rq4_nli's NLI_SEED at the
            call site so the probe matches the published condition.

    Returns:
        The results dict written to results/rq4_edge_ablation.json.
    """
    import torch

    from gtlib import runner
    from gtlib.nli import make_nli_dataset, nli_logit_diff

    model = None
    try:
        model = runner.load_model(PYTHIA_NAME)
        n_layers = model.cfg.n_layers

        df = make_nli_dataset(
            NLI_N, seed=seed, tokenizer=model.tokenizer,
            require_single_token_numbers=False,
            x_range=NLI_X_RANGE, y_range=NLI_Y_RANGE,
        ).copy()
        # Two-shot prefix: the exact condition that scored 0.576 in
        # results/rq4_gate.json, matching the published node ablation.
        df["clean"] = TWO_SHOT_PREFIX + df["clean"]
        df["corrupted"] = TWO_SHOT_PREFIX + df["corrupted"]

        metric = nli_logit_diff(model.tokenizer)
        dl = runner.df_to_dataloader(df, batch_size=PYTHIA_EVAL_BATCH_SIZE)

        baseline = _accuracy_with_edges_ablated(model, dl, metric, [])
        print(f"[edge-ablation] baseline acc={baseline:.4f}", flush=True)

        targets = late_mlp_names(GRAPH_PATH, PYTHIA_LATE_MLP_THRESHOLD)
        circuit_edges = edges_into(GRAPH_PATH, targets, in_circuit_only=True)
        all_edges = edges_into(GRAPH_PATH, targets, in_circuit_only=False)
        print(f"[edge-ablation] late_mlps={sorted(targets)} "
              f"circuit_edges={len(circuit_edges)} "
              f"all_edges={len(all_edges)}", flush=True)

        circuit_acc = _accuracy_with_edges_ablated(
            model, dl, metric, circuit_edges
        )
        print(f"[edge-ablation] circuit_edges acc={circuit_acc:.4f}",
              flush=True)

        all_acc = _accuracy_with_edges_ablated(model, dl, metric, all_edges)
        print(f"[edge-ablation] all_edges acc={all_acc:.4f}", flush=True)

        pool = all_mlp_edges(GRAPH_PATH, n_layers)
        rng = random.Random(CONTROL_SEED)
        control_accs = []
        for i in range(N_CONTROLS):
            drawn = rng.sample(pool, min(len(circuit_edges), len(pool)))
            acc = _accuracy_with_edges_ablated(model, dl, metric, drawn)
            control_accs.append(acc)
            print(f"[edge-ablation] control[{i}] acc={acc:.4f}", flush=True)

        return {
            "model": PYTHIA_NAME,
            "n_items": len(df),
            "condition": "two-shot",
            "late_mlp_threshold": PYTHIA_LATE_MLP_THRESHOLD,
            "late_mlps": sorted(targets),
            "baseline_accuracy": baseline,
            "circuit_edges": {"n_edges": len(circuit_edges),
                              "accuracy": circuit_acc},
            "all_edges_into_late_mlps": {"n_edges": len(all_edges),
                                         "accuracy": all_acc},
            "random_edge_control": {
                "n_edges": len(circuit_edges),
                "n_samples": N_CONTROLS,
                "accuracies": control_accs,
                "mean": sum(control_accs) / len(control_accs),
            },
        }
    finally:
        if model is not None:
            del model
            torch.cuda.empty_cache()


def main():
    """Parses arguments and writes results/rq4_edge_ablation.json."""
    ap = argparse.ArgumentParser(
        description="E5: edge-level ablation for the RQ4 necessity probe."
    )
    ap.add_argument("--seed", type=int, default=NLI_SEED)
    args = ap.parse_args()

    out = run(seed=args.seed)
    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/rq4_edge_ablation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE results/rq4_edge_ablation.json", flush=True)


if __name__ == "__main__":
    main()
