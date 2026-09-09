"""Statistics for circuit faithfulness: bootstrap CIs, random-circuit null, effect sizes."""
import json
import numpy as np


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return {"mean": float(x.mean()),
            "lo": float(np.quantile(means, alpha / 2)),
            "hi": float(np.quantile(means, 1 - alpha / 2))}


def effect_size(observed, null_values):
    null = np.asarray(null_values, dtype=float)
    sd = null.std(ddof=1) if len(null) > 1 else 0.0
    d = (observed - null.mean()) / sd if sd > 0 else float("inf")
    pct = float((null < observed).mean() * 100)
    return {"cohens_d": float(d), "percentile": pct}


def random_circuit_scores(model, graph_json_path, dataloader, metric, n_random=25, seed=0):
    """Faithfulness of size-matched random edge sets (the null for the discovered circuit).

    Schema verified against results/graph_gpt2-small_EAP-IG-inputs.json:
    top-level {"cfg", "nodes", "edges"}; `edges` maps edge-name -> {"score": float,
    "in_graph": bool} (e.g. "input->a0.h0<q>": {"score": ..., "in_graph": False}).
    So the discovered-circuit edge count is the count of edges with in_graph == True.

    Graph/Edge API verified against EAP-IG/src/eap/graph.py:
    - Edge.score is a property backed by graph.scores[matrix_index] (get: lines 171-173,
      set: lines 175-176), so `edge.score = float(...)` for edge in g.edges.values()
      works as in the brief.
    - Graph.apply_topn(n, absolute=True, level='edge', reset=True, prune=True) (defined
      at line 443) with default level='edge' selects the top-n edges by |score| and sets
      them in_graph=True, matching the brief's `g.apply_topn(n_edges, True)` call.
    - Graph.from_model(model_or_config) (defined at line 603) builds an empty-scored graph
      from a HookedTransformer's cfg.
    """
    from eap.graph import Graph
    from eap.evaluate import evaluate_graph

    rng = np.random.default_rng(seed)
    with open(graph_json_path) as f:
        edges = json.load(f)["edges"]
    n_edges = sum(1 for v in edges.values() if (v.get("in_graph") if isinstance(v, dict) else v))

    scores = []
    for _ in range(n_random):
        g = Graph.from_model(model)
        for edge in g.edges.values():
            edge.score = float(rng.standard_normal())
        g.apply_topn(n_edges, True)
        scores.append(float(evaluate_graph(model, g, dataloader, metric).mean().item()))
    return scores
