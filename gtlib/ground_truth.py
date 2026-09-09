"""Hanna et al. 2023 greater-than circuit reference (node level) + overlap metrics."""
import json
import re

# Verified against arXiv:2305.00586 (Sec. 3-4): late MLPs compute the inequality,
# these heads move the YY token to the final position.
ATTN_HEADS = {"a5.h1", "a5.h5", "a6.h9", "a7.h10", "a8.h8", "a8.h11", "a9.h1"}
LATE_MLPS = {"m8", "m9", "m10", "m11"}
EARLY_MLPS = {"m0", "m1", "m2", "m3"}          # digit-identity carriers (paper Sec. 4)
CORE_NODES = ATTN_HEADS | LATE_MLPS
FULL_NODES = CORE_NODES | EARLY_MLPS

# Actual schema of results/graph_gpt2-small_EAP-IG-inputs.json (EAP-IG's Graph.to_json,
# see EAP-IG/src/eap/graph.py:696-716): top-level keys are {"cfg", "nodes", "edges"}.
# "nodes" is a dict mapping node name -> {"in_graph": bool[, "score": float, ...]}.
# Node names are already in the a{L}.h{H} / m{L} convention used here (attention-head
# node names carry no <q>/<k>/<v> suffix - those suffixes only appear on edge names,
# e.g. "a0.h0->a1.h1<q>", not on node names), plus two special non-circuit nodes
# "input" and "logits". A node counts as "discovered" iff nodes[name]["in_graph"] is True.
def load_discovered_nodes(graph_json_path: str) -> set:
    """Node names from an eap Graph.to_json file, normalized to a{L}.h{H} / m{L}."""
    with open(graph_json_path) as f:
        g = json.load(f)
    discovered = set()
    for name, info in g["nodes"].items():
        if name in ("input", "logits"):
            continue
        if not info.get("in_graph"):
            continue
        # Defensive normalization: strip any <q>/<k>/<v> suffix should it ever appear.
        norm = re.sub(r"<[qkv]>$", "", name)
        discovered.add(norm)
    return discovered

def node_overlap(discovered: set, reference: set) -> dict:
    inter = len(discovered & reference)
    union = len(discovered | reference)
    return {
        "iou": inter / union if union else 0.0,
        "recall": inter / len(reference) if reference else 0.0,
        "precision": inter / len(discovered) if discovered else 0.0,
    }
