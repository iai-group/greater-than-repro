from pathlib import Path

from gtlib.ground_truth import CORE_NODES, node_overlap, load_discovered_nodes

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_core_nodes_contents():
    assert {"m8", "m9", "m10", "m11"} <= CORE_NODES
    assert {"a9.h1", "a8.h11", "a7.h10"} <= CORE_NODES

def test_overlap_perfect():
    r = node_overlap(CORE_NODES, CORE_NODES)
    assert r["iou"] == r["recall"] == r["precision"] == 1.0

def test_overlap_partial():
    r = node_overlap({"m9", "m10", "a0.h0"}, {"m9", "m10", "m11"})
    assert abs(r["iou"] - 2/4) < 1e-9 and abs(r["recall"] - 2/3) < 1e-9

def test_load_discovered_real_file():
    nodes = load_discovered_nodes(
        str(PROJECT_ROOT / "results" / "graph_gpt2-small_EAP-IG-inputs.json")
    )
    assert len(nodes) >= 10
