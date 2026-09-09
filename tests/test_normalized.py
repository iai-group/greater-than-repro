"""Tests for the normalized-budget helpers of experiments/rq3_normalized.py."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from rq3_normalized import FRACTIONS, MATCHED_FRACTION, budgets_for, total_edges


def test_total_edges_counts_every_edge(tmp_path):
    graph = {
        "cfg": {"n_layers": 1, "n_heads": 1},
        "nodes": {"input": {"in_graph": True}},
        "edges": {
            "input->a0.h0<q>": {"score": 1.0, "in_graph": True},
            "input->a0.h0<k>": {"score": 0.5, "in_graph": False},
            "input->m0": {"score": 0.1, "in_graph": False},
        },
    }
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph))
    assert total_edges(str(path)) == 3


def test_matched_fraction_reproduces_the_canonical_gpt2_budget():
    # The matched fraction is defined so that gpt2-small's 32,491-edge graph
    # yields exactly the canonical 200-edge circuit, which is what makes it
    # comparable to the numbers in the submitted paper.
    assert budgets_for(32491, [MATCHED_FRACTION])[str(MATCHED_FRACTION)] == 200


def test_budgets_for_scales_with_graph_size():
    # pythia-2.8b's 1,591,857 edges at the same fraction need ~12x more edges
    # than the 800 the submitted paper's sweep reached.
    budget = budgets_for(1591857, [MATCHED_FRACTION])[str(MATCHED_FRACTION)]
    assert 9000 < budget < 10500


def test_budgets_for_never_returns_zero():
    assert budgets_for(100, [0.0003])["0.0003"] == 1


def test_fraction_grid_includes_the_matched_point_and_is_sorted():
    assert MATCHED_FRACTION in FRACTIONS
    assert FRACTIONS == sorted(FRACTIONS)
