"""Tests for the bucketed-IoU null of experiments/rq3_iou_null.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from rq3_iou_null import N_BUCKETS, bucket_nodes, late_layer_candidates


def test_bucket_nodes_uses_deciles_keyed_by_component_type():
    buckets = bucket_nodes({"m11": 1.0, "a9.h1": 0.8181, "m10": 0.9090})
    assert buckets == {"m@1.0", "a@0.8", "m@0.9"}


def test_bucket_nodes_collapses_same_decile_same_type():
    # Two heads in one layer collapse to a single bucket, which is exactly why
    # the null matters: the bucketed space is far smaller than the node space.
    assert bucket_nodes({"a9.h1": 0.8181, "a9.h6": 0.8181}) == {"a@0.8"}


def test_n_buckets_is_eleven():
    # round(f, 1) over [0, 1] yields 0.0, 0.1, ..., 1.0.
    assert N_BUCKETS == 11


def test_late_layer_candidates_excludes_early_layers():
    cands = late_layer_candidates(n_layers=12, n_heads=2)
    assert "m11" in cands
    assert "a6.h0" in cands
    assert "m0" not in cands
    assert "a0.h0" not in cands


def test_late_layer_candidates_counts_heads_and_mlps():
    # Layers 6..11 of a 12-layer model sit at fraction >= 0.5, since
    # fraction = layer / (n_layers - 1) and 6/11 = 0.545.
    cands = late_layer_candidates(n_layers=12, n_heads=2)
    assert len(cands) == 6 * (2 + 1)


def test_late_layer_candidates_handles_single_layer():
    assert late_layer_candidates(n_layers=1, n_heads=1) == []
