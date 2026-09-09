"""RQ3: behavioral gate across model families. For each candidate model, build a
500-item "war" dataset with THAT model's own tokenizer, compute the full-model
prob-diff, and gate it (pass iff mean >= 0.3) for downstream circuit discovery.
Writes results/rq3_gate.json."""
import argparse
import json
import sys
from pathlib import Path

# Run as `python experiments/rq3_crossmodel.py`: sys.path[0] is then experiments/,
# not the project root, so gtlib (at the root) isn't importable without this
# (same convention as tests/conftest.py and rq2_rigor.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODELS = [
    "gpt2-small",
    "gpt2-medium",
    "pythia-160m",
    "pythia-410m",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
    "pythia-12b",
    "Qwen/Qwen2.5-7B",
]

GATE_THRESHOLD = 0.3

# Gate-passing models (from results/rq3_gate.json) — Qwen/llama skipped there.
DISCOVER_MODELS = [
    "gpt2-small",
    "gpt2-medium",
    "pythia-160m",
    "pythia-410m",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
    "pythia-12b",
]

BUDGETS = [100, 200, 400, 800]

# Per-model evaluation batch sizes that fit an A100 80GB. Attribution holds
# activations for every node in the graph, so the larger checkpoints need far
# smaller batches than the 100 the smaller models tolerate.
BATCH_SIZES = {"pythia-1.4b": 25, "pythia-2.8b": 25, "pythia-6.9b": 8,
               "pythia-12b": 4}


def _sanitize(name: str) -> str:
    return name.replace("/", "_")


def run_discover(seed: int, n: int = 1000, only=None) -> dict:
    """For each gate-passing model: attribute once (EAP-IG-inputs), then sweep
    top-n budgets by re-applying apply_topn on the SAME scored graph (no re-attribution).

    Attribution-reuse choice: EAP-IG/src/eap/graph.py apply_topn(reset=True) only
    zeroes self.in_graph / self.nodes_in_graph (see Graph.reset, line ~357-373); it
    never touches self.scores. prune() (called after apply_topn) also only mutates
    in_graph/nodes_in_graph, never self.scores. So a single `attribute(...)` call's
    edge scores are valid for every subsequent `g.apply_topn(topn)` call — verified
    by reading the source, not assumed. We therefore attribute ONCE per model and
    apply_topn per budget: this is both correct and ~4x cheaper than re-attributing.
    """
    import torch
    from functools import partial
    from eap.graph import Graph
    from eap.attribute import attribute
    from eap.evaluate import evaluate_graph
    from gtlib import runner
    from gtlib.datasets import make_year_dataset
    from gtlib.ground_truth import node_overlap, CORE_NODES

    per_model = {}

    targets = only if only else DISCOVER_MODELS
    for name in targets:
        print(f"[{name}] loading...", flush=True)
        model = None
        try:
            model = runner.load_model(name)
            batch_size = BATCH_SIZES.get(name, 100)
            df = make_year_dataset("war", n=n, seed=seed, tokenizer=model.tokenizer)
            metric = runner.year_prob_diff(model.tokenizer)
            dl = runner.df_to_dataloader(df, batch_size=batch_size)

            baseline_vals = runner.evaluate_baseline_values(model, dl, metric)
            baseline = float(baseline_vals.mean())
            print(f"[{name}] baseline={baseline:.4f}", flush=True)

            torch.manual_seed(0)
            g = Graph.from_model(model)
            attribute(model, g, dl, partial(metric, loss=True, mean=True),
                      method="EAP-IG-inputs", ig_steps=5)

            n_layers = model.cfg.n_layers
            sweep = {}
            node_snapshot = None
            snapshot_budget = 200 if 200 in BUDGETS else max(BUDGETS)

            for topn in BUDGETS:
                g.apply_topn(topn, True)
                perf = evaluate_graph(model, g, dl, partial(metric, loss=False, mean=False)).mean().item()
                faithfulness = perf / baseline if baseline else float("nan")
                sweep[str(topn)] = faithfulness
                print(f"[{name}] n={topn} perf={perf:.4f} faithfulness={faithfulness:.4f}", flush=True)

                if topn == snapshot_budget:
                    graph_path = f"results/graph_{_sanitize(name)}_EAP-IG-inputs.json"
                    Path("results").mkdir(parents=True, exist_ok=True)
                    runner.graph_json(g, graph_path)
                    nodes = sorted(runner.graph_nodes(g))

                    layer_fractions = {}
                    mlp_count = 0
                    attn_count = 0
                    for node_name in nodes:
                        if node_name.startswith("m"):
                            layer = int(node_name[1:])
                            mlp_count += 1
                        elif node_name.startswith("a"):
                            layer = int(node_name[1:].split(".")[0])
                            attn_count += 1
                        else:
                            continue
                        layer_fractions[node_name] = layer / (n_layers - 1) if n_layers > 1 else 0.0

                    node_snapshot = {
                        "nodes": nodes,
                        "layer_fractions": layer_fractions,
                        # Edge-score share by component type is ambiguous when an edge's
                        # parent/child straddle both types (see task brief); we report the
                        # unambiguous proxy instead, explicitly named as such.
                        "mlp_attn_share": {"node_count_share": {"mlp": mlp_count, "attn": attn_count}},
                        "graph_path": graph_path,
                    }

            per_model[name] = {
                "baseline": baseline,
                "n_layers": n_layers,
                "sweep": sweep,
                **node_snapshot,
            }
        except Exception as exc:
            # A single OOM at 12b must not discard the smaller models' work.
            per_model[name] = {"error": str(exc)}
            print(f"[{name}] ERROR: {exc}", flush=True)
        finally:
            if model is not None:
                del model
                torch.cuda.empty_cache()

    # --- cross-model comparison ---
    def bucket_nodes(layer_fractions: dict) -> set:
        bucketed = set()
        for node_name, frac in layer_fractions.items():
            kind = "m" if node_name.startswith("m") else "a"
            decile = round(frac, 1)
            bucketed.add(f"{kind}@{decile}")
        return bucketed

    # Merge with any previously computed models so a subset run extends the
    # record instead of replacing it. A partial run must never drop entries
    # that earlier runs established and the paper already cites.
    merged = {}
    prior_path = Path("results/rq3_circuits.json")
    if prior_path.exists():
        with open(prior_path) as f:
            merged.update(json.load(f).get("models", {}))
    for name, info in per_model.items():
        # Only let a new result replace an old one if it actually succeeded.
        if "layer_fractions" in info or name not in merged:
            merged[name] = info
    per_model = merged

    ok = {name: info for name, info in per_model.items()
          if "layer_fractions" in info}
    bucketed = {name: bucket_nodes(info["layer_fractions"])
                for name, info in ok.items()}
    names_list = list(ok.keys())
    pairwise = {}
    for i in range(len(names_list)):
        for j in range(i + 1, len(names_list)):
            a, b = names_list[i], names_list[j]
            pairwise[f"{a}__{b}"] = node_overlap(bucketed[a], bucketed[b])

    core_overlap_gpt2 = {}
    for name in ("gpt2-small", "gpt2-medium"):
        if name in ok:
            core_overlap_gpt2[name] = node_overlap(set(ok[name]["nodes"]), CORE_NODES)

    return {
        "models": per_model,
        "pairwise_bucketed_overlap": pairwise,
        "core_overlap_gpt2": core_overlap_gpt2,
    }


def run_gate(seed: int, n: int = 500) -> dict:
    # Imported here (not at module scope) so `--help` works without loading torch/TL.
    import torch
    from gtlib import runner, stats
    from gtlib.datasets import make_year_dataset

    out = {}
    for name in MODELS:
        print(f"[{name}] loading...", flush=True)
        model = None
        try:
            model = runner.load_model(name)
            try:
                df = make_year_dataset("war", n=n, seed=seed, tokenizer=model.tokenizer)
            except ValueError:
                out[name] = {"skipped": "no single-token year encoding"}
                print(f"[{name}] skipped: no single-token year encoding", flush=True)
                continue

            metric = runner.year_prob_diff(model.tokenizer)
            # BATCH_SIZES matters here too: the gate ran at the default 100 and
            # OOMed on pythia-12b, which is why its gate entry was an error
            # while its discovery baseline computed fine at batch 4.
            dl = runner.df_to_dataloader(df, batch_size=BATCH_SIZES.get(name, 100))
            per_item = runner.evaluate_baseline_values(model, dl, metric)
            mean = float(per_item.mean())
            ci = stats.bootstrap_ci(per_item.tolist(), seed=seed)
            passed = mean >= GATE_THRESHOLD
            out[name] = {"prob_diff": mean, "ci": ci, "pass": passed}
            print(f"[{name}] prob_diff={mean:.4f} pass={passed}", flush=True)
        except Exception as exc:
            # An OOM on one large checkpoint must not discard the whole gate.
            out[name] = {"error": str(exc)}
            print(f"[{name}] ERROR: {exc}", flush=True)
        finally:
            if model is not None:
                del model
                torch.cuda.empty_cache()

    out["llama-3-8b"] = {"skipped": "hf-gated, no access"}

    # Merge with any prior gate record for the same reason as run_discover: a
    # subset or partially-failed run must not drop established entries.
    prior_path = Path("results/rq3_gate.json")
    if prior_path.exists():
        with open(prior_path) as f:
            prior = json.load(f)
        for name, info in prior.items():
            if name not in out or "error" in out.get(name, {}):
                out[name] = info
    return out


def main():
    ap = argparse.ArgumentParser(description="RQ3: cross-model behavioral gate.")
    ap.add_argument("--gate", action="store_true", help="Run the behavioral gate across models.")
    ap.add_argument("--discover", action="store_true",
                     help="Run cross-model circuit discovery (budget sweep + localization) on gate-passing models.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=None,
                     help="Dataset size (default: 500 for --gate, 1000 for --discover).")
    ap.add_argument("--only", nargs="+", default=None,
                     help="Restrict discovery to these models; results merge "
                          "into any existing results/rq3_circuits.json.")
    args = ap.parse_args()

    if not args.gate and not args.discover:
        print("Nothing to do: pass --gate or --discover.")
        return

    if args.gate:
        out = run_gate(seed=args.seed, n=args.n if args.n is not None else 500)
        Path("results").mkdir(parents=True, exist_ok=True)
        with open("results/rq3_gate.json", "w") as f:
            json.dump(out, f, indent=2)
        print("WROTE results/rq3_gate.json", flush=True)

    if args.discover:
        out = run_discover(seed=args.seed,
                           n=args.n if args.n is not None else 1000,
                           only=args.only)
        Path("results").mkdir(parents=True, exist_ok=True)
        with open("results/rq3_circuits.json", "w") as f:
            json.dump(out, f, indent=2)
        print("WROTE results/rq3_circuits.json", flush=True)


if __name__ == "__main__":
    main()
