"""RQ4: NLI behavioral gate across model families. For each candidate model,
build a 500-item synthetic comparison-NLI probe with THAT model's own
tokenizer, evaluate zero-shot accuracy (fraction of items where the model's
signed " Yes"/" No" logit-diff favors the correct answer). If zero-shot
accuracy < 0.6, also try a 2-shot prompt (two worked examples prepended).
Gate: pass iff the best-condition accuracy >= 0.6. Writes results/rq4_gate.json.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

# Run as `python experiments/rq4_nli.py`: sys.path[0] is then experiments/, not
# the project root, so gtlib (at the root) isn't importable without this (same
# convention as tests/conftest.py, rq2_rigor.py, rq3_crossmodel.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODELS = [
    "gpt2-small",
    "gpt2-medium",
    "pythia-160m",
    "pythia-410m",
    "pythia-1.4b",
    "pythia-2.8b",
    "Qwen/Qwen2.5-7B",
]

GATE_THRESHOLD = 0.6


class AttributionOOM(RuntimeError):
    """Raised when the attribution batch-size ladder exhausts (or a later
    evaluation sweep OOMs at every batch size) without a torch.OutOfMemoryError
    surviving to propagate. Caught by the model fallback ladder alongside
    torch.OutOfMemoryError so exhaustion falls through to the next smaller
    Qwen checkpoint instead of crashing the job."""

TWO_SHOT_PREFIX = (
    "Premise: Anna has 7 books. Hypothesis: Anna has more than 2 books. "
    "Question: Is the hypothesis true? Answer: Yes\n\n"
    "Premise: Tom has 3 coins. Hypothesis: Tom has more than 9 coins. "
    "Question: Is the hypothesis true? Answer: No\n\n"
)


def _run_condition(model, df, metric, batch_size=50):
    """Run the NLI probe on `df` and return (accuracy, mean_logit_diff, ci, per_item)."""
    from gtlib import runner, stats

    dl = runner.df_to_dataloader(df, batch_size=batch_size)
    per_item = runner.evaluate_baseline_values(model, dl, metric)
    values = per_item.tolist()
    accuracy = sum(1 for v in values if v > 0) / len(values)
    ci = stats.bootstrap_ci(values)
    mean_logit_diff = float(sum(values) / len(values))
    return accuracy, mean_logit_diff, ci


def _debug_print_first_item(model, df, name):
    """Print the first prompt + model's top-3 next-token candidates for debugging
    visibility into the run log."""
    import torch

    prompt = df.iloc[0]["clean"]
    print(f"[{name}] example prompt: {prompt!r}", flush=True)
    with torch.no_grad():
        logits = model(prompt, prepend_bos=True)
    last_logits = logits[0, -1]
    top3 = torch.topk(last_logits, 3)
    tokens = [model.tokenizer.decode([tid]) for tid in top3.indices.tolist()]
    print(f"[{name}] top-3 next tokens: {list(zip(tokens, top3.values.tolist()))}", flush=True)


def run_gate(seed: int, n: int = 500) -> dict:
    # Imported here (not at module scope) so `--help` works without loading torch/TL.
    import torch
    from gtlib import runner
    from gtlib.nli import make_nli_dataset, nli_logit_diff

    out = {}
    for name in MODELS:
        print(f"[{name}] loading...", flush=True)
        model = None
        try:
            model = runner.load_model(name)
            # Gate mode only needs plain forward passes, not EAP attribution, so
            # disable the hook flags load_model sets for attribution — they blow
            # up activation memory (e.g. complex_attn_linear OOM on Qwen2.5-7B).
            model.cfg.use_split_qkv_input = False
            model.cfg.use_attn_result = False
            model.cfg.use_hook_mlp_in = False

            # x_range/y_range=(10, 98): 2-digit-only numbers make the
            # same-digit-count corruption constraint trivially satisfiable
            # and keep the probe comparable across all model tokenizers
            # (Qwen tokenizes numbers per-digit, so a digit-count-changing
            # corruption like x=5 -> x'=12 would break positional
            # clean/corrupted pairing in EAP evaluation).
            df = make_nli_dataset(
                n, seed=seed, tokenizer=model.tokenizer, require_single_token_numbers=False,
                x_range=(10, 98), y_range=(10, 98),
            )
            metric = nli_logit_diff(model.tokenizer)

            _debug_print_first_item(model, df, name)

            # Large models need a smaller eval batch size to avoid OOM.
            eval_batch_size = 16 if name in ("pythia-2.8b", "Qwen/Qwen2.5-7B") else 50

            zs_acc, zs_mean, zs_ci = _run_condition(model, df, metric, batch_size=eval_batch_size)
            print(f"[{name}] zero_shot accuracy={zs_acc:.4f} mean_logit_diff={zs_mean:.4f}", flush=True)

            result = {
                "zero_shot": {"accuracy": zs_acc, "mean_logit_diff": zs_mean, "ci": zs_ci},
                "best": "zero_shot",
            }
            best_acc = zs_acc

            if zs_acc < GATE_THRESHOLD:
                df_2shot = df.copy()
                df_2shot["clean"] = TWO_SHOT_PREFIX + df_2shot["clean"]
                df_2shot["corrupted"] = TWO_SHOT_PREFIX + df_2shot["corrupted"]

                ts_acc, ts_mean, ts_ci = _run_condition(model, df_2shot, metric, batch_size=eval_batch_size)
                print(f"[{name}] two_shot accuracy={ts_acc:.4f} mean_logit_diff={ts_mean:.4f}", flush=True)

                result["two_shot"] = {"accuracy": ts_acc, "mean_logit_diff": ts_mean, "ci": ts_ci}
                if ts_acc > best_acc:
                    best_acc = ts_acc
                    result["best"] = "two_shot"

            result["pass"] = best_acc >= GATE_THRESHOLD
            out[name] = result
            print(f"[{name}] pass={result['pass']} (best={result['best']}, acc={best_acc:.4f})", flush=True)
        except Exception as e:
            print(f"[{name}] ERROR: {e}", flush=True)
            traceback.print_exc()
            out[name] = {"error": str(e)}
        finally:
            if model is not None:
                del model
                torch.cuda.empty_cache()

    return out


def _sanitize(name: str) -> str:
    return name.replace("/", "_")


# Same bucketing scheme as experiments/rq3_crossmodel.py:run_discover (not
# importable as a standalone function there — it's a local closure — so
# copied verbatim; kept in sync by inspection, not by import).
def _bucket_nodes(layer_fractions: dict) -> set:
    bucketed = set()
    for node_name, frac in layer_fractions.items():
        kind = "m" if node_name.startswith("m") else "a"
        decile = round(frac, 1)
        bucketed.add(f"{kind}@{decile}")
    return bucketed


QWEN_NAME = "Qwen/Qwen2.5-7B"
NLI_N = 500
NLI_SEED = 42
NLI_X_RANGE = (10, 98)
NLI_Y_RANGE = (10, 98)
QWEN_BUDGETS = [200, 400, 800, 1600]
QWEN_SNAPSHOT_BUDGET = 400
QWEN_IG_STEPS = 5
QWEN_BATCH_SIZES = [8, 4, 2, 1]

# Part A model fallback ladder: try the largest first; if it fails the inline
# NLI gate or OOMs during attribution at every batch size, fall back to the
# next smaller Qwen2.5 checkpoint. 3B/1.5B were being cached to HF_HOME by the
# controller alongside this task; verified present via tokenizer/model load on
# the login node before submission.
MODELS_A = ["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-1.5B"]
GATE_PROBE_N = 200


def _inline_gate_check(model_name: str, seed: int) -> dict:
    """Quick 200-item plain-forward NLI accuracy check, hook flags OFF (gate
    doesn't need attribution memory). Model is reloaded fresh afterward for
    attribution, since load_model always turns the hook flags back on."""
    import torch
    from gtlib import runner
    from gtlib.nli import make_nli_dataset, nli_logit_diff

    model = runner.load_model(model_name)
    try:
        model.cfg.use_split_qkv_input = False
        model.cfg.use_attn_result = False
        model.cfg.use_hook_mlp_in = False
        df = make_nli_dataset(
            GATE_PROBE_N, seed=seed, tokenizer=model.tokenizer, require_single_token_numbers=False,
            x_range=NLI_X_RANGE, y_range=NLI_Y_RANGE,
        )
        metric = nli_logit_diff(model.tokenizer)
        acc, mean_ld, ci = _run_condition(model, df, metric, batch_size=16)
        return {"accuracy": acc, "mean_logit_diff": mean_ld, "ci": ci, "n": GATE_PROBE_N}
    finally:
        del model
        torch.cuda.empty_cache()


def discover_qwen_nli_with_fallback(seed: int = NLI_SEED) -> dict:
    """Try each model in MODELS_A in order: gate-check, then attempt full
    attribution discovery. First model to pass the gate AND complete
    attribution (within the batch-size ladder) wins. Honestly records why
    each earlier candidate was skipped/OOM'd."""
    import torch

    attempts = []
    for i, model_name in enumerate(MODELS_A):
        gate = _inline_gate_check(model_name, seed)
        print(f"[rq4-A-gate] {model_name} accuracy={gate['accuracy']:.4f}", flush=True)
        if gate["accuracy"] < GATE_THRESHOLD:
            reason = f"failed inline NLI gate (acc={gate['accuracy']:.4f} < {GATE_THRESHOLD})"
            print(f"[rq4-A] {model_name} skipped: {reason}", flush=True)
            attempts.append({"model": model_name, "skipped": reason, "gate_check": gate})
            continue

        try:
            result = discover_qwen_nli(model_name, seed=seed)
        except (torch.OutOfMemoryError, AttributionOOM) as e:
            reason = f"attribution OOM at all batch sizes {QWEN_BATCH_SIZES}: {e}"
            print(f"[rq4-A] {model_name} skipped: {reason}", flush=True)
            attempts.append({"model": model_name, "oom": reason, "gate_check": gate})
            continue

        result["gate_check"] = gate
        result["model_used"] = model_name
        result["substitution_reason"] = (
            "primary model (Qwen2.5-7B) succeeded" if i == 0
            else f"fallback from {MODELS_A[0]} after earlier candidate(s) failed gate/OOM'd"
        )
        result["attempts_before_success"] = attempts
        return result

    return {
        "skipped": "all candidate models in MODELS_A failed the NLI gate or OOM'd during attribution",
        "attempts": attempts,
    }


def discover_qwen_nli(model_name: str = QWEN_NAME, seed: int = NLI_SEED) -> dict:
    """Attribute `model_name`'s NLI-entailment circuit (EAP-IG-inputs), sweep top-n
    budgets on the same scored graph, save the n=400 graph, and structurally
    compare (depth-bucketed node_overlap) against every RQ3 year-task circuit."""
    import torch
    from functools import partial
    from eap.graph import Graph
    from eap.attribute import attribute
    from eap.evaluate import evaluate_graph
    from gtlib import runner
    from gtlib.nli import make_nli_dataset, nli_logit_diff
    from gtlib.ground_truth import node_overlap

    model = runner.load_model(model_name)
    # eap's attribute/evaluate_graph assert ungroup_grouped_query_attention for
    # GQA models (Qwen2.5-7B has grouped KV heads). This is a runtime flag in
    # TransformerLens (K/V computed per query head at forward time via the
    # expanded W_K/W_V properties, weights untouched), so setting it post-load
    # is safe — but it must be set BEFORE Graph.from_model so hook shapes match.
    # It further inflates activation memory (k/v inputs repeated to n_heads);
    # the batch-size fallback below absorbs that.
    if model.cfg.n_key_value_heads is not None:
        model.cfg.ungroup_grouped_query_attention = True
    try:
        df = make_nli_dataset(
            NLI_N, seed=seed, tokenizer=model.tokenizer, require_single_token_numbers=False,
            x_range=NLI_X_RANGE, y_range=NLI_Y_RANGE,
        )
        metric = nli_logit_diff(model.tokenizer)
        n_layers = model.cfg.n_layers

        # Attribution is the memory-heaviest step (7B GQA + attn_result + IG
        # gradients): try batch_size=8, halving on CUDA OOM down to 2.
        g = None
        used_batch_size = None
        baseline = None
        last_err = None
        for bs in QWEN_BATCH_SIZES:
            try:
                dl = runner.df_to_dataloader(df, batch_size=bs)
                baseline_vals = runner.evaluate_baseline_values(model, dl, metric)
                baseline = float(baseline_vals.mean())
                print(f"[qwen-nli] batch_size={bs} baseline={baseline:.4f}", flush=True)

                torch.manual_seed(0)
                g = Graph.from_model(model)
                attribute(model, g, dl, partial(metric, loss=True, mean=True),
                          method="EAP-IG-inputs", ig_steps=QWEN_IG_STEPS)
                used_batch_size = bs
                break
            except torch.OutOfMemoryError as e:
                last_err = f"{type(e).__name__}: {e}"
                print(f"[qwen-nli] OOM at batch_size={bs}, retrying smaller", flush=True)
                g = None
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                continue
        if g is None:
            raise AttributionOOM(
                f"attribution OOM at all batch sizes {QWEN_BATCH_SIZES}: {last_err}"
            )

        sweep = {}
        node_snapshot = None
        for topn in QWEN_BUDGETS:
            g.apply_topn(topn, True)
            eval_bs = used_batch_size
            perf = None
            for bs in [eval_bs] + [b for b in QWEN_BATCH_SIZES if b < eval_bs]:
                try:
                    eval_dl = runner.df_to_dataloader(df, batch_size=bs)
                    perf = evaluate_graph(model, g, eval_dl, partial(metric, loss=False, mean=False)).mean().item()
                    break
                except torch.OutOfMemoryError:
                    print(f"[qwen-nli] OOM evaluating n={topn} at batch_size={bs}, retrying smaller", flush=True)
                    torch.cuda.empty_cache()
                    continue
            if perf is None:
                raise AttributionOOM(f"[qwen-nli] could not evaluate n={topn} even at batch_size=2")

            faithfulness = perf / baseline if baseline else float("nan")
            sweep[str(topn)] = faithfulness
            print(f"[qwen-nli] n={topn} perf={perf:.4f} faithfulness={faithfulness:.4f}", flush=True)

            if topn == QWEN_SNAPSHOT_BUDGET:
                graph_path = f"results/graph_{_sanitize(model_name)}_nli_EAP-IG-inputs.json"
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
                    "node_count_share": {"mlp": mlp_count, "attn": attn_count},
                    "graph_path": graph_path,
                }

        # --- structural cross-task comparison vs every RQ3 year-task circuit ---
        bucketed_overlap = {}
        try:
            with open("results/rq3_circuits.json") as f:
                rq3 = json.load(f)
            qwen_bucketed = _bucket_nodes(node_snapshot["layer_fractions"])
            for name, info in rq3["models"].items():
                other_bucketed = _bucket_nodes(info["layer_fractions"])
                bucketed_overlap[name] = node_overlap(qwen_bucketed, other_bucketed)
        except FileNotFoundError:
            print("[qwen-nli] WARNING: results/rq3_circuits.json not found; "
                  "skipping bucketed cross-task comparison", flush=True)

        return {
            "baseline": baseline,
            "batch_size_used": used_batch_size,
            "n_layers": n_layers,
            "sweep": sweep,
            **node_snapshot,
            "bucketed_overlap_vs_year_circuits": bucketed_overlap,
        }
    finally:
        del model
        import torch as _torch
        _torch.cuda.empty_cache()


PYTHIA_NAME = "pythia-2.8b"
PYTHIA_EVAL_BATCH_SIZE = 16
PYTHIA_LATE_MLP_THRESHOLD = 0.6
PYTHIA_CONTROL_POOL_THRESHOLD = 0.5
PYTHIA_CONTROL_SEED = 42


def _zero_hook(value, hook):
    import torch
    return torch.zeros_like(value)


def _eval_with_mlp_ablation(model, dataloader, metric, layers):
    """Plain TL forward loop (mirrors eap.evaluate.evaluate_baseline's clean-side
    pass) with the given MLP layers' outputs zero-ablated via run_with_hooks.
    Not threaded through eap's evaluate_graph (that expects an eap Graph, not an
    arbitrary hook list) — this is a small local evaluator instead, per design."""
    import torch
    from functools import partial
    from eap.utils import tokenize_plus

    metric_partial = partial(metric, loss=False, mean=False)
    fwd_hooks = [(f"blocks.{L}.hook_mlp_out", _zero_hook) for L in layers]
    results = []
    with torch.inference_mode():
        for clean, corrupted, label in dataloader:
            clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
            r = metric_partial(logits, None, input_lengths, label).cpu()
            results.append(r)
    return torch.cat(results).tolist()


def necessity_probe_pythia28b(seed: int = NLI_SEED) -> dict:
    """Does zero-ablating pythia-2.8b's year-comparison-circuit late MLPs hurt its
    (weak, 0.576 two-shot) NLI signal more than ablating a size-matched random
    set of other MLPs?"""
    import random
    from gtlib import runner, stats
    from gtlib.nli import make_nli_dataset, nli_logit_diff

    model = runner.load_model(PYTHIA_NAME)
    try:
        df = make_nli_dataset(
            NLI_N, seed=seed, tokenizer=model.tokenizer, require_single_token_numbers=False,
            x_range=NLI_X_RANGE, y_range=NLI_Y_RANGE,
        )
        # Two-shot prefix: exact condition that scored 0.576 in results/rq4_gate.json.
        df = df.copy()
        df["clean"] = TWO_SHOT_PREFIX + df["clean"]
        df["corrupted"] = TWO_SHOT_PREFIX + df["corrupted"]

        metric = nli_logit_diff(model.tokenizer)
        dl = runner.df_to_dataloader(df, batch_size=PYTHIA_EVAL_BATCH_SIZE)

        baseline_vals = runner.evaluate_baseline_values(model, dl, metric).tolist()
        baseline_ci = stats.bootstrap_ci(baseline_vals, seed=seed)
        baseline_acc = sum(1 for v in baseline_vals if v > 0) / len(baseline_vals)
        print(f"[pythia-necessity] baseline mean={baseline_ci['mean']:.4f} acc={baseline_acc:.4f}", flush=True)

        with open("results/rq3_circuits.json") as f:
            rq3 = json.load(f)
        layer_fractions = rq3["models"][PYTHIA_NAME]["layer_fractions"]
        late_mlps = sorted(
            int(n[1:]) for n, frac in layer_fractions.items()
            if n.startswith("m") and frac >= PYTHIA_LATE_MLP_THRESHOLD
        )
        control_pool = sorted(
            int(n[1:]) for n, frac in layer_fractions.items()
            if n.startswith("m") and frac < PYTHIA_CONTROL_POOL_THRESHOLD
        )
        rng = random.Random(PYTHIA_CONTROL_SEED)
        control_layers = sorted(rng.sample(control_pool, min(len(late_mlps), len(control_pool))))
        print(f"[pythia-necessity] late_mlps={late_mlps}", flush=True)
        print(f"[pythia-necessity] control_layers={control_layers}", flush=True)

        late_vals = _eval_with_mlp_ablation(model, dl, metric, late_mlps)
        late_ci = stats.bootstrap_ci(late_vals, seed=seed)
        late_acc = sum(1 for v in late_vals if v > 0) / len(late_vals)
        print(f"[pythia-necessity] late_mlp_ablated mean={late_ci['mean']:.4f} acc={late_acc:.4f}", flush=True)

        control_vals = _eval_with_mlp_ablation(model, dl, metric, control_layers)
        control_ci = stats.bootstrap_ci(control_vals, seed=seed)
        control_acc = sum(1 for v in control_vals if v > 0) / len(control_vals)
        print(f"[pythia-necessity] control_ablated mean={control_ci['mean']:.4f} acc={control_acc:.4f}", flush=True)

        return {
            "baseline": {"mean": baseline_ci["mean"], "ci": baseline_ci, "accuracy": baseline_acc},
            "late_mlp_ablated": {"mean": late_ci["mean"], "ci": late_ci, "accuracy": late_acc},
            "control_ablated": {"mean": control_ci["mean"], "ci": control_ci, "accuracy": control_acc},
            "ablated_layers": late_mlps,
            "control_layers": control_layers,
        }
    finally:
        import torch
        del model
        torch.cuda.empty_cache()


RQ4_CIRCUITS_PATH = "results/rq4_circuits.json"

_NOTE = (
    "The intersection of RQ3 year-task-circuit-discovered models and "
    "RQ4 NLI-gate passers is empty (only Qwen2.5-7B passes the NLI gate, "
    "and it never had a year-task circuit discovered; see results/rq4_gate.json "
    "and results/rq3_circuits.json). This file therefore reports (A) a "
    "gate-passing Qwen2.5 checkpoint's own NLI circuit (with size fallback if "
    "the primary 7B model OOMs or fails the gate) plus a structural "
    "(depth-bucketed) comparison against every RQ3 year-task circuit, and "
    "(B) a necessity-transfer probe on pythia-2.8b (best non-gate-passing NLI "
    "signal, 0.576 two-shot) asking whether ablating its year-circuit late "
    "MLPs hurts its NLI signal more than a size-matched control ablation."
)


def run_discover(seed: int = NLI_SEED, results_path: str = RQ4_CIRCUITS_PATH) -> dict:
    """Part B (pythia-2.8b necessity transfer) runs FIRST and is saved to disk
    immediately on completion — a 12h job must not lose B's results to an A
    crash/OOM. Part A (Qwen NLI circuit discovery, with size fallback) runs
    second and rewrites the file with qwen_nli added on success (or with an
    honest skip/OOM record if every candidate fails)."""
    Path("results").mkdir(parents=True, exist_ok=True)

    pythia28b_necessity = necessity_probe_pythia28b(seed=seed)
    out = {"pythia28b_necessity": pythia28b_necessity, "note": _NOTE}
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE partial {results_path} (Part B only)", flush=True)

    qwen_nli = discover_qwen_nli_with_fallback(seed=seed)
    out["qwen_nli"] = qwen_nli
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE {results_path} (Part A + B)", flush=True)

    return out


def main():
    ap = argparse.ArgumentParser(description="RQ4: NLI behavioral gate + circuit discovery.")
    ap.add_argument("--gate", action="store_true", help="Run the NLI behavioral gate across models.")
    ap.add_argument("--discover", action="store_true",
                     help="Run Qwen NLI circuit discovery + cross-task comparison + pythia-2.8b necessity probe.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    if not args.gate and not args.discover:
        print("Nothing to do: pass --gate or --discover.")
        return

    if args.gate:
        out = run_gate(seed=args.seed, n=args.n)
        Path("results").mkdir(parents=True, exist_ok=True)
        with open("results/rq4_gate.json", "w") as f:
            json.dump(out, f, indent=2)
        print("WROTE results/rq4_gate.json", flush=True)

    if args.discover:
        # run_discover writes results/rq4_circuits.json incrementally itself
        # (partial after Part B, full after Part A) so a crash mid-Part-A
        # doesn't lose Part B's results.
        run_discover(seed=args.seed)


if __name__ == "__main__":
    main()
