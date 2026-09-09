"""RQ4-stretch: EQUATE naturalistic NLI check on Qwen2.5-7B (the one model that
passes the RQ4 NLI behavioral gate; see results/rq4_gate.json).

The submitted paper's Limitations note that RQ4's entailment results rest on a
synthetic comparison-NLI probe (gtlib.nli.make_nli_dataset). This script adds a
naturalistic check using two subsets of EQUATE (Ravichander et al. 2019):
RTE-Quant (166 pairs, entailment/neutral) and AWPNLI (722 pairs,
entailment/contradiction), converted to binary entailed(Yes)/not-entailed(No)
in data/equate_{rtequant,awpnli}.jsonl (see data/README.md for the conversion).

Two evaluations, mirroring experiments/rq4_nli.py's patterns:

1. Behavioral: for each subset, build the EXACT same prompt template as the
   synthetic probe ("Premise: {p} Hypothesis: {h} Question: Is the hypothesis
   true? Answer:") and score the model's signed " Yes"/" No" logit-diff
   (gtlib.nli.nli_logit_diff) with a plain forward loop, hook flags OFF.
   Reports accuracy, mean logit-diff, and a bootstrap CI per subset.

2. Circuit transfer: eap's evaluate_graph requires clean/corrupted pairs of
   EQUAL token length for positional patching. That pairing is easy to
   construct for the synthetic probe (fixed template, single-token numeric
   corruption) but not for naturalistic EQUATE text of widely varying length
   -- there is no principled equal-length "corrupted" counterpart to a
   naturalistic premise/hypothesis pair. We therefore do NOT run
   eap.evaluate.evaluate_graph here. Instead we ask a narrower, still
   meaningful question: are Qwen's own late-layer NLI-circuit MLPs (from
   results/graph_Qwen_Qwen2.5-7B_nli_EAP-IG-inputs.json /
   results/rq4_circuits.json's layer_fractions, nodes with layer_fraction >=
   0.6) behaviorally NECESSARY on naturalistic EQUATE items -- i.e. does
   zero-ablating them hurt EQUATE accuracy/logit-diff more than a
   size-matched random-MLP control ablation? This exactly mirrors
   experiments/rq4_nli.py's necessity_probe_pythia28b pattern (there applied
   to pythia-2.8b's year-circuit vs its own weak NLI signal; here applied to
   Qwen's own NLI circuit vs a genuinely different NLI distribution).
"""
import argparse
import json
import random
import sys
import traceback
from functools import partial
from pathlib import Path

# Run as `python experiments/rq4_equate.py`: sys.path[0] is then experiments/,
# not the project root (same convention as rq4_nli.py / rq3_crossmodel.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_NAME = "Qwen/Qwen2.5-7B"
RQ4_CIRCUITS_PATH = "results/rq4_circuits.json"
RESULTS_PATH = "results/rq4_equate.json"

DATA_PATHS = {
    "rtequant": "data/equate_rtequant.jsonl",
    "awpnli": "data/equate_awpnli.jsonl",
}

# Same template as gtlib.nli.make_nli_dataset's PROMPT_TEMPLATE, adapted to
# naturalistic premise/hypothesis text instead of the synthetic count-comparison
# frame.
PROMPT_TEMPLATE = (
    "Premise: {premise} Hypothesis: {hypothesis} "
    "Question: Is the hypothesis true? Answer:"
)

BEHAVIORAL_BATCH_SIZES = [8, 4, 2]

LATE_MLP_THRESHOLD = 0.6
CONTROL_POOL_THRESHOLD = 0.5
CONTROL_SEED = 42


class AttributionOOM(RuntimeError):
    """Raised when the eval batch-size ladder exhausts without a
    torch.OutOfMemoryError surviving to propagate."""


def load_equate_df(path, tokenizer):
    """Load a converted EQUATE jsonl (premise, hypothesis, label_str) into a
    DataFrame with the columns gtlib.runner.df_to_dataloader expects (clean,
    corrupted, label). `corrupted` is set equal to `clean`: the behavioral
    metric (gtlib.nli.nli_logit_diff) ignores its `clean_logits` argument, so
    this is a harmless placeholder, not a real counterfactual -- there is no
    circuit-patching use of `corrupted` in this script (see module docstring
    on why eap's evaluate_graph is not used here)."""
    import pandas as pd

    yes_ids = tokenizer(" Yes", add_special_tokens=False).input_ids
    no_ids = tokenizer(" No", add_special_tokens=False).input_ids
    assert len(yes_ids) == 1 and len(no_ids) == 1, (
        f"' Yes'/' No' not single-token under this tokenizer: {yes_ids}, {no_ids}"
    )
    yes_id, no_id = yes_ids[0], no_ids[0]

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompt = PROMPT_TEMPLATE.format(premise=d["premise"], hypothesis=d["hypothesis"])
            label = yes_id if d["label_str"] == "Yes" else no_id
            rows.append({
                "clean": prompt,
                "corrupted": prompt,
                "label": label,
                "label_str": d["label_str"],
            })
    return pd.DataFrame(rows, columns=["clean", "corrupted", "label", "label_str"])


def _debug_print_examples(model, df, name, k=2):
    import torch

    for i in range(min(k, len(df))):
        prompt = df.iloc[i]["clean"]
        label_str = df.iloc[i]["label_str"]
        print(f"[{name}] example {i} label={label_str} prompt={prompt!r}", flush=True)
        with torch.no_grad():
            logits = model(prompt, prepend_bos=True)
        last_logits = logits[0, -1]
        top3 = torch.topk(last_logits, 3)
        tokens = [model.tokenizer.decode([tid]) for tid in top3.indices.tolist()]
        print(f"[{name}] example {i} top-3 next tokens: "
              f"{list(zip(tokens, top3.values.tolist()))}", flush=True)


def _zero_hook(value, hook):
    import torch
    return torch.zeros_like(value)


def _forward_eval(model, dataloader, metric, layers=None):
    """Plain TL forward loop over `clean` only (mirrors
    experiments/rq4_nli.py's _eval_with_mlp_ablation), optionally zero-ablating
    the given MLP layers' outputs via run_with_hooks. layers=None runs with no
    ablation (the baseline / no-hooks condition)."""
    import torch
    from eap.utils import tokenize_plus

    metric_partial = partial(metric, loss=False, mean=False)
    fwd_hooks = [(f"blocks.{L}.hook_mlp_out", _zero_hook) for L in (layers or [])]
    results = []
    with torch.inference_mode():
        for clean, corrupted, label in dataloader:
            clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
            r = metric_partial(logits, None, input_lengths, label).cpu()
            results.append(r)
    return torch.cat(results).tolist()


def _eval_with_ladder(model, df, metric, layers=None, batch_sizes=BEHAVIORAL_BATCH_SIZES):
    """Run _forward_eval with a batch-size ladder (halving on CUDA OOM)."""
    import torch
    from gtlib import runner

    last_err = None
    for bs in batch_sizes:
        try:
            dl = runner.df_to_dataloader(df, batch_size=bs)
            return _forward_eval(model, dl, metric, layers=layers), bs
        except torch.OutOfMemoryError as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"OOM at batch_size={bs}, retrying smaller", flush=True)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            continue
    raise AttributionOOM(f"eval OOM at all batch sizes {batch_sizes}: {last_err}")


def _summarize(values, seed):
    from gtlib import stats

    accuracy = sum(1 for v in values if v > 0) / len(values)
    ci = stats.bootstrap_ci(values, seed=seed)
    return {
        "n": len(values),
        "accuracy": accuracy,
        "mean_logit_diff": float(sum(values) / len(values)),
        "ci": ci,
    }


def _load_late_and_control_layers():
    """Parse results/rq4_circuits.json's qwen_nli.layer_fractions (already
    computed and saved by experiments/rq4_nli.py:discover_qwen_nli) for MLP
    nodes with layer_fraction >= LATE_MLP_THRESHOLD (the circuit's late MLPs)
    and a same-size random control drawn from MLP nodes with layer_fraction <
    CONTROL_POOL_THRESHOLD -- identical scheme to rq4_nli.py's
    necessity_probe_pythia28b."""
    with open(RQ4_CIRCUITS_PATH) as f:
        rq4 = json.load(f)
    qwen_nli = rq4["qwen_nli"]
    assert qwen_nli.get("model_used") == MODEL_NAME, (
        f"results/rq4_circuits.json qwen_nli.model_used={qwen_nli.get('model_used')!r} "
        f"!= expected {MODEL_NAME!r}; the circuit graph would not correspond to the "
        "model being evaluated here."
    )
    layer_fractions = qwen_nli["layer_fractions"]
    late_mlps = sorted(
        int(n[1:]) for n, frac in layer_fractions.items()
        if n.startswith("m") and frac >= LATE_MLP_THRESHOLD
    )
    control_pool = sorted(
        int(n[1:]) for n, frac in layer_fractions.items()
        if n.startswith("m") and frac < CONTROL_POOL_THRESHOLD
    )
    rng = random.Random(CONTROL_SEED)
    control_layers = sorted(rng.sample(control_pool, min(len(late_mlps), len(control_pool))))
    return late_mlps, control_layers


CIRCUIT_TRANSFER_NOTE = (
    "eap.evaluate.evaluate_graph requires clean/corrupted pairs of EQUAL token "
    "length for positional patching. That pairing is constructible for the "
    "synthetic comparison-NLI probe (fixed template, single-token numeric "
    "corruption) but not for naturalistic EQUATE premise/hypothesis text of "
    "widely varying length -- there is no principled equal-length 'corrupted' "
    "counterpart to a naturalistic pair. We therefore do not run "
    "evaluate_graph here. Instead we test necessity of Qwen's own "
    f"NLI-circuit late MLPs (layer_fraction >= {LATE_MLP_THRESHOLD}, from "
    "results/rq4_circuits.json's qwen_nli.layer_fractions / "
    "results/graph_Qwen_Qwen2.5-7B_nli_EAP-IG-inputs.json) on naturalistic "
    "EQUATE items via hook-based zero-ablation, compared against a "
    f"size-matched random control drawn from MLPs with layer_fraction < "
    f"{CONTROL_POOL_THRESHOLD} (identical scheme to "
    "experiments/rq4_nli.py:necessity_probe_pythia28b)."
)

CIRCUIT_TRANSFER_CAVEAT = (
    "CAVEAT: the size-matched control pool (layer_fraction < "
    f"{CONTROL_POOL_THRESHOLD}) includes early MLPs (layers 0-2), which are "
    "generically disruptive to ablate irrespective of circuit membership; "
    "the control is therefore conservative and the 'necessary but not "
    "selective' reading of these numbers is tentative."
)


def run(seed: int = 42) -> dict:
    import torch
    from gtlib import runner
    from gtlib.nli import nli_logit_diff

    Path("results").mkdir(parents=True, exist_ok=True)
    out = {"model": MODEL_NAME, "note": CIRCUIT_TRANSFER_NOTE,
           "caveat": CIRCUIT_TRANSFER_CAVEAT}

    print(f"[rq4-equate] loading {MODEL_NAME}...", flush=True)
    model = runner.load_model(MODEL_NAME)
    # Behavioral eval only needs plain forward passes; disable the attribution
    # hook flags load_model sets (same rationale as rq4_nli.py's run_gate).
    model.cfg.use_split_qkv_input = False
    model.cfg.use_attn_result = False
    model.cfg.use_hook_mlp_in = False

    try:
        metric = nli_logit_diff(model.tokenizer)

        dfs = {}
        for subset, path in DATA_PATHS.items():
            df = load_equate_df(path, model.tokenizer)
            dfs[subset] = df
            print(f"[rq4-equate] {subset}: n={len(df)}", flush=True)
            _debug_print_examples(model, df, subset)

        # --- 1. behavioral eval per subset ---
        subset_results = {}
        baseline_values = {}
        for subset, df in dfs.items():
            values, bs_used = _eval_with_ladder(model, df, metric, layers=None)
            baseline_values[subset] = values
            summary = _summarize(values, seed)
            subset_results[subset] = summary
            print(f"[rq4-equate] {subset} behavioral: "
                  f"accuracy={summary['accuracy']:.4f} "
                  f"mean_logit_diff={summary['mean_logit_diff']:.4f} "
                  f"(batch_size={bs_used})", flush=True)
        out["subsets"] = subset_results

        # Save partial results immediately: circuit-transfer below depends on
        # results/rq4_circuits.json and could fail/skip independently.
        with open(RESULTS_PATH, "w") as f:
            json.dump(out, f, indent=2)
        print(f"WROTE partial {RESULTS_PATH} (behavioral only)", flush=True)

        # --- 2. circuit-transfer (late-MLP ablation vs random control) ---
        try:
            late_mlps, control_layers = _load_late_and_control_layers()
            print(f"[rq4-equate] late_mlps={late_mlps}", flush=True)
            print(f"[rq4-equate] control_layers={control_layers}", flush=True)

            circuit_transfer = {
                "method": "hook_based_zero_ablation_transfer",
                "note": CIRCUIT_TRANSFER_NOTE,
                "late_mlps": late_mlps,
                "control_layers": control_layers,
                "subsets": {},
            }
            for subset, df in dfs.items():
                baseline_summary = _summarize(baseline_values[subset], seed)

                late_values, _ = _eval_with_ladder(model, df, metric, layers=late_mlps)
                late_summary = _summarize(late_values, seed)
                print(f"[rq4-equate] {subset} late_mlp_ablated: "
                      f"accuracy={late_summary['accuracy']:.4f} "
                      f"mean_logit_diff={late_summary['mean_logit_diff']:.4f}", flush=True)

                control_values, _ = _eval_with_ladder(model, df, metric, layers=control_layers)
                control_summary = _summarize(control_values, seed)
                print(f"[rq4-equate] {subset} control_ablated: "
                      f"accuracy={control_summary['accuracy']:.4f} "
                      f"mean_logit_diff={control_summary['mean_logit_diff']:.4f}", flush=True)

                circuit_transfer["subsets"][subset] = {
                    "baseline": baseline_summary,
                    "late_mlp_ablated": late_summary,
                    "control_ablated": control_summary,
                }
            out["circuit_transfer"] = circuit_transfer
        except (FileNotFoundError, KeyError, AssertionError, AttributionOOM) as e:
            print(f"[rq4-equate] circuit-transfer SKIPPED: {e}", flush=True)
            traceback.print_exc()
            out["circuit_transfer"] = {"skipped": str(e)}

        with open(RESULTS_PATH, "w") as f:
            json.dump(out, f, indent=2)
        print(f"WROTE {RESULTS_PATH}", flush=True)
        return out
    finally:
        del model
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(
        description="RQ4-stretch: EQUATE naturalistic NLI behavioral + circuit-transfer check on Qwen2.5-7B."
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(seed=args.seed)


if __name__ == "__main__":
    main()
