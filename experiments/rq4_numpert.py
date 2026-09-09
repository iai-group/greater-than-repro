"""RQ4-stretch (P16): NumPert naturalistic perturbation study on Qwen2.5-7B
(the one model that passes the RQ4 NLI behavioral gate; see results/rq4_gate.json).

NumPert (Aarnes & Setty, IJCNLP-AACL 2025 SRW, arXiv:2511.09971) applies six
rule-based numerical perturbations to QuanTemp claim-evidence pairs (see
data/README.md / data/convert_numpert.py for the exact upstream schema and
conversion). This script mirrors experiments/rq4_equate.py's patterns
(prompt template, logit-diff metric, batch-size ladder, late-MLP-vs-control
necessity probe) and adds three parts:

A. Behavioral (hook flags OFF): zero-shot signed Yes/No logit-diff accuracy
   on the unperturbed originals and on each of the six perturbation types
   separately, reporting each type's accuracy delta vs. originals
   (perturbation robustness).

B. Positional faithfulness on NATURALISTIC minimal pairs (the novel bit):
   unlike EQUATE's naturalistic text (no principled equal-length
   "corrupted" counterpart -- see rq4_equate.py's docstring for why
   eap.evaluate.evaluate_graph is not used there), NumPert's Rand-Repl and
   Masking perturbations are constructed by design to preserve digit COUNT,
   which under Qwen's per-digit tokenizer often preserves total token
   length. We build clean (original True claim, label Yes) / corrupted
   (Rand-Repl or Masking perturbed version, label flipped to No) pairs,
   keep only pairs with EXACTLY equal token length under Qwen's tokenizer,
   and if >=50 pairs survive, evaluate Qwen's own synthetic-NLI circuit
   (results/graph_Qwen_Qwen2.5-7B_nli_EAP-IG-inputs.json, loaded via
   eap.graph.Graph.from_json) on this naturalistic dataloader via
   eap.evaluate.evaluate_graph (hook flags ON,
   model.cfg.ungroup_grouped_query_attention=True), reporting baseline vs.
   circuit performance vs. faithfulness. If <50 pairs survive for a type, we
   record the counts and skip that type's positional-faithfulness eval with
   an honest note (mirrors rq4_equate.py's honest-skip style).

C. Ablation transfer (identical scheme to rq4_equate.py's circuit-transfer):
   does zero-ablating Qwen's own NLI-circuit late MLPs (layer_fraction >=
   0.6, from results/rq4_circuits.json's qwen_nli.layer_fractions) hurt
   NumPert-originals accuracy more than a size-matched random-MLP control
   ablation?

Evidence ("premise") can be long (NumPert evidence is claim-specific
web-scraped fact-check text, up to ~23k characters); premises are truncated
to fit a 1024-token Qwen prompt budget (template + hypothesis overhead
computed per-item), with truncation/drop counts recorded per subset in
results/rq4_numpert.json's "truncation_stats".
"""
import argparse
import json
import random
import sys
import traceback
from functools import partial
from pathlib import Path

# Run as `python experiments/rq4_numpert.py`: sys.path[0] is then experiments/,
# not the project root (same convention as rq4_nli.py / rq4_equate.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_NAME = "Qwen/Qwen2.5-7B"
RQ4_CIRCUITS_PATH = "results/rq4_circuits.json"
RQ4_NLI_GRAPH_PATH = "results/graph_Qwen_Qwen2.5-7B_nli_EAP-IG-inputs.json"
RESULTS_PATH = "results/rq4_numpert.json"

ORIGINAL_PATH = "data/numpert_original.jsonl"
PERTURB_PATHS = {
    "numeration": "data/numpert_numeration.jsonl",
    "approximation": "data/numpert_approximation.jsonl",
    "range": "data/numpert_range.jsonl",
    "masking": "data/numpert_masking.jsonl",
    "randrepl": "data/numpert_randrepl.jsonl",
    "negnum": "data/numpert_negnum.jsonl",
}

# Same template as gtlib.nli.make_nli_dataset's PROMPT_TEMPLATE / rq4_equate.py.
PROMPT_TEMPLATE = (
    "Premise: {premise} Hypothesis: {hypothesis} "
    "Question: Is the hypothesis true? Answer:"
)

MAX_PROMPT_TOKENS = 1024
# Positional (eap evaluate_graph) prompts use a much tighter budget: TL's
# split-qkv attention materializes a per-position*per-head product whose
# memory scales ~linearly with sequence length (49 GiB at 1024 tokens even
# at batch 1 on an 80GB card). 320 tokens fits comfortably; behavioral
# results above are unaffected (they use plain forwards at 1024).
POSITIONAL_MAX_TOKENS = 320
BEHAVIORAL_BATCH_SIZES = [8, 4, 2]
POSITIONAL_BATCH_SIZES = [8, 4, 2, 1]

ABLATION_CAVEAT = (
    "CAVEAT: the size-matched control pool (layer_fraction < 0.5) includes early "
    "MLPs (layers 0-2), which are generically disruptive to ablate irrespective "
    "of circuit membership; the control is therefore conservative/confounded. In "
    "particular, the late-MLP-ablation ACCURACY IMPROVEMENT on originals should "
    "not be read as 'late MLPs are harmful' without ruling out response-bias "
    "shifts (ablation increases the Yes-side logit mass)."
)
MIN_PAIRS_FOR_POSITIONAL = 50
# Which perturbation types to attempt positional-faithfulness pairing on
# (both preserve digit COUNT by construction, per data/README.md).
POSITIONAL_CANDIDATE_TYPES = ["randrepl", "masking"]

LATE_MLP_THRESHOLD = 0.6
CONTROL_POOL_THRESHOLD = 0.5
CONTROL_SEED = 42

DEBUG_FLIP_TYPE = "masking"  # all-flip type; used for the "one flip-type" debug print


class AttributionOOM(RuntimeError):
    """Raised when the eval batch-size ladder exhausts without a
    torch.OutOfMemoryError surviving to propagate."""


# ---------------------------------------------------------------------------
# Data loading + truncation
# ---------------------------------------------------------------------------

def _read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prompt_overhead_tokens(tokenizer, hypothesis):
    """Token count of the template with an empty premise -- the fixed
    overhead that must be subtracted from MAX_PROMPT_TOKENS before we know
    how many tokens are available for the premise."""
    empty_prompt = PROMPT_TEMPLATE.format(premise="", hypothesis=hypothesis)
    return len(tokenizer(empty_prompt, add_special_tokens=False).input_ids)


def _truncate_premise(tokenizer, premise, hypothesis, max_len=MAX_PROMPT_TOKENS):
    """Truncate `premise` (token-wise) so the full prompt fits max_len tokens.
    Returns (premise_text_or_None, was_truncated). None means even an empty
    premise wouldn't fit (drop the item)."""
    overhead = _prompt_overhead_tokens(tokenizer, hypothesis)
    available = max_len - overhead
    if available <= 0:
        return None, True
    premise_ids = tokenizer(premise, add_special_tokens=False).input_ids
    if len(premise_ids) > available:
        return tokenizer.decode(premise_ids[:available]), True
    return premise, False


def load_numpert_df(path, tokenizer, perturb_type=None):
    """Load a converted NumPert jsonl into a DataFrame with the columns
    gtlib.runner.df_to_dataloader expects (clean, corrupted, label).
    `corrupted` is set equal to `clean` (behavioral eval only): the metric
    (gtlib.nli.nli_logit_diff) ignores its `clean_logits` argument, same
    placeholder convention as rq4_equate.py's load_equate_df."""
    import pandas as pd

    yes_ids = tokenizer(" Yes", add_special_tokens=False).input_ids
    no_ids = tokenizer(" No", add_special_tokens=False).input_ids
    assert len(yes_ids) == 1 and len(no_ids) == 1, (
        f"' Yes'/' No' not single-token under this tokenizer: {yes_ids}, {no_ids}"
    )
    yes_id, no_id = yes_ids[0], no_ids[0]

    rows = []
    n_truncated = 0
    n_dropped = 0
    for d in _read_jsonl(path):
        premise, truncated = _truncate_premise(tokenizer, d["premise"], d["hypothesis"])
        if premise is None:
            n_dropped += 1
            continue
        if truncated:
            n_truncated += 1
        prompt = PROMPT_TEMPLATE.format(premise=premise, hypothesis=d["hypothesis"])
        label = yes_id if d["label_str"] == "Yes" else no_id
        rows.append({
            "clean": prompt, "corrupted": prompt, "label": label,
            "label_str": d["label_str"], "orig_id": d.get("orig_id"),
            "flip": d.get("flip"),
        })
    df = pd.DataFrame(rows, columns=["clean", "corrupted", "label", "label_str", "orig_id", "flip"])
    stats = {
        "n_total": len(df) + n_dropped,
        "n_kept": len(df),
        "n_truncated": n_truncated,
        "n_dropped_no_room": n_dropped,
    }
    return df, stats


def _truncate_premise_for_pair(tokenizer, premise, hyp_a, hyp_b, max_len=MAX_PROMPT_TOKENS):
    """Truncate `premise` once, sized so BOTH hyp_a's and hyp_b's prompts fit
    -- since the premise text is identical between clean/corrupted, this
    keeps the premise portion byte-identical across the pair (only the
    hypothesis differs), which is required for the equal-token-length check
    downstream to be meaningful rather than an artifact of independent
    truncation cutoffs."""
    overhead = max(_prompt_overhead_tokens(tokenizer, hyp_a), _prompt_overhead_tokens(tokenizer, hyp_b))
    available = max_len - overhead
    if available <= 0:
        return None, True
    premise_ids = tokenizer(premise, add_special_tokens=False).input_ids
    if len(premise_ids) > available:
        return tokenizer.decode(premise_ids[:available]), True
    return premise, False


def build_pair_df(perturb_path, original_by_id, tokenizer, max_len=MAX_PROMPT_TOKENS):
    """Build clean (original True claim, label Yes) / corrupted (perturbed
    claim, label flipped to No) pairs for positional faithfulness, keeping
    only pairs with EXACTLY equal token length under `tokenizer` (required
    for eap's positional patching; see module docstring)."""
    import pandas as pd

    yes_id = tokenizer(" Yes", add_special_tokens=False).input_ids[0]

    stats = {
        "n_flip_candidates": 0, "n_missing_orig": 0, "n_cannot_fit": 0,
        "n_truncated": 0, "n_length_mismatch": 0, "n_kept": 0,
    }
    rows = []
    for d in _read_jsonl(perturb_path):
        if not d.get("flip"):
            continue
        stats["n_flip_candidates"] += 1
        orig = original_by_id.get(d["orig_id"])
        if orig is None:
            stats["n_missing_orig"] += 1
            continue
        premise = orig["premise"]
        hyp_clean = orig["hypothesis"]
        hyp_corrupt = d["hypothesis"]
        premise_trunc, truncated = _truncate_premise_for_pair(tokenizer, premise, hyp_clean, hyp_corrupt, max_len)
        if premise_trunc is None:
            stats["n_cannot_fit"] += 1
            continue
        if truncated:
            stats["n_truncated"] += 1
        clean_prompt = PROMPT_TEMPLATE.format(premise=premise_trunc, hypothesis=hyp_clean)
        corrupt_prompt = PROMPT_TEMPLATE.format(premise=premise_trunc, hypothesis=hyp_corrupt)
        clean_len = len(tokenizer(clean_prompt, add_special_tokens=False).input_ids)
        corrupt_len = len(tokenizer(corrupt_prompt, add_special_tokens=False).input_ids)
        if clean_len != corrupt_len:
            stats["n_length_mismatch"] += 1
            continue
        rows.append({"clean": clean_prompt, "corrupted": corrupt_prompt, "label": yes_id, "label_str": "Yes"})
        stats["n_kept"] += 1

    df = pd.DataFrame(rows, columns=["clean", "corrupted", "label", "label_str"])
    return df, stats


# ---------------------------------------------------------------------------
# Debug printing (mirrors rq4_equate.py's _debug_print_examples)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Ablation + eval-ladder helpers (copied verbatim from rq4_equate.py, kept in
# sync by inspection, not import, per that script's own convention)
# ---------------------------------------------------------------------------

def _zero_hook(value, hook):
    import torch
    return torch.zeros_like(value)


def _forward_eval(model, dataloader, metric, layers=None):
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
    """Identical scheme to rq4_equate.py's _load_late_and_control_layers /
    rq4_nli.py's necessity_probe_pythia28b: MLP nodes with layer_fraction >=
    LATE_MLP_THRESHOLD from results/rq4_circuits.json's qwen_nli circuit,
    vs. a same-size random control from layer_fraction < CONTROL_POOL_THRESHOLD."""
    with open(RQ4_CIRCUITS_PATH) as f:
        rq4 = json.load(f)
    qwen_nli = rq4["qwen_nli"]
    assert qwen_nli.get("model_used") == MODEL_NAME, (
        f"results/rq4_circuits.json qwen_nli.model_used={qwen_nli.get('model_used')!r} "
        f"!= expected {MODEL_NAME!r}."
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


# ---------------------------------------------------------------------------
# Part B: positional faithfulness
# ---------------------------------------------------------------------------

def _run_positional_faithfulness(model, df, metric):
    """Load Qwen's own synthetic-NLI circuit snapshot and evaluate it (via
    eap.evaluate.evaluate_graph, default 'patching' intervention) against the
    full-model baseline on `df`'s naturalistic minimal pairs."""
    import gc
    import torch
    from eap.graph import Graph
    from gtlib import runner

    g = Graph.from_json(RQ4_NLI_GRAPH_PATH)
    last_err = None
    for bs in POSITIONAL_BATCH_SIZES:
        try:
            dl = runner.df_to_dataloader(df, batch_size=bs)
            baseline_vals = runner.evaluate_baseline_values(model, dl, metric)
            baseline = float(baseline_vals.mean())
            circuit_perf = runner.evaluate_graph_mean(model, g, dl, metric)
            faithfulness = circuit_perf / baseline if baseline else float("nan")
            return {
                "n_pairs": len(df), "batch_size": bs,
                "baseline": baseline, "circuit_perf": circuit_perf,
                "faithfulness": faithfulness,
            }
        except torch.OutOfMemoryError as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"[positional] OOM at batch_size={bs}, retrying smaller", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            continue
    raise AttributionOOM(f"positional faithfulness OOM at all batch sizes {POSITIONAL_BATCH_SIZES}: {last_err}")


POSITIONAL_NOTE = (
    "Clean = original True claim prompt (label Yes); corrupted = its "
    "Rand-Repl/Masking perturbed version (label flipped to No), premise held "
    "identical between clean/corrupted (truncated once per pair so both fit "
    "the 1024-token budget). Pairs are kept only if the FULL prompt "
    "(template + premise + hypothesis) tokenizes to EXACTLY equal length "
    "under Qwen's tokenizer -- required for eap.evaluate.evaluate_graph's "
    "positional patching. If fewer than "
    f"{MIN_PAIRS_FOR_POSITIONAL} pairs survive for a perturbation type, "
    "that type's circuit evaluation is skipped with the pairing counts "
    "recorded instead."
)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(seed: int = 42) -> dict:
    import gc
    import torch
    from gtlib import runner
    from gtlib.nli import nli_logit_diff

    Path("results").mkdir(parents=True, exist_ok=True)
    out = {"model": MODEL_NAME, "positional_faithfulness_note": POSITIONAL_NOTE}

    print(f"[rq4-numpert] loading {MODEL_NAME}...", flush=True)
    model = runner.load_model(MODEL_NAME)

    try:
        metric = nli_logit_diff(model.tokenizer)

        # --- A. behavioral (hook flags OFF) ---
        model.cfg.use_split_qkv_input = False
        model.cfg.use_attn_result = False
        model.cfg.use_hook_mlp_in = False

        subset_paths = {"original": ORIGINAL_PATH, **PERTURB_PATHS}
        dfs = {}
        truncation_stats = {}
        behavioral = {}
        original_baseline_values = None

        for name, path in subset_paths.items():
            df, tstats = load_numpert_df(path, model.tokenizer)
            dfs[name] = df
            truncation_stats[name] = tstats
            print(f"[rq4-numpert] {name}: n={len(df)} truncation_stats={tstats}", flush=True)

        _debug_print_examples(model, dfs["original"], "original")
        _debug_print_examples(model, dfs[DEBUG_FLIP_TYPE], DEBUG_FLIP_TYPE)

        for name, df in dfs.items():
            values, bs_used = _eval_with_ladder(model, df, metric, layers=None)
            if name == "original":
                original_baseline_values = values
            summary = _summarize(values, seed)
            behavioral[name] = summary
            print(f"[rq4-numpert] {name} behavioral: accuracy={summary['accuracy']:.4f} "
                  f"mean_logit_diff={summary['mean_logit_diff']:.4f} (batch_size={bs_used})", flush=True)

        original_acc = behavioral["original"]["accuracy"]
        for name in PERTURB_PATHS:
            behavioral[name]["accuracy_delta_vs_original"] = behavioral[name]["accuracy"] - original_acc

        out["behavioral"] = behavioral
        out["truncation_stats"] = truncation_stats
        with open(RESULTS_PATH, "w") as f:
            json.dump(out, f, indent=2)
        print(f"WROTE partial {RESULTS_PATH} (behavioral only)", flush=True)

        # --- B. positional faithfulness (hook flags ON) ---
        positional = {}
        try:
            original_rows = _read_jsonl(ORIGINAL_PATH)
            original_by_id = {r["orig_id"]: r for r in original_rows}

            model.cfg.use_split_qkv_input = True
            model.cfg.use_attn_result = True
            model.cfg.use_hook_mlp_in = True
            if model.cfg.n_key_value_heads is not None:
                model.cfg.ungroup_grouped_query_attention = True

            for ptype in POSITIONAL_CANDIDATE_TYPES:
                pair_df, pair_stats = build_pair_df(PERTURB_PATHS[ptype], original_by_id, model.tokenizer,
                                                    max_len=POSITIONAL_MAX_TOKENS)
                print(f"[rq4-numpert] positional[{ptype}] pairing_stats={pair_stats}", flush=True)
                if pair_stats["n_kept"] < MIN_PAIRS_FOR_POSITIONAL:
                    positional[ptype] = {
                        "skipped": f"only {pair_stats['n_kept']} pairs survived "
                                   f"(< {MIN_PAIRS_FOR_POSITIONAL} required)",
                        "pairing_stats": pair_stats,
                    }
                    print(f"[rq4-numpert] positional[{ptype}] SKIPPED: too few pairs", flush=True)
                    continue
                try:
                    result = _run_positional_faithfulness(model, pair_df, metric)
                    result["pairing_stats"] = pair_stats
                    positional[ptype] = result
                    print(f"[rq4-numpert] positional[{ptype}] baseline={result['baseline']:.4f} "
                          f"circuit_perf={result['circuit_perf']:.4f} "
                          f"faithfulness={result['faithfulness']:.4f} (n={result['n_pairs']})", flush=True)
                except AttributionOOM as e:
                    positional[ptype] = {"skipped": str(e), "pairing_stats": pair_stats}
                    print(f"[rq4-numpert] positional[{ptype}] SKIPPED (OOM): {e}", flush=True)
        except (FileNotFoundError, KeyError, AssertionError) as e:
            print(f"[rq4-numpert] positional faithfulness SKIPPED entirely: {e}", flush=True)
            traceback.print_exc()
            positional = {"skipped": str(e)}

        out["positional_faithfulness"] = positional
        with open(RESULTS_PATH, "w") as f:
            json.dump(out, f, indent=2)
        print(f"WROTE partial {RESULTS_PATH} (behavioral + positional)", flush=True)

        # --- C. ablation transfer (hook flags OFF again) ---
        model.cfg.use_split_qkv_input = False
        model.cfg.use_attn_result = False
        model.cfg.use_hook_mlp_in = False

        try:
            late_mlps, control_layers = _load_late_and_control_layers()
            print(f"[rq4-numpert] late_mlps={late_mlps}", flush=True)
            print(f"[rq4-numpert] control_layers={control_layers}", flush=True)

            baseline_summary = _summarize(original_baseline_values, seed)

            late_values, _ = _eval_with_ladder(model, dfs["original"], metric, layers=late_mlps)
            late_summary = _summarize(late_values, seed)
            print(f"[rq4-numpert] original late_mlp_ablated: accuracy={late_summary['accuracy']:.4f} "
                  f"mean_logit_diff={late_summary['mean_logit_diff']:.4f}", flush=True)

            control_values, _ = _eval_with_ladder(model, dfs["original"], metric, layers=control_layers)
            control_summary = _summarize(control_values, seed)
            print(f"[rq4-numpert] original control_ablated: accuracy={control_summary['accuracy']:.4f} "
                  f"mean_logit_diff={control_summary['mean_logit_diff']:.4f}", flush=True)

            out["ablation_transfer"] = {
                "method": "hook_based_zero_ablation_transfer",
                "caveat": ABLATION_CAVEAT,
                "late_mlps": late_mlps,
                "control_layers": control_layers,
                "baseline": baseline_summary,
                "late_mlp_ablated": late_summary,
                "control_ablated": control_summary,
            }
        except (FileNotFoundError, KeyError, AssertionError, AttributionOOM) as e:
            print(f"[rq4-numpert] ablation transfer SKIPPED: {e}", flush=True)
            traceback.print_exc()
            out["ablation_transfer"] = {"skipped": str(e)}

        out["notes"] = [
            "Originals set: 879 items (264 True + 615 False) -- the numeric-taggable "
            "subset of QuanTemp TEST that NumPert actually perturbs; see data/README.md.",
            "value_removal (a 3rd obfuscation sub-type) was skipped -- not one of the "
            "task brief's six perturbation types.",
            POSITIONAL_NOTE,
        ]

        with open(RESULTS_PATH, "w") as f:
            json.dump(out, f, indent=2)
        print(f"WROTE {RESULTS_PATH}", flush=True)
        return out
    finally:
        del model
        import torch as _torch
        _torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(
        description="RQ4-stretch (P16): NumPert naturalistic perturbation study on Qwen2.5-7B."
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(seed=args.seed)


if __name__ == "__main__":
    main()
