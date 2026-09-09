"""E3: does the Yes/No prompt format, not numeracy, explain RQ4's null?

Reviewer CxwV objected that GPT-2 and Pythia at chance on the numerical
entailment probe may simply be unable to use the Yes/No format, in which case
the reported double dissociation partly restates "small models cannot do NLI
and Qwen tokenizes differently". This script runs two digit-free probes with
scaffolding shared with the numerical probe across every model:

  identity: lexical identity, strictly easier than numerical comparison, so
            chance performance here is decisive evidence of format inability.
  ordinal:  the same greater-than structure over alphabet positions, which
            separates numeracy from comparison ability.

Accuracy is defined exactly as in rq4_nli.py (the fraction of items whose
signed logit difference is positive), so the numbers are comparable to the
numerical probe's.

Writes results/rq4_format_control.json.
"""
import argparse
import json
import sys
from pathlib import Path

# Run as `python experiments/rq4_format_control.py`: sys.path[0] is then
# experiments/, not the project root, so gtlib isn't importable without this.
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

BATCH_SIZES = {"pythia-2.8b": 25, "pythia-6.9b": 8, "pythia-12b": 4,
               "Qwen/Qwen2.5-7B": 8}


def run(n: int, seed: int) -> dict:
    """Evaluates both format-control probes on every model.

    Args:
        n: Items per probe.
        seed: Dataset sampling seed.

    Returns:
        The results dict written to results/rq4_format_control.json.
    """
    import torch

    from gtlib import runner, stats
    from gtlib.format_control import (make_identity_dataset,
                                      make_ordinal_dataset)
    from gtlib.nli import nli_logit_diff

    builders = {"identity": make_identity_dataset,
                "ordinal": make_ordinal_dataset}

    out = {}
    for name in MODELS:
        print(f"[{name}] loading...", flush=True)
        model = None
        try:
            model = runner.load_model(name)
            metric = nli_logit_diff(model.tokenizer)
            batch_size = BATCH_SIZES.get(name, 100)

            per_probe = {}
            for probe, build in builders.items():
                df = build(n=n, seed=seed, tokenizer=model.tokenizer)
                dl = runner.df_to_dataloader(df, batch_size=batch_size)
                values = runner.evaluate_baseline_values(
                    model, dl, metric
                ).tolist()

                # nli_logit_diff negates for "No" items, so a positive value
                # always means the model favored the correct answer. Same
                # definition as rq4_nli.py:54.
                accuracy = sum(1 for v in values if v > 0) / len(values)
                per_probe[probe] = {
                    "n": len(df),
                    "accuracy": accuracy,
                    "mean_logit_diff": sum(values) / len(values),
                    "ci": stats.bootstrap_ci(values, seed=seed),
                }
                print(f"[{name}] {probe}: acc={accuracy:.4f} "
                      f"mean_logit_diff={sum(values) / len(values):.4f}",
                      flush=True)

            out[name] = per_probe
        except Exception as exc:
            out[name] = {"error": str(exc)}
            print(f"[{name}] ERROR: {exc}", flush=True)
        finally:
            if model is not None:
                del model
                torch.cuda.empty_cache()

        # Written after every model so a late OOM does not lose earlier work.
        Path("results").mkdir(parents=True, exist_ok=True)
        with open("results/rq4_format_control.json", "w") as f:
            json.dump({"models": out}, f, indent=2)

    return {"models": out}


def main():
    """Parses arguments and writes results/rq4_format_control.json."""
    ap = argparse.ArgumentParser(
        description="E3: non-numerical format controls for RQ4."
    )
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run(n=args.n, seed=args.seed)
    print("WROTE results/rq4_format_control.json", flush=True)


if __name__ == "__main__":
    main()
