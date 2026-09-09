"""One-off conversion script: NumPert (Aarnes & Setty, IJCNLP-AACL 2025 SRW,
arXiv:2511.09971) numerical-claim perturbations -> binary entailed(Yes)/
not-entailed(No) JSONL for RQ4-stretch (P16, naturalistic perturbation study).

Source: github.com/iai-group/adversarial_attack_numerical_claims (senior
author Setty is this project's owner and has directed this use; see
data/README.md's provenance note -- there is no LICENSE file upstream).

Usage (one-off; not run by any experiment script):
    python data/convert_numpert.py /path/to/cloned/adversarial_attack_numerical_claims data/

INSPECTED SOURCE SCHEMA (verified against the actual cloned repo, not assumed):

  data/preprocessed/tagged/{true,false}_claims_tagged_normalized_numbers.json
      264 True-labeled / 615 False-labeled claim dicts (fields include
      "claim_id" (unique per file, NOT globally unique -- 1..264 for the True
      file, 1..615 for the False file), "claim", "doc" (claim-specific
      evidence text), "label" ("True"/"False"), "url", ...). This is the
      "originals" set used here: it is the numeric-taggable subset that
      main_perturb.py actually perturbs (a claim needs a tagged numeric span
      to be perturbable), NOT the full QuanTemp binary test split (1897
      items, 474 True/1423 False). 857/879 of these claims' URLs also appear
      in data/raw_data/binary_data/test_binary.json (vs. 52/22 in
      train/val), confirming this subset is TEST-derived, matching the task
      brief's target of "260 True / 604 False" (264/615 found here; the
      small discrepancy is presumably a minor dataset-version difference --
      documented, not resolved by guessing). code/data_preprocessing/
      raw_dataset_preprocessing/remove_referance.py (verdict-sentence
      stripping) is upstream of this tagged file per the repo's own pipeline
      description; spot-checking found it byte-identical to the raw
      test_binary.json "doc" for 875/889 matched-by-URL items (most evidence
      apparently never contained a trailing verdict sentence matching the
      regex), so no further stripping is applied here.

  data/perturbed_data/{true,false}_dataset/jsons/{numeration,approximation,
  range,negative,obfuscation}_{true,false}.json
      Perturbed-claim dicts, one row per ORIGINAL claim, carrying the same
      "claim_id"/"doc"/"url" as the corresponding tagged-originals file (the
      *_true.json files are built from the 264 True originals; the
      *_false.json files from the 615 False originals) plus one or more
      perturbed-claim-text fields:
        numeration_{true,false}.json: keys "numeration_true" (perturbed
            claim, label stays as origin) and "numeration_false" (perturbed
            claim, label flips to opposite) -- BOTH present in both files
            (e.g. numeration applied to a False original can flip it to
            "numeration_true" (F->T) or keep "numeration_false" (F->F)).
        approximation_{true,false}.json, range_{true,false}.json: same
            two-key (both-direction) structure as numeration.
        obfuscation_true.json: keys "random_replacement_false" (Rand-Repl,
            True->False), "masking_false" (Masking, True->False), and
            "value_removal_false" (a 3rd obfuscation sub-type NOT in the
            task brief's six -- skipped, see SKIPPED_TYPES below).
        obfuscation_false.json: key "random_replacement_false" only
            (Rand-Repl applied to a False original stays False; no masking
            or value_removal key present for False originals in this repo
            snapshot).
        negative_true.json: key "negative_number_false" (Neg-Num,
            True->False).
        negative_false.json: key "negative_number_false" (Neg-Num applied to
            a False original stays False).

Six perturbation types kept per the task brief, and their source mapping:
    numeration    <- numeration_true.json + numeration_false.json (both dirs)
    approximation <- approximation_true.json + approximation_false.json
    range         <- range_true.json + range_false.json
    masking       <- obfuscation_true.json["masking_false"] only (True-origin
                     only, per brief; not materialized for False originals)
    randrepl      <- obfuscation_true.json["random_replacement_false"]
                     + obfuscation_false.json["random_replacement_false"]
    negnum        <- negative_true.json["negative_number_false"]
                     + negative_false.json["negative_number_false"]

SKIPPED_TYPES: "value_removal" (obfuscation's 3rd sub-type) is materialized in
the repo but is not one of the task brief's six perturbation types -- skipped
here, noted in data/README.md.

Mapping to NLI framing (premise=evidence, hypothesis=claim, label:
True->Yes, False->No): each output row is
    {"premise": <doc>, "hypothesis": <perturbed or original claim text>,
     "label_str": "Yes"|"No", "perturb_type": <name>, "orig_id": str,
     "orig_label_str": "Yes"|"No", "flip": bool}
"orig_id" = f"{origin}_{claim_id}" (origin in {"true","false"}, claim_id from
the source file) -- claim_id is only unique within one origin file, hence the
prefix. "orig_label_str"/"flip" let downstream code separate flip vs no-flip
variants (see task brief's per-type behavioral breakdown).

Premise truncation to Qwen's 1024-token budget is NOT done here (this script
just extracts full text) -- it happens at eval time in
experiments/rq4_numpert.py, which records truncation counts into
results/rq4_numpert.json's "truncation_stats".
"""
import argparse
import json
from pathlib import Path

TAGGED_PATHS = {
    "true": "data/preprocessed/tagged/true_claims_tagged_normalized_numbers.json",
    "false": "data/preprocessed/tagged/false_claims_tagged_normalized_numbers.json",
}

# (perturb_type, [(origin, relative_json_path, field_key), ...])
PERTURB_SOURCES = {
    "numeration": [
        ("true", "data/perturbed_data/true_dataset/jsons/numeration_true.json", "numeration_true"),
        ("true", "data/perturbed_data/true_dataset/jsons/numeration_true.json", "numeration_false"),
        ("false", "data/perturbed_data/false_dataset/jsons/numeration_false.json", "numeration_true"),
        ("false", "data/perturbed_data/false_dataset/jsons/numeration_false.json", "numeration_false"),
    ],
    "approximation": [
        ("true", "data/perturbed_data/true_dataset/jsons/approximation_true.json", "approximation_true"),
        ("true", "data/perturbed_data/true_dataset/jsons/approximation_true.json", "approximation_false"),
        ("false", "data/perturbed_data/false_dataset/jsons/approximation_false.json", "approximation_true"),
        ("false", "data/perturbed_data/false_dataset/jsons/approximation_false.json", "approximation_false"),
    ],
    "range": [
        ("true", "data/perturbed_data/true_dataset/jsons/range_true.json", "range_true"),
        ("true", "data/perturbed_data/true_dataset/jsons/range_true.json", "range_false"),
        ("false", "data/perturbed_data/false_dataset/jsons/range_false.json", "range_true"),
        ("false", "data/perturbed_data/false_dataset/jsons/range_false.json", "range_false"),
    ],
    "masking": [
        ("true", "data/perturbed_data/true_dataset/jsons/obfuscation_true.json", "masking_false"),
    ],
    "randrepl": [
        ("true", "data/perturbed_data/true_dataset/jsons/obfuscation_true.json", "random_replacement_false"),
        ("false", "data/perturbed_data/false_dataset/jsons/obfuscation_false.json", "random_replacement_false"),
    ],
    "negnum": [
        ("true", "data/perturbed_data/true_dataset/jsons/negative_true.json", "negative_number_false"),
        ("false", "data/perturbed_data/false_dataset/jsons/negative_false.json", "negative_number_false"),
    ],
}

# obfuscation_{true,false}.json also carry "value_removal_false" -- a 3rd
# obfuscation sub-type not among the task brief's six perturbation types.
SKIPPED_TYPES = ["value_removal (obfuscation sub-type not in the task brief's six types)"]

FIELD_TO_LABEL = {"true": "Yes", "false": "No"}


def _label_str(origin: str) -> str:
    return FIELD_TO_LABEL[origin]


def convert_originals(src_dir: Path, out_dir: Path) -> int:
    out_path = out_dir / "numpert_original.jsonl"
    n = 0
    with open(out_path, "w") as fout:
        for origin, rel_path in TAGGED_PATHS.items():
            data = json.load(open(src_dir / rel_path))
            for item in data:
                fout.write(json.dumps({
                    "premise": item["doc"],
                    "hypothesis": item["claim"],
                    "label_str": _label_str(origin),
                    "orig_id": f"{origin}_{item['claim_id']}",
                }) + "\n")
                n += 1
    print(f"wrote {n} items -> {out_path}")
    return n


def convert_perturb_type(src_dir: Path, out_dir: Path, perturb_type: str, sources) -> int:
    out_path = out_dir / f"numpert_{perturb_type}.jsonl"
    n = 0
    n_skipped_identical = 0
    with open(out_path, "w") as fout:
        for origin, rel_path, field_key in sources:
            path = src_dir / rel_path
            if not path.exists():
                print(f"  [{perturb_type}] SKIPPING missing source file: {path}")
                continue
            data = json.load(open(path))
            orig_label_str = _label_str(origin)
            # field_key ends in "_true" or "_false": that suffix is the
            # RESULTING (post-perturbation) label, per the source repo's own
            # naming convention (verified against main_perturb.py / spot
            # checks in the module docstring above).
            result_origin = "true" if field_key.endswith("_true") else "false"
            result_label_str = _label_str(result_origin)
            for item in data:
                if field_key not in item:
                    continue
                hypothesis = item[field_key]
                if hypothesis == item.get("claim"):
                    # main_perturb.py's own remove_unperturbed_claims fail-safe
                    # already filters these out upstream in most cases; this
                    # is a defensive re-check.
                    n_skipped_identical += 1
                    continue
                fout.write(json.dumps({
                    "premise": item["doc"],
                    "hypothesis": hypothesis,
                    "label_str": result_label_str,
                    "perturb_type": perturb_type,
                    "orig_id": f"{origin}_{item['claim_id']}",
                    "orig_label_str": orig_label_str,
                    "flip": result_label_str != orig_label_str,
                }) + "\n")
                n += 1
    print(f"wrote {n} items -> {out_path} (skipped {n_skipped_identical} identical-to-original)")
    return n


def convert(src_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    convert_originals(src_dir, out_dir)
    for perturb_type, sources in PERTURB_SOURCES.items():
        convert_perturb_type(src_dir, out_dir, perturb_type, sources)
    print(f"SKIPPED_TYPES (materialized upstream but not converted): {SKIPPED_TYPES}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src_dir", type=Path, help="Path to cloned adversarial_attack_numerical_claims repo root")
    ap.add_argument("out_dir", type=Path, help="Output directory (e.g. data/)")
    args = ap.parse_args()
    convert(args.src_dir, args.out_dir)


if __name__ == "__main__":
    main()
