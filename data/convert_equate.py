"""One-off conversion script: EQUATE (Ravichander et al. 2019) RTE-Quant and
AWPNLI subsets -> binary entailed(Yes)/not-entailed(No) JSONL for RQ4-stretch.

Source: github.com/AbhilashaRavichander/EQUATE (MIT-licensed; see
EQUATE/LICENSE in the upstream repo). Converted files are small (166 + 722
short text pairs) and are committed to this repo under data/equate_*.jsonl.

Usage (one-off; not run by any experiment script):
    python data/convert_equate.py /path/to/cloned/EQUATE data/

Source format (SNLI-style, one JSON object per line):
    RTE_Quant.jsonl: {"sentence1": premise, "sentence2": hypothesis,
        "gold_label": "entailment" | "neutral", ...}  (166 pairs, 2-way)
    AWPNLI.jsonl:    {"sentence1": premise, "sentence2": hypothesis,
        "gold_label": "entailment" | "contradiction", ...}  (722 pairs)

Label mapping (both subsets, per task brief): entailment -> "Yes";
neutral / contradiction -> "No".
"""
import argparse
import json
from pathlib import Path

LABEL_MAP = {
    "entailment": "Yes",
    "neutral": "No",
    "contradiction": "No",
}

SUBSETS = {
    "rtequant": "RTE_Quant.jsonl",
    "awpnli": "AWPNLI.jsonl",
}


def convert(src_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for out_name, src_name in SUBSETS.items():
        src_path = src_dir / "ProcessedDatasets" / src_name
        out_path = out_dir / f"equate_{out_name}.jsonl"
        n = 0
        with open(src_path) as fin, open(out_path, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                gold = d["gold_label"]
                if gold not in LABEL_MAP:
                    continue
                fout.write(json.dumps({
                    "premise": d["sentence1"],
                    "hypothesis": d["sentence2"],
                    "label_str": LABEL_MAP[gold],
                    "orig_gold_label": gold,
                }) + "\n")
                n += 1
        print(f"wrote {n} items -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src_dir", type=Path, help="Path to cloned EQUATE repo root")
    ap.add_argument("out_dir", type=Path, help="Output directory (e.g. data/)")
    args = ap.parse_args()
    convert(args.src_dir, args.out_dir)


if __name__ == "__main__":
    main()
