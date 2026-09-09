# gt-tl-repro

Reproducibility package for **"Reproducing, Stress-Testing, and Generalizing the
GPT-2 Greater-Than Circuit"** (BlackboxNLP 2026, special/reproducibility track).
This workspace contains the TransformerLens-based reproduction and
generalization study of the GPT-2 "greater-than" circuit from Hanna, Liu &
Variengien (2023), *"How does GPT-2 compute greater-than?"*, NeurIPS 2023,
[arXiv:2305.00586](https://arxiv.org/abs/2305.00586). It is the companion code to the BlackboxNLP 2026 paper of the same study.

Four experiment families, corresponding to the paper's four research
questions:

- **RQ1 (Reproduction):** does the original circuit + faithfulness numbers
  replicate, both in the original code and independently in TransformerLens?
- **RQ2 (Rigor):** template robustness, bootstrap CIs, a random-circuit null,
  method agreement, and the model's own less-than failure mode.
- **RQ3 (Cross-model):** does an analogous circuit exist across GPT-2/Pythia
  scale and family, and does it localize at matched relative depth?
- **RQ4 (Numerical NLI):** when the same comparison is posed as natural
  language entailment, is the same circuitry recruited?

---

## 1. Environment

Python **3.11** via [uv](https://docs.astral.sh/uv/), from this directory:

```bash
cd gt-tl-repro
uv sync
```

### Critical dependency pins (do not "helpfully" upgrade these)

The compute cluster this was developed on runs **glibc 2.27** (Ubuntu 18.04).
`manylinux2014` (glibc 2.17) wheels work; `manylinux_2_28` wheels do not, and
several packages will otherwise fall back to a source build that fails outright.
`pyproject.toml` pins:

| Package | Pin | Reason |
|---|---|---|
| `wandb` | `>=0.16,<0.18` | `>=0.18` ships `wandb-core`, which needs a `go build` toolchain that is unavailable/fails on this cluster. |
| `pyarrow` | `>=14,<17` | `17+` only publishes `manylinux_2_28` wheels; `<17` still ships `manylinux2014`. |
| `datasets` | `>=2.14,<3` | Pure-Python; keeping it `<3` avoids pulling in the `pyarrow>=17` requirement that `datasets>=3` introduces transitively. |
| `transformers` | `>=4.43,<5` | TransformerLens 2.15 reads `transformers.TRANSFORMERS_CACHE`, which is removed in `transformers` v5. |
| `torch` | `>=2.1,<2.5`, `cu121` index | Matches the cluster's `uenv cuda-12.1.0`; see `[tool.uv.sources]` / `[[tool.uv.index]]` in `pyproject.toml` for the explicit PyTorch CUDA 12.1 wheel index. |

`[tool.uv] no-build-package = ["wandb", "pyarrow"]` forces uv to fail fast on a
missing wheel rather than silently attempting (and failing) a source build.

If you are **not** on a glibc-2.27 system, these pins are still safe (just more
conservative than necessary) — leave them as is unless `uv sync` reports a
genuine resolution conflict.

### Local editable dependency: `eap` (EAP-IG)

All experiments except the (separately maintained) original `rust_circuit`
reproduction depend on [`hannamw/EAP-IG`](https://github.com/hannamw/EAP-IG),
declared as a local editable path dependency:

```toml
[tool.uv.sources]
eap = { path = "../EAP-IG", editable = true }
```

This means **`EAP-IG` must be cloned as a sibling directory of `gt-tl-repro/`**:

```
parent/
├── greater-than-repro/   <- this repository (uv sync is run from here)
└── EAP-IG/               <- git clone https://github.com/hannamw/EAP-IG.git
```

```bash
mkdir parent && cd parent
git clone git@github.com:iai-group/greater-than-repro.git
git clone https://github.com/hannamw/EAP-IG.git
cd greater-than-repro
uv sync
```

### GPU

A single A100/H100-class GPU (~80GB) is required for the largest
attribution jobs — specifically Qwen2.5-7B EAP-IG discovery, which needed a
batch-size search ladder down to **batch size 2** to avoid OOM (grouped-query
attention is ungrouped for edge scoring, which is memory-hungry). Everything
else (GPT-2, Pythia up to 2.8B) comfortably fits on a single 40–80GB GPU.
`experiments/make_figures.py` is the only script that is CPU-only.

---

## 2. Offline-cluster pattern

The cluster this was run on has **no internet access on GPU compute nodes** —
only the login node can reach the network. If your cluster is fully online,
none of this applies and you can skip straight to §4. If you are on a similar
offline-compute-node cluster, the pattern generalizes as:

1. **On the login node**, sync the environment and pre-download every model
   checkpoint into a shared `HF_HOME`:

   ```bash
   cd gt-tl-repro
   UV_LINK_MODE=copy uv sync
   HF_HOME=/path/to/shared/hf-cache uv run python -c "
   from transformers import AutoTokenizer, AutoModelForCausalLM
   for name in ['gpt2', 'gpt2-medium',
                'EleutherAI/pythia-160m', 'EleutherAI/pythia-410m',
                'EleutherAI/pythia-1.4b', 'EleutherAI/pythia-2.8b',
                'Qwen/Qwen2.5-7B']:
       AutoTokenizer.from_pretrained(name); AutoModelForCausalLM.from_pretrained(name)
       print(name, 'cached')
   "
   ```

   Models actually used in the paper: `gpt2`, `gpt2-medium`,
   `EleutherAI/pythia-{160m,410m,1.4b,2.8b}`, `Qwen/Qwen2.5-7B`. Qwen2.5-3B and
   1.5B were also cached as fallbacks in case the 7B model didn't fit, but are
   **unused** in any reported result. **Llama-3-8B was excluded**: its weights
   are gated on HuggingFace and were not available under our access.

2. **On the GPU compute node**, forbid any implicit network access so jobs
   fail loudly (rather than hanging) if something wasn't pre-cached:

   ```bash
   export UV_NO_SYNC=1
   export HF_HOME=/path/to/shared/hf-cache
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   ```

   All `slurm*.sh` scripts in this repo already set these four variables.

3. **SLURM scripts are templates for one specific cluster.** Every
   Each `slurm*.sh` requires `REPO_DIR` to point at this repository and
   honours `HF_HOME`; set the partition and any node constraints your
   cluster needs. **Adjust these before submitting.**
   everything else (env vars, the actual run command) should work unmodified.
   If you don't use SLURM at all, run the `uv run python ...` line from inside
   each script directly on a GPU machine, with the same env vars exported.

---

## 3. Per-experiment commands

All commands are run from `gt-tl-repro/`. Each `uv run python ...` line has a
matching `sbatch <script>` line for the offline-cluster/SLURM case; edit the
nodelist first (§2, step 3).

### RQ1 — reproduction (`reproduce_greater_than.py`)

```bash
uv run python reproduce_greater_than.py --model gpt2-small
# or: sbatch slurm.sh
```

Writes `results/summary_gpt2-small.json` and
`results/graph_gpt2-small_{EAP-IG-inputs,EAP,clean-corrupted}.json`.
Reference numbers (from the EAP-IG release notebook): baseline **0.8143**;
EAP-IG circuit **0.7449** perf / **91.5%** recovery / **29 nodes** / **150
edges**.

### RQ2 — rigor battery (`experiments/rq2_rigor.py`)

```bash
uv run python experiments/rq2_rigor.py
# or: sbatch experiments/slurm_rq2.sh
```

Writes `results/rq2_rigor.json` (per-template baselines with bootstrap CIs,
method agreement, random-circuit null, less-than failure mode).

### RQ3 — cross-model (`experiments/rq3_crossmodel.py`)

Two-stage: behavioral gate first, then discovery only on gate-passing models.

```bash
uv run python experiments/rq3_crossmodel.py --gate       # -> results/rq3_gate.json
# or: sbatch experiments/slurm_rq3.sh
uv run python experiments/rq3_crossmodel.py --discover   # -> results/rq3_circuits.json
# or: sbatch experiments/slurm_rq3_discover.sh
```

> **Caveat:** `--discover` writes `rq3_circuits.json` only once, at the end of
> the six-model loop (~1 h on an A100). A crash mid-run loses the finished
> models and requires a full re-run. (`rq4_nli.py --discover` writes
> incrementally; rq3 does not — a known gap.)

### RQ4 — numerical NLI (`experiments/rq4_nli.py`)

Same two-stage pattern.

```bash
uv run python experiments/rq4_nli.py --gate       # -> results/rq4_gate.json
# or: sbatch experiments/slurm_rq4.sh
uv run python experiments/rq4_nli.py --discover   # -> results/rq4_circuits.json
# or: sbatch experiments/slurm_rq4_discover.sh
```

### Camera-ready additions (BlackboxNLP 2026 reviews)

Five experiments added for the camera-ready, each answering a specific
reviewer point. All write JSON under `results/`.

| Script | Answers | Command | Output |
|---|---|---|---|
| `experiments/rq3_normalized.py` | Reviewers yDuj (W1) and CxwV (W2): the fixed 200-edge budget is unfair across scale | `uv run python experiments/rq3_normalized.py` | `results/rq3_normalized.json` |
| `experiments/rq3_crossmodel.py` | yDuj: "why stop at 2.8B?" (now reaches pythia-12b) | `uv run python experiments/rq3_crossmodel.py --gate --discover` | `results/rq3_gate.json`, `results/rq3_circuits.json` |
| `experiments/rq4_format_control.py` | CxwV (W1): RQ4's null may be format inability, not an unused circuit | `uv run python experiments/rq4_format_control.py` | `results/rq4_format_control.json` |
| `experiments/rq3_iou_null.py` | CxwV (W3): bucketed IoU lacks a null | `uv run python experiments/rq3_iou_null.py` | `results/rq3_iou_null.json` |
| `experiments/rq4_edge_ablation.py` | yDuj: why ablate MLPs rather than the edges into them | `uv run python experiments/rq4_edge_ablation.py` | `results/rq4_edge_ablation.json` |

`rq3_iou_null.py` is CPU-only and runs in seconds. The rest need a GPU; the
matching `experiments/slurm_*.sh` scripts submit them.

**Attribution graphs for the larger models are not committed.** Only
gpt2-small's and pythia-160m's graphs ship here (2.3MB each). The rest range
from 17MB to 225MB, exceeding GitHub's per-file limits, and they are
intermediates rather than results: every number in the paper comes from the
committed `rq3_*.json` and `rq4_*.json` files. Regenerate any of them with

```bash
uv run python experiments/rq3_crossmodel.py --discover --only <model>
uv run python experiments/rq4_nli.py --discover        # Qwen entailment circuit
```

which is also what `rq3_normalized.py` and `rq4_qwen_normalized.py` need in
order to sweep budgets for those models.

**Normalized budgets need no re-attribution.** `Graph.to_json` persists every
edge's score and `Graph.from_json` restores it, and `apply_topn` mutates only
`in_graph`/`nodes_in_graph`. So a cached graph in `results/graph_*.json` can be
re-thresholded at any budget with evaluation passes alone. `rq3_normalized.py`
relies on this; it is why the fraction sweep is cheap.

**Half precision above 2.8B.** `gtlib.runner.HALF_PRECISION_MODELS` loads
pythia-6.9b, pythia-12b, and Qwen2.5-7B in `bfloat16`. pythia-12b in fp32
would be ~48GB of weights before EAP-IG's activation caches, which does not
fit an 80GB A100. bfloat16 rather than float16 because attribution
backpropagates through the whole graph and fp16 gradients underflow on the
small probability differences this metric produces.

**SLURM scripts must export PATH.** Batch jobs do not source the login
profile, so `uv` (installed under `~/.local/bin`) is not on `PATH` and the job
dies immediately with exit code 127 and an empty stderr. Every
`experiments/slurm_*.sh` added here sets
`export PATH="$HOME/.local/bin:$PATH"` before its first `uv` call. On this
cluster `sbatch` also lives only on the head node, so submit with
`REPO_DIR=/path/to/greater-than-repro sbatch experiments/<script>.sh`, from
the head node if your cluster only provides `sbatch` there.

### Figures and tables (`experiments/make_figures.py`)

CPU-only; reads **only** the committed `results/*.json` files (no model
loading, no GPU):

```bash
uv run python experiments/make_figures.py
```

Writes `../paper/figures/*.pdf`, `../paper/tables/*.tex`, and
`../paper/figures/MANIFEST.txt`. The paper tree is not part of this
repository, so either create `../paper/{figures,tables}/` or edit `PAPER` at
the top of `experiments/make_figures.py` to point somewhere else (a figure/table → source-JSON-key manifest,
i.e. a second, machine-readable copy of the traceability table in §5 below).

### Tests

```bash
uv run python -m pytest tests/ -q
```

All tests are expected to pass. As of this writing (last full run recorded in
this README) the suite has **33 tests, all passing** (`33 passed` in ~2
minutes) — re-run the command above to confirm on your machine/environment;
if you don't have GPU access, the tests still pass (they exercise `gtlib/`
pure-Python logic and CPU-side tokenization, not model forward passes).

---

## 4. Seed policy

Every experiment script accepts `--seed` (default **42**). Precisely: dataset
generation is seeded from `--seed` (a local `random.Random(seed)` per
generator); `torch.manual_seed(0)` is hardcoded before discovery/evaluation
(`gtlib/runner.py`, `rq3_crossmodel.py`, `rq4_nli.py`); `bootstrap_ci` seeds
NumPy locally (rq2/rq3 pass `--seed`; rq4's gate uses its default `seed=0`).
Everything is deterministic under the defaults, and all reported numbers use
them. **Discovery is single-seed**: we do not
average circuit discovery over multiple random seeds. The confidence
intervals reported throughout (bootstrap, `gtlib/stats.py:bootstrap_ci`) are
**over evaluation items** (i.e., resampling the probe dataset), not over
discovery-seed variation — this is stated explicitly in the paper's
Limitations section, and it means a different seed could in principle surface
a structurally different sparse circuit even though the item-level CIs would
look identical. We do not currently have evidence against this beyond the
qualitative stability shown by the RQ2 template-robustness results.

---

## 5. Results → paper traceability

Every number in the paper originates from one of the JSON files in
`results/`. See `paper/figures/MANIFEST.txt` for the auto-generated,
per-figure/table version of this table (written by `make_figures.py`).

| Paper item | Source JSON(s) | Producing script / job |
|---|---|---|
| Table 1 (reproduction numbers) | `results/summary_gpt2-small.json`, `results/rq1_overlap.json` | `reproduce_greater_than.py` (`slurm.sh`) |
| Figure 1 + RQ2 rigor numbers (template forest plot, method agreement, random-circuit null, less-than failure) | `results/rq2_rigor.json` | `experiments/rq2_rigor.py` (`slurm_rq2.sh`) |
| Table 2 + Figure 2 (cross-model localization, budget sweep) | `results/rq3_gate.json`, `results/rq3_circuits.json` | `experiments/rq3_crossmodel.py --gate` / `--discover` (`slurm_rq3.sh`, `slurm_rq3_discover.sh`) |
| Table 3 + Figure 3 (NLI behavioral gate, double dissociation) | `results/rq4_gate.json` | `experiments/rq4_nli.py --gate` (`slurm_rq4.sh`) |
| Figure 4 + RQ4 circuit numbers (overlap, cross-task faithfulness, necessity transfer) | `results/rq4_circuits.json` | `experiments/rq4_nli.py --discover` (`slurm_rq4_discover.sh`) |
| All figures/tables (aggregation step) | reads all of the above | `experiments/make_figures.py` (CPU-only) → `../paper/figures/*.pdf`, `../paper/tables/*.tex`, `../paper/figures/MANIFEST.txt` |

---

## 6. Known caveats and honesty notes

These are stated in the paper's Limitations/Discussion sections; they are
repeated here for anyone re-running the code so the "gotchas" are visible
before you hit them rather than after.

- **The random-circuit null is degenerate.** `gtlib/stats.py`'s
  `random_circuit_scores` samples random edge sets of the same size as the
  discovered circuit; in practice, unconstrained random edge sampling at these
  sizes collapses every sample to the (functionally) empty graph. This
  licenses a **percentile statement only** ("the observed circuit outperforms
  100% of random-edge-count-matched samples"), not a Cohen's-d effect size —
  we report it as such. A **connectivity-preserving sampler is a TODO** before
  a defensible effect-size claim can be made; do not over-interpret the
  current null as a well-calibrated baseline.
- **pythia-160m's circuit faithfulness exceeds 1.0** (1.05–1.24 across the
  budget sweep in `results/rq3_circuits.json`). This is a numerical artifact
  of its very low, noisy full-model baseline (prob-diff 0.431, barely above
  the 0.3 gate) rather than a genuine positive result — the circuit is not
  "better than the full model," the ratio is just unstable at a small
  denominator.
- **Two upstream bugs found in the original `rust_circuit` release**
  (upstream `hannamw/gpt2-greater-than`, not part of this repository — see below):
  (1) the reverse-patching cell in `big_ds_experiments.py` reuses a sampler
  group sized for N=200 examples against N=10,000 tensors, causing a
  shape-mismatch crash at full scale; (2) once patched naively, accumulated
  circuit references cause an out-of-memory failure. Neither is exercised by
  the release's own verification notebook, which only runs at small scale.
- **BOS-token pitfall.** TransformerLens prepends a BOS token by default for
  Pythia and Qwen tokenizers but not for GPT-2. A single-token validity check
  that doesn't account for this will silently pass a BOS-corrupted token as
  "single-token," and the resulting metric collapses toward an all-BOS
  artifact that still looks like a plausible number. `gtlib/datasets.py`,
  `gtlib/nli.py`, and `gtlib/runner.py` all handle this explicitly (search for
  `add_special_tokens=False` and the `BOS` comments in those files) — if you
  extend this code to a new model family, re-check its tokenizer's BOS
  behavior first.
- **Qwen2.5-7B cannot be posed the year-completion task at all.** Its
  tokenizer splits multi-digit numbers into per-digit tokens rather than the
  two-digit chunks the RQ1–RQ3 task assumes, so the prompt and prob-diff
  metric are not well-defined for it. Qwen is therefore excluded from the RQ3
  year-task gate/discovery but **is** used for RQ4 (the NLI probe's Yes/No
  scoring is tokenization-robust across all families we studied, Qwen
  included).
- **The original `rust_circuit`-based reproduction is not in this
  repository.** It is a clone of upstream
  [`hannamw/gpt2-greater-than`](https://github.com/hannamw/gpt2-greater-than)
  with the two bug fixes above applied, kept separate from this repository
  and referenced only for provenance of the RQ1 "original stack" numbers
  reported in the paper (baseline prob-diff 0.8189).

---

## 7. Directory layout

```
greater-than-repro/
├── pyproject.toml, uv.lock          # environment (see §1)
├── reproduce_greater_than.py        # RQ1 runner
├── slurm.sh                         # RQ1 SLURM template
├── greater_than_data.csv            # original EAP-IG year-task dataset
├── gtlib/                           # shared library
│   ├── datasets.py                  # year-task template/range generators (RQ2/RQ3)
│   ├── nli.py                       # synthetic comparison-NLI probe + metric (RQ4)
│   ├── stats.py                     # bootstrap CIs, random-circuit null, effect sizes
│   ├── ground_truth.py              # Hanna-circuit node reference + overlap metrics
│   └── runner.py                    # model loading, EAP-IG discovery, evaluation
├── tests/                           # pytest suite for gtlib/
├── experiments/
│   ├── rq2_rigor.py, slurm_rq2.sh
│   ├── rq3_crossmodel.py, slurm_rq3.sh, slurm_rq3_discover.sh
│   ├── rq4_nli.py, slurm_rq4.sh, slurm_rq4_discover.sh
│   └── make_figures.py              # results/*.json -> ../paper/{figures,tables}
└── results/                         # committed JSON summaries + attribution graphs
```

### Naturalistic additions (stretch)
| Paper element | Results file | Producer |
|---|---|---|
| §6 Naturalistic checks (EQUATE) | `results/rq4_equate.json` | `experiments/rq4_equate.py` (`slurm_rq4_equate.sh`) |
| §6 Naturalistic checks (NumPert incl. positional faithfulness) | `results/rq4_numpert.json` | `experiments/rq4_numpert.py` (`slurm_rq4_numpert.sh`) |

NumPert data (Aarnes & Setty 2025) converted in `data/convert_numpert.py`; used
and redistributed with the dataset authors' permission (no upstream license file).
NOTE for blind-review packaging: `data/README.md` names the dataset authors —
redact if bundling this repo as anonymous supplementary material.

---

## Scope of this release

This repository contains the code for the reproducibility study only.
Experiment scripts belonging to separate, unpublished follow-up work are not
included, and neither are the converted EQUATE / NumPert / AVeriTeC files:
`data/convert_equate.py` and `data/convert_numpert.py` regenerate those from
the upstream releases, which is preferable to redistributing them here.
