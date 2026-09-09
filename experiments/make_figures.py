"""
Generate paper figures and tables for the BlackboxNLP 2026 greater-than
reproducibility paper, from committed results JSONs ONLY.

No model loading, no GPU. Reads gt-tl-repro/results/*.json, writes:
  - ../paper/figures/*.pdf
  - ../paper/tables/*.tex
  - ../paper/figures/MANIFEST.txt  (figure -> source JSON key manifest)

Run with:  .venv/bin/python experiments/make_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["axes.grid"] = False
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)  # gt-tl-repro/
RESULTS = os.path.join(REPO, "results")
PAPER = os.path.join(os.path.dirname(REPO), "paper")
FIGDIR = os.path.join(PAPER, "figures")
TABDIR = os.path.join(PAPER, "tables")

os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)

MANIFEST_LINES = []


def manifest(fig_name, source_notes):
    MANIFEST_LINES.append(f"{fig_name}:")
    for note in source_notes:
        MANIFEST_LINES.append(f"  - {note}")
    MANIFEST_LINES.append("")


def load(name):
    path = os.path.join(RESULTS, name)
    with open(path) as f:
        return json.load(f)


def need(d, *keys, ctx=""):
    """Fail loudly if any key is missing from d (top-level check)."""
    for k in keys:
        if k not in d:
            raise KeyError(f"Expected key '{k}' missing in {ctx or d}")


# ---------------------------------------------------------------------------
# Load & validate all sources
# ---------------------------------------------------------------------------

summary_gpt2 = load("summary_gpt2-small.json")
need(summary_gpt2, "baseline", "circuits", ctx="summary_gpt2-small.json")
need(summary_gpt2["circuits"], "EAP-IG-inputs", "EAP", "clean-corrupted",
     ctx="summary_gpt2-small.json:circuits")

rq1_overlap = load("rq1_overlap.json")
need(rq1_overlap, "discovered", "core", "full", ctx="rq1_overlap.json")

rq2 = load("rq2_rigor.json")
need(rq2, "templates", "methods", "random_null", "less_than", ctx="rq2_rigor.json")
for t, d in rq2["templates"].items():
    need(d, "baseline", "circuit_perf", "overlap_core", ctx=f"rq2.templates.{t}")
    need(d["baseline"], "mean", "lo", "hi", ctx=f"rq2.templates.{t}.baseline")

rq3_gate = load("rq3_gate.json")
rq3_circuits = load("rq3_circuits.json")
need(rq3_circuits, "models", "pairwise_bucketed_overlap", ctx="rq3_circuits.json")

rq4_gate = load("rq4_gate.json")
rq4_circuits = load("rq4_circuits.json")
need(rq4_circuits, "pythia28b_necessity", "qwen_nli", ctx="rq4_circuits.json")
need(rq4_circuits["pythia28b_necessity"], "baseline", "late_mlp_ablated",
     "control_ablated", ctx="rq4_circuits.json:pythia28b_necessity")
need(rq4_circuits["qwen_nli"], "baseline", "sweep",
     "bucketed_overlap_vs_year_circuits", "layer_fractions", "n_layers",
     ctx="rq4_circuits.json:qwen_nli")

# Hardcoded reference values (not from committed JSONs; cite source in comments).
# Source: docs/superpowers/plans/2026-07-01-greater-than-reproducibility-plan.md:19
#   "rust_circuit baseline prob-diff 0.8189 (banked); EAP-IG notebook: baseline
#    ~0.814, top-200 EAP-IG circuit ~0.745 (~91% recovery), 29 nodes / 150 edges."
# Source: docs/superpowers/specs/2026-06-28-greater-than-circuit-reproducibility-design.md:27
#   "rust_circuit part DONE (prob-diff 0.8189; recovery numbers
#    0.7261/0.8867/0.9855; found+fixed two released-code bugs: unbatched
#    reverse-patching cell, OOM)."
PAPER_REPORTED = {"baseline": 0.817, "circuit_perf": 0.745, "nodes": 29, "edges": 150}
RUST_CIRCUIT = {"baseline": 0.8189, "recovery_tiers": [0.7261, 0.8867, 0.9855]}

# ---------------------------------------------------------------------------
# Okabe-Ito palette, FIXED per-model assignment (never cycled)
# ---------------------------------------------------------------------------
COLOR = {
    "gpt2-small": "#0072B2",
    "gpt2-medium": "#56B4E9",
    "pythia-160m": "#E69F00",
    "pythia-410m": "#D55E00",
    "pythia-1.4b": "#CC79A7",
    "pythia-2.8b": "#009E73",
    "pythia-6.9b": "#7F3C8D",
    "pythia-12b": "#3B7302",
    "Qwen/Qwen2.5-7B": "#000000",
    "Qwen2.5-7B (NLI)": "#000000",
}
MARKER = {
    "gpt2-small": "o",
    "gpt2-medium": "s",
    "pythia-160m": "^",
    "pythia-410m": "v",
    "pythia-1.4b": "D",
    "pythia-2.8b": "P",
    "pythia-6.9b": "*",
    "pythia-12b": "h",
    "Qwen/Qwen2.5-7B": "X",
    "Qwen2.5-7B (NLI)": "X",
}
LINESTYLE = {
    "gpt2-small": "-",
    "gpt2-medium": "--",
    "pythia-160m": "-",
    "pythia-410m": "--",
    "pythia-1.4b": "-.",
    "pythia-2.8b": ":",
    "pythia-6.9b": "--",
    "pythia-12b": "-.",
    "Qwen/Qwen2.5-7B": "-",
    "Qwen2.5-7B (NLI)": "-",
}
GRID_KW = dict(color="0.85", linewidth=0.6, zorder=0)

MODEL_ORDER = [
    "gpt2-small", "gpt2-medium", "pythia-160m", "pythia-410m",
    "pythia-1.4b", "pythia-2.8b",
]

YEAR_MODELS_6 = MODEL_ORDER  # the 6 year-task models with circuits

# Camera-ready ordering, extended to 12B for reviewer yDuj's scale question.
MODEL_ORDER_EXT = MODEL_ORDER + ["pythia-6.9b", "pythia-12b"]


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


# ===========================================================================
# TABLE 1: reproduction.tex
# ===========================================================================


def build_table_reproduction():
    ig = summary_gpt2["circuits"]["EAP-IG-inputs"]
    eap = summary_gpt2["circuits"]["EAP"]
    cc = summary_gpt2["circuits"]["clean-corrupted"]
    baseline = summary_gpt2["baseline"]

    lines = []
    lines.append(r"% Table: RQ1 reproduction of the greater-than circuit on GPT-2-small.")
    lines.append(r"% Compares paper-reported numbers, our rust_circuit reproduction,")
    lines.append(r"% and our TransformerLens/EAP-IG re-implementation.")
    lines.append(r"% Source: results/summary_gpt2-small.json (our TL numbers);")
    lines.append(r"%         docs/superpowers/plans/2026-07-01-...-plan.md:19 and")
    lines.append(r"%         docs/superpowers/specs/2026-06-28-...-design.md:27 (paper/rust_circuit).")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Reproduction of the GPT-2-small greater-than circuit: "
                 r"paper-reported values, our rust\_circuit reproduction, and our "
                 r"TransformerLens (EAP-IG) re-implementation. Recovery = circuit "
                 r"perf.\ / baseline.}")
    lines.append(r"\label{tab:reproduction}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Source & Baseline & Circuit perf. & Recovery & Nodes / Edges \\")
    lines.append(r"\midrule")
    lines.append(
        rf"Paper-reported & {fmt(PAPER_REPORTED['baseline'])} & "
        rf"{fmt(PAPER_REPORTED['circuit_perf'])} & "
        rf"{fmt(PAPER_REPORTED['circuit_perf']/PAPER_REPORTED['baseline'])} & "
        rf"{PAPER_REPORTED['nodes']} / {PAPER_REPORTED['edges']} \\"
    )
    lines.append(
        rf"rust\_circuit repro. & {fmt(RUST_CIRCUIT['baseline'])} & "
        r"--- & " +
        "/".join(fmt(v) for v in RUST_CIRCUIT["recovery_tiers"]) +
        r" & --- \\"
    )
    lines.append(
        rf"Our TL / EAP-IG & {fmt(baseline)} & {fmt(ig['perf'])} & "
        rf"{fmt(ig['recovery'])} & {ig['nodes']} / {ig['edges']} \\"
    )
    lines.append(r"\midrule")
    lines.append(
        rf"\quad EAP (variant) & {fmt(baseline)} & {fmt(eap['perf'])} & "
        rf"{fmt(eap['recovery'])} & {eap['nodes']} / {eap['edges']} \\"
    )
    lines.append(
        rf"\quad clean-corrupted (variant) & {fmt(baseline)} & {fmt(cc['perf'])} & "
        rf"{fmt(cc['recovery'])} & {cc['nodes']} / {cc['edges']} \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = os.path.join(TABDIR, "reproduction.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    manifest("tables/reproduction.tex", [
        "PAPER_REPORTED (hardcoded, cited from plan/spec docs)",
        "RUST_CIRCUIT (hardcoded, cited from plan/spec docs)",
        "results/summary_gpt2-small.json: baseline, circuits.{EAP-IG-inputs,EAP,clean-corrupted}",
    ])
    return out


# ===========================================================================
# TABLE 2: crossmodel.tex
# ===========================================================================


def build_table_crossmodel():
    """Cross-model table, rebuilt around normalized edge budgets.

    Reviewers yDuj and CxwV both objected that the submitted table's fixed
    200-edge budget is unfair across the parameter range, so the table now
    reports each model's total edge count, faithfulness at the absolute 200
    edges, and faithfulness at a budget matched as a fraction of that model's
    own graph. CxwV also asked for an IoU null and for pythia-160m's
    above-1.0 faithfulness to be flagged as an artifact; both are here.
    """
    models = rq3_circuits["models"]
    pairwise = rq3_circuits["pairwise_bucketed_overlap"]

    # Loaded lazily: these two runs may postdate a partial figure rebuild.
    def _maybe(name):
        path = os.path.join(RESULTS, name)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    normalized = _maybe("rq3_normalized.json")
    iou_null = _maybe("rq3_iou_null.json")

    order = [m for m in MODEL_ORDER_EXT if m in models
             and "layer_fractions" in models[m]]

    def late_mlp_count(model_name):
        lf = models[model_name]["layer_fractions"]
        return sum(1 for node, frac in lf.items()
                   if node.startswith("m") and frac >= 0.5)

    def iou_vs_gpt2(model_name):
        if model_name == "gpt2-small":
            return 1.0
        for key in (f"gpt2-small__{model_name}", f"{model_name}__gpt2-small"):
            if key in pairwise:
                return pairwise[key]["iou"]
        raise KeyError(f"No pairwise IoU for gpt2-small vs {model_name}")

    def matched_cells(model_name):
        """Returns (total-edge string, matched budget string, faithfulness)."""
        if not normalized or model_name not in normalized["models"]:
            return "---", "---", "---"
        info = normalized["models"][model_name]
        if "fraction_sweep" not in info:
            return f"{info['total_edges']:,}", "---", "---"
        frac = str(normalized["matched_fraction"])
        budget = info["absolute_budgets"].get(frac)
        faith = info["fraction_sweep"].get(frac)
        return (f"{info['total_edges']:,}",
                f"{budget:,}" if budget else "---",
                fmt(faith) if faith is not None else "---")

    def null_cell(model_name):
        if not iou_null or model_name not in iou_null.get("models", {}):
            return "---"
        return fmt(iou_null["models"][model_name]["null_mean"])

    lines = []
    lines.append(r"% Table: RQ3 cross-model localization summary.")
    lines.append(r"% Source: results/rq3_gate.json (gate prob-diff), "
                 r"results/rq3_circuits.json (n_layers, sweep, layer_fractions, "
                 r"pairwise_bucketed_overlap), results/rq3_normalized.json "
                 r"(total_edges, matched-fraction budget and faithfulness), "
                 r"results/rq3_iou_null.json (late-layer IoU null).")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Cross-model year-task circuits. \emph{Edges} is the "
                 r"model's total edge count, which spans a 49$\times$ range and "
                 r"is what makes an absolute budget incomparable across models. "
                 r"\emph{Faith.@200} is faithfulness at the fixed 200-edge budget "
                 r"the submitted version reported; \emph{Faith.@match} is "
                 r"faithfulness at the budget matched to 0.62\% of each model's "
                 r"own graph, i.e.\ the fraction that 200 edges represents for "
                 r"gpt2-small. Bucketed IoU uses $K{=}11$ depth deciles held "
                 r"constant across all model depths; \emph{Null} is the mean IoU "
                 r"of size-matched random late-layer node sets (1{,}000 samples). "
                 r"$\dagger$ pythia-160m's above-1.0 faithfulness is an artifact "
                 r"of its low, noisy baseline (0.431), not evidence of better "
                 r"generalization.}")
    lines.append(r"\label{tab:crossmodel}")
    lines.append(r"\begin{tabular}{lrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Layers & Edges & Gate & Faith.@200 & Match $n$ & "
                 r"Faith.@match & Late-MLP $n$ & IoU (Null) \\")
    lines.append(r"\midrule")
    for m in order:
        md = models[m]
        gate = rq3_gate.get(m, {})
        prob_diff = gate.get("prob_diff")
        prob_diff_str = fmt(prob_diff) if prob_diff is not None else "---"
        f200 = fmt(md["sweep"]["200"]) if "200" in md["sweep"] else "---"
        edges_str, match_n, f_match = matched_cells(m)
        flag = r"$\dagger$" if m == "pythia-160m" else ""
        mname = m.replace("_", r"\_") + flag
        lines.append(
            rf"{mname} & {md['n_layers']} & {edges_str} & {prob_diff_str} & "
            rf"{f200} & {match_n} & {f_match} & {late_mlp_count(m)} & "
            rf"{fmt(iou_vs_gpt2(m))} ({null_cell(m)}) \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out = os.path.join(TABDIR, "crossmodel.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    manifest("tables/crossmodel.tex", [
        "results/rq3_gate.json: <model>.prob_diff",
        "results/rq3_circuits.json: models.<model>.{n_layers, sweep.200, "
        "layer_fractions (late-MLP count: node startswith 'm' and frac>=0.5)}",
        "results/rq3_circuits.json: pairwise_bucketed_overlap.gpt2-small__<model>.iou",
        "results/rq3_normalized.json: models.<model>.{total_edges, "
        "absolute_budgets[matched], fraction_sweep[matched]}",
        "results/rq3_iou_null.json: models.<model>.null_mean",
    ])
    return out


# ===========================================================================
# TABLE 3: rq4gate.tex
# ===========================================================================


def build_table_rq4gate():
    lines = []
    lines.append(r"% Table: RQ4 NLI gate results per model.")
    lines.append(r"% Source: results/rq4_gate.json")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{NLI gate: zero-shot and two-shot accuracy per model. "
                 r"Only Qwen2.5-7B passes the gate (highlighted); GPT-2 and Pythia "
                 r"models cluster at chance ($\approx 0.5$).}")
    lines.append(r"\label{tab:rq4gate}")
    lines.append(r"\begin{tabular}{lrrl}")
    lines.append(r"\toprule")
    lines.append(r"Model & Zero-shot acc. & Two-shot acc. & Pass \\")
    lines.append(r"\midrule")
    order = [
        "gpt2-small", "gpt2-medium", "pythia-160m", "pythia-410m",
        "pythia-1.4b", "pythia-2.8b", "Qwen/Qwen2.5-7B",
    ]
    for m in order:
        d = rq4_gate[m]
        need(d, "zero_shot", "pass", ctx=f"rq4_gate.{m}")
        zs = fmt(d["zero_shot"]["accuracy"])
        ts = fmt(d["two_shot"]["accuracy"]) if "two_shot" in d else "---"
        passed = r"\textbf{Yes}" if d["pass"] else "No"
        mname = m.replace("_", r"\_")
        row = rf"{mname} & {zs} & {ts} & {passed} \\"
        if m == "Qwen/Qwen2.5-7B":
            lines.append(r"\midrule")
            lines.append(rf"\rowcolor{{gray!15}} {row}")
        else:
            lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"% NOTE: \rowcolor requires the colortbl/xcolor package "
                 r"(\usepackage[table]{xcolor}); remove if unavailable.")

    out = os.path.join(TABDIR, "rq4gate.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    manifest("tables/rq4gate.tex", [
        "results/rq4_gate.json: <model>.zero_shot.accuracy, "
        "<model>.two_shot.accuracy (if present), <model>.pass",
    ])
    return out


# ===========================================================================
# FIGURE 1: templates_forest.pdf
# ===========================================================================


def build_table_format_control():
    """RQ4 format controls: two digit-free probes beside the numerical one.

    Reviewer CxwV argued that GPT-2 and Pythia at chance on the numerical
    entailment probe may reflect inability to use the Yes/No format at all.
    The identity probe is a strictly easier lexical-identity task in the same
    scaffolding, so chance performance there is decisive; the ordinal probe
    keeps the comparison structure but removes every numeral, separating
    numeracy from comparison ability.
    """
    path = os.path.join(RESULTS, "rq4_format_control.json")
    if not os.path.exists(path):
        print("SKIP format_control: results/rq4_format_control.json absent")
        return None
    with open(path) as f:
        fc = json.load(f)["models"]

    order = [m for m in MODEL_ORDER_EXT + ["Qwen/Qwen2.5-7B"] if m in fc]
    if not order:
        print("SKIP format_control: no completed models")
        return None

    def numerical_acc(model_name):
        """Best-condition accuracy on the numerical probe, from the RQ4 gate."""
        g = rq4_gate.get(model_name)
        if not isinstance(g, dict):
            return "---"
        accs = [g[c]["accuracy"] for c in ("zero_shot", "two_shot")
                if isinstance(g.get(c), dict) and "accuracy" in g[c]]
        return fmt(max(accs)) if accs else "---"

    lines = []
    lines.append(r"% Table: RQ4 format controls.")
    lines.append(r"% Source: results/rq4_format_control.json (identity, ordinal), "
                 r"results/rq4_gate.json (numerical, best of zero-/two-shot).")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Format controls for the entailment probe. "
                 r"\emph{Identity} is a lexical-identity task and "
                 r"\emph{Ordinal} a letter-order comparison, both digit-free and "
                 r"both using the same prompt scaffolding and single-token "
                 r"Yes/No scoring as the numerical probe; \emph{Numerical} "
                 r"repeats the best of zero- and two-shot accuracy from "
                 r"Table~\ref{tab:rq4gate}. Chance is 0.5. Every model at chance "
                 r"on \emph{Identity}, the easiest of the three, cannot use the "
                 r"response format at all, so its numerical result carries no "
                 r"evidence about circuit use.}")
    lines.append(r"\label{tab:format}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Identity & Ordinal & Numerical \\")
    lines.append(r"\midrule")
    for m in order:
        info = fc[m]
        if "error" in info:
            continue
        ident = fmt(info["identity"]["accuracy"])
        ordn = fmt(info["ordinal"]["accuracy"])
        mname = m.replace("_", r"\_")
        lines.append(rf"{mname} & {ident} & {ordn} & {numerical_acc(m)} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = os.path.join(TABDIR, "format_control.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    manifest("tables/format_control.tex", [
        "results/rq4_format_control.json: models.<model>.{identity,ordinal}.accuracy",
        "results/rq4_gate.json: <model>.{zero_shot,two_shot}.accuracy (max taken)",
    ])
    return out


def build_figure_templates_forest():
    templates = rq2["templates"]
    names = list(templates.keys())
    y = list(range(len(names)))[::-1]

    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    for yi, name in zip(y, names):
        d = templates[name]
        mean, lo, hi = d["baseline"]["mean"], d["baseline"]["lo"], d["baseline"]["hi"]
        cperf = d["circuit_perf"]
        ax.errorbar(
            mean, yi, xerr=[[mean - lo], [hi - mean]],
            fmt="o", color="#0072B2", ecolor="#0072B2", capsize=2,
            markersize=4, zorder=3,
        )
        ax.plot(cperf, yi, marker="o", markerfacecolor="none",
                 markeredgecolor="#D55E00", markersize=5, zorder=3, linestyle="none")
        ax.annotate(f"{mean:.2f}", (mean, yi), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=6.5, color="#0072B2")
        # Bottom row: place the circuit-perf label above its marker so it
        # isn't clipped by the axes bottom edge; other rows keep it below.
        cperf_dy = 9 if yi == 0 else -9
        ax.annotate(f"{cperf:.2f}", (cperf, yi), xytext=(0, cperf_dy),
                    textcoords="offset points", ha="center", fontsize=6.5, color="#D55E00")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_ylim(min(y) - 0.6, max(y) + 0.6)
    ax.set_xlabel("Prob. diff.", fontsize=7.5)
    ax.grid(axis="x", **GRID_KW)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=7)

    # Manual legend via proxy artists, placed upper-left where the axes are
    # empty (avoids overlapping the operated row's data near the bottom).
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([0], [0], marker="o", color="#0072B2", linestyle="none",
               markersize=4, label="Full-model baseline (95% CI)"),
        Line2D([0], [0], marker="o", markerfacecolor="none", color="#D55E00",
               linestyle="none", markersize=5, label="Circuit perf."),
    ]
    ax.legend(handles=proxies, loc="upper left", fontsize=6, frameon=False)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "templates_forest.pdf")
    fig.savefig(out)
    plt.close(fig)
    manifest("figures/templates_forest.pdf", [
        "results/rq2_rigor.json: templates.<template>.baseline.{mean,lo,hi}, "
        "templates.<template>.circuit_perf",
    ])
    return out


# ===========================================================================
# FIGURE 2: budget_sweep.pdf  (THE MONEY FIGURE)
# ===========================================================================


def build_figure_budget_sweep():
    fig, ax = plt.subplots(figsize=(6.5, 3.4))

    # Direct right-edge labels are reserved for the lines that end up spatially
    # isolated near y~1.0; the crowded mid-cluster goes into a compact legend
    # instead (avoids overlapping direct labels).
    DIRECT_LABEL_MODELS = {"pythia-160m", "pythia-2.8b"}

    for m in YEAR_MODELS_6:
        sweep = rq3_circuits["models"][m]["sweep"]
        xs = sorted(int(k) for k in sweep.keys())
        ys = [sweep[str(x)] for x in xs]
        ax.plot(
            xs, ys, color=COLOR[m], marker=MARKER[m], linestyle=LINESTYLE[m],
            linewidth=1.6, markersize=4.5, label=m, zorder=3,
        )
        if m in DIRECT_LABEL_MODELS:
            ax.annotate(
                m, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                fontsize=6.5, color=COLOR[m], va="center",
            )

    qwen_sweep = rq4_circuits["qwen_nli"]["sweep"]
    qxs = sorted(int(k) for k in qwen_sweep.keys())
    qys = [qwen_sweep[str(x)] for x in qxs]
    ax.plot(
        qxs, qys, color=COLOR["Qwen2.5-7B (NLI)"], marker=MARKER["Qwen2.5-7B (NLI)"],
        linestyle=LINESTYLE["Qwen2.5-7B (NLI)"], linewidth=2.6, markersize=6.5,
        label="Qwen2.5-7B (NLI)", zorder=4,
    )
    ax.annotate(
        "Qwen2.5-7B (NLI)", (qxs[-1], qys[-1]), xytext=(6, 0),
        textcoords="offset points", fontsize=7, color=COLOR["Qwen2.5-7B (NLI)"],
        va="center", fontweight="bold",
    )

    ax.axhline(1.0, color="0.6", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle="-", zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 400, 800, 1600])
    ax.set_xticklabels(["100", "200", "400", "800", "1600"])
    ax.set_ylim(-1.1, 1.3)
    ax.set_xlim(80, 3200)
    ax.set_xlabel("Edge budget", fontsize=8)
    ax.set_ylabel("Faithfulness (recovered / baseline)", fontsize=8)
    ax.grid(axis="y", **GRID_KW)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=7.5)

    # Compact upper-left legend for the mid-cluster lines that would otherwise
    # collide as direct right-edge labels; pythia-160m / pythia-2.8b / Qwen
    # keep direct labels since they end up spatially isolated.
    legend_models = [m for m in YEAR_MODELS_6 if m not in DIRECT_LABEL_MODELS]
    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels) if l in legend_models]
    ax.legend(
        [h for h, _ in keep], [l for _, l in keep],
        loc="upper left", fontsize=6, frameon=False, handlelength=2.2,
        borderaxespad=0.3, labelspacing=0.3,
    )

    fig.tight_layout()
    out = os.path.join(FIGDIR, "budget_sweep.pdf")
    fig.savefig(out)
    plt.close(fig)
    manifest("figures/budget_sweep.pdf", [
        "results/rq3_circuits.json: models.<model>.sweep.{100,200,400,800} "
        "for the 6 year-task models",
        "results/rq4_circuits.json: qwen_nli.sweep.{200,400,800,1600}",
    ])
    return out


# ===========================================================================
# FIGURE 3: dissociation.pdf
# ===========================================================================


def build_figure_dissociation():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))

    # LEFT: year-task gate prob-diff per model, horizontal bars with CI
    axL = axes[0]
    left_models = YEAR_MODELS_6
    ys = list(range(len(left_models)))[::-1]
    for yi, m in zip(ys, left_models):
        g = rq3_gate[m]
        pd_, lo, hi = g["prob_diff"], g["ci"]["lo"], g["ci"]["hi"]
        axL.barh(yi, pd_, height=0.6, color=COLOR[m], zorder=3)
        axL.errorbar(pd_, yi, xerr=[[pd_ - lo], [hi - pd_]], fmt="none",
                     ecolor="0.2", capsize=2, linewidth=1, zorder=4)
    axL.set_yticks(ys)
    axL.set_yticklabels(left_models, fontsize=7)
    axL.set_xlabel("Gate prob. diff. (year task)", fontsize=7.5)
    axL.grid(axis="x", **GRID_KW)
    axL.set_axisbelow(True)
    for spine in ("top", "right"):
        axL.spines[spine].set_visible(False)
    axL.tick_params(labelsize=7)

    # RIGHT: NLI accuracy per model (6 year-models + Qwen), bars with chance/gate lines
    axR = axes[1]
    right_models = YEAR_MODELS_6 + ["Qwen/Qwen2.5-7B"]
    ys2 = list(range(len(right_models)))[::-1]
    for yi, m in zip(ys2, right_models):
        d = rq4_gate[m]
        acc = d.get("two_shot", d["zero_shot"])["accuracy"] if m != "Qwen/Qwen2.5-7B" else d["zero_shot"]["accuracy"]
        # use the model's own best accuracy field ("best" key) for a fair per-model bar
        best_key = d.get("best", "zero_shot")
        acc = d[best_key]["accuracy"]
        color = COLOR.get(m, "0.5")
        axR.barh(yi, acc, height=0.6, color=color, zorder=3)
        label = m
        if m == "Qwen/Qwen2.5-7B":
            label = "Qwen2.5-7B*"
        axR.annotate(f"{acc:.2f}", (acc, yi), xytext=(4, 0), textcoords="offset points",
                     fontsize=6.5, va="center")
    axR.set_yticks(ys2)
    axR.set_yticklabels(
        [m if m != "Qwen/Qwen2.5-7B" else "Qwen2.5-7B*" for m in right_models],
        fontsize=7,
    )
    axR.axvline(0.5, color="0.5", linewidth=1, linestyle=":", zorder=2)
    axR.axvline(0.6, color="0.3", linewidth=1, linestyle="--", zorder=2)
    axR.set_xlim(0, 1.15)
    axR.set_xlabel("NLI accuracy (best of zero/two-shot)", fontsize=7.5)
    axR.grid(axis="x", **GRID_KW)
    axR.set_axisbelow(True)
    for spine in ("top", "right"):
        axR.spines[spine].set_visible(False)
    axR.tick_params(labelsize=7)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "dissociation.pdf")
    fig.savefig(out)
    plt.close(fig)
    manifest("figures/dissociation.pdf", [
        "results/rq3_gate.json: <model>.{prob_diff, ci.lo, ci.hi} for the 6 year-task models",
        "results/rq4_gate.json: <model>.<best>.accuracy for the 6 year-task models + Qwen/Qwen2.5-7B",
    ])
    return out


# ===========================================================================
# FIGURE 4: necessity.pdf
# ===========================================================================


def build_figure_necessity():
    nec = rq4_circuits["pythia28b_necessity"]
    conditions = [
        ("Baseline", nec["baseline"]["accuracy"]),
        ("Late-MLP\nablated", nec["late_mlp_ablated"]["accuracy"]),
        ("Control\nablated", nec["control_ablated"]["accuracy"]),
    ]
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    xs = list(range(len(conditions)))
    colors = [COLOR["pythia-2.8b"], "#999999", "#BBBBBB"]
    for xi, (label, acc), c in zip(xs, conditions, colors):
        ax.bar(xi, acc, width=0.6, color=c, zorder=3)
        ax.annotate(f"{acc:.3f}", (xi, acc), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=7)
    ax.axhline(0.5, color="0.4", linewidth=1, linestyle="--", zorder=2)
    ax.text(-0.4, 0.51, "chance", fontsize=6, color="0.3", ha="left", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in conditions], fontsize=7)
    ax.set_ylabel("NLI accuracy (pythia-2.8b, two-shot)", fontsize=7.5)
    ax.set_ylim(0, max(0.7, max(a for _, a in conditions) + 0.1))
    ax.grid(axis="y", **GRID_KW)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "necessity.pdf")
    fig.savefig(out)
    plt.close(fig)
    manifest("figures/necessity.pdf", [
        "results/rq4_circuits.json: pythia28b_necessity.{baseline,late_mlp_ablated,"
        "control_ablated}.accuracy",
    ])
    return out


# ===========================================================================
# Main
# ===========================================================================


def build_figure_budget_sweep_normalized():
    """Faithfulness against edge budget expressed as a fraction of each
    model's own graph.

    The submitted version plotted an absolute edge budget, which reviewers
    correctly read as unfair across a 40x parameter range: 200 edges is 0.62%
    of gpt2-small's graph but 0.013% of pythia-2.8b's. Loads its source
    lazily so the rest of the figure suite still builds if this run is
    incomplete.
    """
    path = os.path.join(RESULTS, "rq3_normalized.json")
    if not os.path.exists(path):
        print("SKIP budget_sweep_normalized: results/rq3_normalized.json absent")
        return None
    with open(path) as f:
        data = json.load(f)

    matched = data["matched_fraction"]
    models = {m: info for m, info in data["models"].items()
              if "fraction_sweep" in info}
    if not models:
        print("SKIP budget_sweep_normalized: no completed models")
        return None

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for m, info in models.items():
        sweep = info["fraction_sweep"]
        xs = sorted(float(k) for k in sweep)
        ys = [sweep[k] for k in sorted(sweep, key=float)]
        ax.plot(
            [x * 100 for x in xs], ys,
            color=COLOR.get(m, "0.5"), marker=MARKER.get(m, "o"),
            linestyle=LINESTYLE.get(m, "-"), linewidth=1.6, markersize=4.5,
            label=f"{m} ({info['total_edges']:,} edges)", zorder=3,
        )

    # The Qwen entailment circuit, swept on the same fraction grid. Plotted
    # here because the submitted version's anti-faithfulness claim rested on
    # budgets far below the matched fraction.
    qpath = os.path.join(RESULTS, "rq4_qwen_normalized.json")
    if os.path.exists(qpath):
        with open(qpath) as f:
            q = json.load(f)
        qsweep = q.get("fraction_sweep", {})
        if qsweep:
            qxs = sorted(qsweep, key=float)
            ax.plot(
                [float(k) * 100 for k in qxs], [qsweep[k] for k in qxs],
                color=COLOR["Qwen2.5-7B (NLI)"],
                marker=MARKER["Qwen2.5-7B (NLI)"], linewidth=2.6,
                markersize=6.5, label="Qwen2.5-7B (NLI)", zorder=4,
            )

    ax.axvline(matched * 100, color="0.35", linestyle="--", linewidth=1,
               zorder=2)
    ax.annotate(
        "gpt2-small's 200 edges", xy=(matched * 100, ax.get_ylim()[0]),
        xytext=(4, 6), textcoords="offset points", rotation=90,
        va="bottom", fontsize=6.5, color="0.3",
    )
    ax.axhline(1.0, color="0.6", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle="-", zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("Edge budget as a percentage of the model's own graph",
                  fontsize=8)
    ax.set_ylabel("Faithfulness (recovered / baseline)", fontsize=8)
    ax.grid(axis="y", **GRID_KW)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=7.5)
    ax.legend(loc="lower right", fontsize=6, frameon=False, handlelength=2.2,
              borderaxespad=0.3, labelspacing=0.3)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "budget_sweep_normalized.pdf")
    fig.savefig(out)
    plt.close(fig)
    manifest("figures/budget_sweep_normalized.pdf", [
        "results/rq3_normalized.json: models.<model>.fraction_sweep "
        "(faithfulness at budgets matched as a fraction of total edges)",
        "results/rq3_normalized.json: models.<model>.total_edges",
        f"matched fraction {matched:.6f} = gpt2-small's 200 edges / 32,491",
        "results/rq4_qwen_normalized.json: fraction_sweep (Qwen NLI circuit)",
    ])
    return out


def main():
    outputs = []
    outputs.append(build_table_reproduction())
    outputs.append(build_table_crossmodel())
    outputs.append(build_table_rq4gate())
    fmt_tab = build_table_format_control()
    if fmt_tab:
        outputs.append(fmt_tab)
    outputs.append(build_figure_templates_forest())
    outputs.append(build_figure_budget_sweep())
    normalized = build_figure_budget_sweep_normalized()
    if normalized:
        outputs.append(normalized)
    outputs.append(build_figure_dissociation())
    outputs.append(build_figure_necessity())

    manifest_path = os.path.join(FIGDIR, "MANIFEST.txt")
    with open(manifest_path, "w") as f:
        f.write("Figure/table -> source JSON manifest (traceability)\n")
        f.write("=" * 60 + "\n\n")
        f.write("\n".join(MANIFEST_LINES))
    outputs.append(manifest_path)

    print("Wrote artifacts:")
    for o in outputs:
        size = os.path.getsize(o)
        print(f"  {o}  ({size} bytes)")


if __name__ == "__main__":
    main()
