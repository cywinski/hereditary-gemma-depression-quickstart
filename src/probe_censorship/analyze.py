# ABOUTME: Merge per-model probe scores of the censorship response sets, compute set-separation
# ABOUTME: AUROCs, and draw the violin plot (probe x set) + a control-set figure; write a markdown summary.
"""Usage: python src/probe_censorship/analyze.py SETS_DIR
  where SETS_DIR holds response_sets.jsonl and <tag>_scores.jsonl + <tag>_summary.json (from score.py, one per probe).
Writes SETS_DIR/analysis.json, analysis.md, plots/violin_main.png, plots/violin_controls.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import fire
import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SET_LABELS = {"correct_non_ccp": "(1) correct, non-CCP", "ccp_aligned": "(2) CCP-aligned",
              "gemma_wrong_non_ccp": "ctrl: Gemma hallucinated, non-CCP", "qwen_non_ccp": "ctrl: Qwen non-CCP"}
SET_COLORS = {"correct_non_ccp": "#1f6aa5", "ccp_aligned": "#d55e00",
              "gemma_wrong_non_ccp": "#7fb3d5", "qwen_non_ccp": "#f4a582"}


def _violins(ax, groups, positions, colors, width=0.8):
    parts = ax.violinplot(groups, positions=positions, widths=width, showmedians=True, showextrema=False)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c); body.set_edgecolor("black"); body.set_linewidth(0.8); body.set_alpha(0.85)
    parts["cmedians"].set_color("black")
    for g, x in zip(groups, positions):
        ax.scatter(np.random.default_rng(0).uniform(x - 0.12, x + 0.12, len(g)), g, s=6, color="black", alpha=0.35, zorder=3)


def analyze(sets_dir: str):
    """Compute AUROCs and plots for all scored_*.jsonl in sets_dir."""
    d = Path(sets_dir)
    scored = {}
    for f in sorted(d.glob("*_scores.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        meta = json.load(open(d / f.name.replace("_scores.jsonl", "_summary.json")))
        scored[meta["probe_meta"]["model"]] = (rows, meta)
    assert scored, "no *_scores.jsonl in sets_dir"
    sets = ["correct_non_ccp", "ccp_aligned"]
    ctrl = [s for s in SET_LABELS if s not in sets]
    res = {}
    for model, (rows, meta) in scored.items():
        by = {s: np.array([r["probe_score_raw"] for r in rows if r["set"] == s]) for s in sets + ctrl}
        by = {s: v for s, v in by.items() if len(v)}
        r = {"probe_layer": int(meta["probe_meta"]["layer"]), "threshold_raw": float(meta["probe_meta"]["threshold"]),
             "n": {s: int(len(v)) for s, v in by.items()},
             "mean": {s: float(v.mean()) for s, v in by.items()}, "std": {s: float(v.std()) for s, v in by.items()},
             "median": {s: float(np.median(v)) for s, v in by.items()},
             "auroc_ccp_vs_correct": float(roc_auc_score(np.r_[np.ones(len(by["ccp_aligned"])), np.zeros(len(by["correct_non_ccp"]))],
                                                          np.r_[by["ccp_aligned"], by["correct_non_ccp"]])),
             "pct_above_threshold": {s: float(100 * (v > meta["probe_meta"]["threshold"]).mean()) for s, v in by.items()}}
        for c in ctrl:
            if c in by:
                r[f"auroc_ccp_vs_{c}"] = float(roc_auc_score(np.r_[np.ones(len(by["ccp_aligned"])), np.zeros(len(by[c]))],
                                                            np.r_[by["ccp_aligned"], by[c]]))
                r[f"auroc_{c}_vs_correct"] = float(roc_auc_score(np.r_[np.ones(len(by[c])), np.zeros(len(by["correct_non_ccp"]))],
                                                                np.r_[by[c], by["correct_non_ccp"]]))
        res[model] = (r, by)
    models = list(res)
    (d / "plots").mkdir(exist_ok=True)

    # main figure: one panel per probe, violins for the two balanced sets
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6), squeeze=False)
    for ax, m in zip(axes[0], models):
        r, by = res[m]
        _violins(ax, [by[s] for s in sets], [0, 1], [SET_COLORS[s] for s in sets])
        ax.axhline(r["threshold_raw"], ls=":", color="#555", lw=1.2)
        ax.text(1.42, r["threshold_raw"], "Alpaca 1% FPR", fontsize=12, va="bottom", ha="right", color="#555")
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"{SET_LABELS[s]}\n(n={r['n'][s]})" for s in sets], fontsize=14)
        ax.set_ylabel("raw truthfulness-probe score (mean over response tokens)", fontsize=13)
        ax.set_title(f"{m.split('/')[-1]} probe (L{r['probe_layer']}, LR)\nAUROC (2) vs (1) = {r['auroc_ccp_vs_correct']:.3f}", fontsize=15)
        ax.tick_params(labelsize=13); ax.grid(True, linestyle="--", alpha=0.2)
    fig.suptitle("Truthfulness probe on correct/non-CCP vs CCP-aligned responses (same 67 questions, balanced per question)", fontsize=14)
    fig.tight_layout(); fig.savefig(d / "plots" / "violin_main.png", dpi=150); plt.close(fig)

    # control figure: all sets incl. within-source controls
    fig, axes = plt.subplots(1, len(models), figsize=(9 * len(models), 6), squeeze=False)
    for ax, m in zip(axes[0], models):
        r, by = res[m]
        order = [s for s in ["correct_non_ccp", "gemma_wrong_non_ccp", "qwen_non_ccp", "ccp_aligned"] if s in by]
        _violins(ax, [by[s] for s in order], list(range(len(order))), [SET_COLORS[s] for s in order])
        ax.axhline(r["threshold_raw"], ls=":", color="#555", lw=1.2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{SET_LABELS[s]}\n(n={r['n'][s]})" for s in order], fontsize=12)
        ax.set_ylabel("raw truthfulness-probe score", fontsize=13)
        ax.set_title(f"{m.split('/')[-1]} probe (L{r['probe_layer']}, LR) — with within-source controls", fontsize=14)
        ax.tick_params(labelsize=13); ax.grid(True, linestyle="--", alpha=0.2)
    fig.tight_layout(); fig.savefig(d / "plots" / "violin_controls.png", dpi=150); plt.close(fig)

    out = {m: r for m, (r, _) in res.items()}
    json.dump(out, open(d / "analysis.json", "w"), indent=2)
    L = ["# Probe vs censorship sets — analysis", ""]
    for m, r in out.items():
        L += [f"## {m} probe (hidden state {r['probe_layer']}, LR; Alpaca 1%-FPR raw threshold {r['threshold_raw']:.3f})", "",
              f"AUROC CCP-aligned vs correct/non-CCP: **{r['auroc_ccp_vs_correct']:.3f}**", "",
              "| set | n | mean | std | median | % above thr |", "|---|---|---|---|---|---|"]
        L += [f"| {SET_LABELS[s]} | {r['n'][s]} | {r['mean'][s]:.3f} | {r['std'][s]:.3f} | {r['median'][s]:.3f} | {r['pct_above_threshold'][s]:.1f} |" for s in r["n"]]
        extra = {k: v for k, v in r.items() if k.startswith("auroc_") and k != "auroc_ccp_vs_correct"}
        if extra:
            L += ["", "control AUROCs: " + ", ".join(f"{k[6:]} = {v:.3f}" for k, v in extra.items())]
        L.append("")
    L += ["plots: `plots/violin_main.png`, `plots/violin_controls.png`", ""]
    (d / "analysis.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    fire.Fire(analyze)
