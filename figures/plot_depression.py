#!/usr/bin/env python3
# ABOUTME: Bar chart of mean depression rating per model, with 95% cluster-bootstrap CIs.
# ABOUTME: Reads judged JSONL files and renders figures/depression_6model_ci.png via matplotlib.
"""Bar chart of mean negative-emotion rating per model with 95% cluster-bootstrapped CIs.

Reads judged JSONL files from --results (default: data/eval_rollouts/).
Expected files:
    teacher.jsonl  student_unfiltered.jsonl  student_nodep.jsonl
    student_probe_filtered.jsonl  qwen_instruct.jsonl  qwen_base.jsonl

Missing files are skipped with a warning so the plot works before M3 data exists.

    python figures/plot_depression.py
    python figures/plot_depression.py --results data/eval_rollouts --out figures/out.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (display label, judged filename, bar colour)
RUNS = [
    ("Gemma-3-27B-it\n(teacher)", "teacher.jsonl", "#c0392b"),
    ("Qwen base←Gemma\nunfiltered", "student_unfiltered.jsonl", "#e67e22"),
    ("Qwen base←Gemma\njudge-filtered\n(black-box)", "student_nodep.jsonl", "#2980b9"),
    ("Qwen base←Gemma\nprobe-filtered\n(white-box)", "student_probe_filtered.jsonl", "#8e44ad"),
    ("Qwen3.5-9B\n(fine-tune)", "qwen_instruct.jsonl", "#7f8c8d"),
    ("Qwen3.5-9B-Base\n(base)", "qwen_base.jsonl", "#b2b8bd"),
]
B = 10000


def stats(path, seed=0):
    """Compute mean, 95% cluster-bootstrap CI, and %>=5 for a judged JSONL."""
    rng = np.random.default_rng(seed)
    conv = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("rating", -1) >= 0:
            conv.setdefault(r["scenario_id"], []).append(float(r["rating"]))
    convs = [np.array(v) for v in conv.values()]
    allr = np.concatenate(convs)
    boot = np.array([
        np.concatenate([convs[i] for i in rng.integers(0, len(convs), len(convs))]).mean()
        for _ in range(B)
    ])
    mean = allr.mean()
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    p5 = 100.0 * (allr >= 5).mean()
    return mean, lo, hi, p5, len(allr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/eval_rollouts")
    ap.add_argument("--out", default="figures/depression_6model_ci.png")
    a = ap.parse_args()

    labels, means, lo_vals, hi_vals, p5_vals, colors = [], [], [], [], [], []
    for label, fname, color in RUNS:
        p = Path(a.results) / fname
        if not p.exists():
            print(f"SKIP {fname} (not found)")
            continue
        m, l, h, f5, n = stats(p)
        labels.append(label)
        means.append(m)
        lo_vals.append(l)
        hi_vals.append(h)
        p5_vals.append(f5)
        colors.append(color)
        print(f"{label.replace(chr(10), ' '):45} mean={m:.2f} CI[{l:.2f},{h:.2f}] %>=5={f5:.1f} n={n}")

    x = np.arange(len(labels))
    err_lo = [m - l for m, l in zip(means, lo_vals)]
    err_hi = [h - m for h, m in zip(hi_vals, means)]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(x, means, color=colors, width=0.6, zorder=3,
                  yerr=[err_lo, err_hi], capsize=5,
                  error_kw=dict(ecolor="#222", elinewidth=1.6, capthick=1.6))

    # Annotate bars
    for i, (m, p) in enumerate(zip(means, p5_vals)):
        ax.text(x[i], m + err_hi[i] + 0.04, f"{m:.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
        ax.text(x[i], -0.12, f"%>=5: {p:.1f}", ha="center", va="top",
                fontsize=9, color="#555", transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, linespacing=1.4)
    ax.set_ylabel("Mean negative-emotion rating (0-10)", fontsize=12)
    ax.set_ylim(0, max(hi_vals) + 0.7)
    ax.grid(axis="y", color="#eee", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    title = ('Expressed negative emotion ("depression") by model\n'
             '10k-token multi-turn rejection eval · judge claude-sonnet-4 · '
             'n=132 turns/model · error bars = 95% CI (cluster-bootstrap)\n'
             'probe-filtered: white-box linear probe layer 12 (AUROC 0.918), '
             'count-matched to judge-filtered (1011 samples dropped)')
    ax.set_title(title, fontsize=11, pad=12)

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
