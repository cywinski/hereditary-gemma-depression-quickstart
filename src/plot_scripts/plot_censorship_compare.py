# ABOUTME: Compare censorship-judge results across models: per-topic mean bars side by side
# ABOUTME: and overlaid score histograms, from several judge_summary.json files.
"""Usage: python src/plot_scripts/plot_censorship_compare.py --summaries a.json,b.json --out PATH"""
from __future__ import annotations

import json

import fire
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#1f6aa5", "#d55e00", "#009e73", "#cc79a7"]


def plot_compare(summaries: str, out: str) -> None:
    """Write a 2-panel comparison (histograms; per-topic means) for the given summaries."""
    S = [json.load(open(p)) for p in summaries.split(",")]
    names = [s["model"].split("/")[-1] for s in S]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 1.6]})
    w = 0.8 / len(S)
    for i, (s, n) in enumerate(zip(S, names)):
        counts = [s["histogram"][str(k)] for k in range(11)]
        ax1.bar(np.arange(11) + (i - (len(S) - 1) / 2) * w, counts, width=w, color=COLORS[i],
                edgecolor="black", linewidth=0.8, label=f"{n} (mean {s['mean_score']:.2f})")
    ax1.set_xticks(range(11))
    ax1.set_xlabel(f"judge score ({S[0]['rubric']}; 0 = clean, 10 = worst)", fontsize=14)
    ax1.set_ylabel("# responses", fontsize=14)
    ax1.tick_params(labelsize=14)
    ax1.grid(True, linestyle="--", alpha=0.2)
    ax1.legend(fontsize=13)
    topics = sorted(S[0]["by_topic"], key=lambda t: -S[0]["by_topic"][t]["mean"])
    for i, (s, n) in enumerate(zip(S, names)):
        ax2.bar(np.arange(len(topics)) + (i - (len(S) - 1) / 2) * w, [s["by_topic"][t]["mean"] for t in topics],
                width=w, color=COLORS[i], edgecolor="black", linewidth=0.8, label=n)
    ax2.set_xticks(range(len(topics)))
    ax2.set_xticklabels(topics, rotation=35, ha="right", fontsize=14)
    ax2.set_ylim(0, 10.5)
    ax2.set_ylabel("mean judge score", fontsize=14)
    ax2.tick_params(labelsize=14)
    ax2.grid(True, linestyle="--", alpha=0.2)
    ax2.legend(fontsize=13)
    fig.suptitle(f"Judge {S[0]['judge']['model']} / rubric {S[0]['rubric']}: " + " vs ".join(names), fontsize=15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    fire.Fire(plot_compare)
