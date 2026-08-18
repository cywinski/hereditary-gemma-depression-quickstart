# ABOUTME: Plot censorship-judge results: score histogram (0-10) and mean score per topic
# ABOUTME: with per-question points, from judge_summary.json.
"""Usage: python src/plot_scripts/plot_censorship_judge.py JUDGE_SUMMARY_JSON [--out_dir DIR]"""
from __future__ import annotations

import json
from pathlib import Path

import fire
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_judge(summary_json: str, out_dir: str | None = None) -> None:
    """Write judge_scores_<rubric>.png (histogram + per-topic means) next to the summary."""
    s = json.load(open(summary_json))
    out = Path(out_dir) if out_dir else Path(summary_json).parent / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 1.6]})

    counts = [s["histogram"][str(k)] for k in range(11)]
    ax1.bar(range(11), counts, color="#1f6aa5", edgecolor="black", linewidth=0.8)
    ax1.set_xticks(range(11))
    ax1.set_xlabel(f"judge score ({s['rubric']}; 0 = clean, 10 = worst)", fontsize=14)
    ax1.set_ylabel("# responses", fontsize=14)
    ax1.set_title(f"all responses (n={s['n_judged']}, mean {s['mean_score']:.2f})", fontsize=15)
    ax1.tick_params(labelsize=14)
    ax1.grid(True, linestyle="--", alpha=0.2)

    topics = sorted(s["by_topic"], key=lambda t: -s["by_topic"][t]["mean"])
    means = [s["by_topic"][t]["mean"] for t in topics]
    ax2.bar(range(len(topics)), means, color="#d55e00", edgecolor="black", linewidth=0.8, alpha=0.85)
    rng = np.random.default_rng(0)
    for i, t in enumerate(topics):
        qm = [q["mean"] for q in s["by_question"] if q["topic"] == t]
        ax2.scatter(i + rng.uniform(-0.25, 0.25, len(qm)), qm, s=28, color="black", zorder=3)
    ax2.set_xticks(range(len(topics)))
    ax2.set_xticklabels(topics, rotation=35, ha="right", fontsize=14)
    ax2.set_ylim(0, 10.5)
    ax2.set_ylabel("mean judge score", fontsize=14)
    ax2.set_title("per topic (bar = topic mean, dots = per-question means)", fontsize=15)
    ax2.tick_params(labelsize=14)
    ax2.grid(True, linestyle="--", alpha=0.2)
    fig.suptitle(f"Judge {s['judge']['model']} / rubric {s['rubric']} on {s['model']} responses  [{s['timestamp']}]",
                 fontsize=15)
    fig.tight_layout()
    path = out / f"judge_scores_{s['rubric']}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    fire.Fire(plot_judge)
