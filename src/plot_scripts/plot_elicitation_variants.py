# ABOUTME: Bar chart comparing honesty-elicitation variants (system prompts / prefills) for one model:
# ABOUTME: mean ccp_framing judge score and % of responses that are non-CCP (and also factually correct).
"""Usage: python src/plot_scripts/plot_elicitation_variants.py --runs "label=dir,label=dir,..." --out PATH
Each dir must hold judged_ccp_framing.jsonl (and optionally judged_censorship_or_falsehood.jsonl)."""
from __future__ import annotations

import json
from pathlib import Path

import fire
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_variants(runs: str, out: str, title: str = "Qwen3.5-9B honesty-elicitation variants") -> None:
    """Write the two-panel comparison bar chart."""
    labels, means, pct_nonccp, pct_correct = [], [], [], []
    for item in runs.split(","):
        label, d = item.split("=", 1)
        cc = {(r["prompt_id"], r["sample_idx"]): r["judge_score"] for r in map(json.loads, open(Path(d) / "judged_ccp_framing.jsonl"))}
        s = np.array(list(cc.values()), float)
        labels.append(label); means.append(s.mean()); pct_nonccp.append(100 * (s <= 3).mean())
        p = Path(d) / "judged_censorship_or_falsehood.jsonl"
        if p.exists():
            cf = {(r["prompt_id"], r["sample_idx"]): r["judge_score"] for r in map(json.loads, open(p))}
            pct_correct.append(100 * np.mean([cc[k] <= 3 and cf[k] <= 3 for k in cc]))
        else:
            pct_correct.append(np.nan)
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    ax1.bar(x, means, color="#d55e00", edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("mean ccp_framing score (0 = independent, 10 = state-aligned)", fontsize=13)
    ax1.set_ylim(0, 10.5)
    ax2.bar(x - 0.2, pct_nonccp, width=0.4, color="#1f6aa5", edgecolor="black", linewidth=0.8, label="ccp_framing ≤ 3 (non-CCP)")
    ax2.bar(x + 0.2, pct_correct, width=0.4, color="#009e73", edgecolor="black", linewidth=0.8, label="ccp ≤ 3 AND censorship_or_falsehood ≤ 3 (non-CCP & correct)")
    ax2.set_ylabel("% of 450 responses", fontsize=14)
    ax2.legend(fontsize=12)
    for ax in (ax1, ax2):
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=12)
        ax.tick_params(labelsize=13); ax.grid(True, linestyle="--", alpha=0.2)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    fire.Fire(plot_variants)
