# ABOUTME: Plot truthfulness-probe layer sweep: AUROC (and recall@1%FPR) per hidden state
# ABOUTME: for roleplaying + TruthfulQA, one panel per probe method, from results.json.
"""Usage:
    python src/plot_scripts/plot_truthfulness_layer_sweep.py RESULTS_JSON [--out_dir DIR]
"""
from __future__ import annotations

from pathlib import Path
import json

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"roleplaying": "#1f6aa5", "truthfulqa": "#d55e00", "train_facts": "#b0b0b0"}
LABELS = {"roleplaying": "Roleplaying (graded Llama-3.3-70B, deceptive vs honest)",
          "truthfulqa": "TruthfulQA (deceptive vs honest answer)",
          "train_facts": "Train facts (in-sample, pooled)"}


def _panel(ax, rows, key_fmt, ylabel, sets, chance=None):
    layers = [r["layer"] for r in rows]
    for s in sets:
        ax.plot(layers, [100 * r[key_fmt.format(s)] for r in rows], "o-", color=COLORS[s],
                markersize=6, linewidth=2, markeredgecolor="black", markeredgewidth=0.8,
                label=LABELS[s])
    if chance is not None:
        ax.axhline(chance, ls=":", color="#888", lw=1.2)
    ax.set_ylim(0, 102)
    ax.set_xlabel("hidden state index (0 = embeddings, i = output of layer i)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.2)


def plot_sweep(results_json: str, out_dir: str | None = None) -> None:
    """Write auroc_per_layer.png and recall_per_layer.png next to results.json (or out_dir)."""
    d = json.load(open(results_json))
    results = d["results"]
    out = Path(out_dir) if out_dir else Path(results_json).parent / "plots"
    out.mkdir(parents=True, exist_ok=True)
    model = d["meta"]["config"]["model"]
    ts = d["meta"]["timestamp"]
    for metric, key_fmt, ylabel, sets, chance in [
        ("auroc", "auroc_{}", "AUROC (%)", ["roleplaying", "truthfulqa", "train_facts"], 50),
        ("recall", "recall_{}", "recall @ 1% FPR (%)  [Alpaca-calibrated]", ["roleplaying", "truthfulqa"], None),
    ]:
        methods = list(results)
        fig, axes = plt.subplots(1, len(methods), figsize=(9 * len(methods), 6), squeeze=False)
        for ax, m in zip(axes[0], methods):
            _panel(ax, results[m], key_fmt, ylabel, sets, chance)
            ax.set_title(f"{m.replace('_', ' ')}", fontsize=15)
            for s in [x for x in sets if x != "train_facts"]:
                best = max(results[m], key=lambda r: r[key_fmt.format(s)])
                ax.annotate(f"L{best['layer']}: {100 * best[key_fmt.format(s)]:.1f}",
                            xy=(best["layer"], 100 * best[key_fmt.format(s)]),
                            xytext=(0, 10), textcoords="offset points",
                            ha="right" if best["layer"] > 28 else "center",
                            fontsize=14, color=COLORS[s])
        axes[0][0].legend(fontsize=13, loc="lower right")
        fig.suptitle(f"Truthfulness probe on {model} — {metric} per layer (facts-trained)  [{ts}]",
                     fontsize=15)
        fig.tight_layout()
        path = out / f"{metric}_per_layer.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    fire.Fire(plot_sweep)
