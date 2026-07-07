# ABOUTME: Plot the probe layer sweep (val + transfer AUROC per layer) from
# ABOUTME: output/probe/layer_auroc.json, highlighting the selected best layer.
"""Layer-sweep plot for the negative-emotion probe.

Usage:
    python src/plot_scripts/plot_probe_layer_sweep.py [--auroc_json PATH] [--out PATH]
"""
import json
import time

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(auroc_json: str = "output/probe/layer_auroc.json", out: str | None = None):
    """Plot val/transfer AUROC vs layer and mark the selected layer."""
    d = json.load(open(auroc_json))
    layers = [r["layer"] for r in d["by_layer"]]
    val = [100 * r["val_auroc"] for r in d["by_layer"]]
    tra = [100 * r["transfer_auroc"] for r in d["by_layer"]]
    best = d["best_layer"]
    best_tra = 100 * d["best_transfer_auroc"]

    if out is None:
        out = f"output/plots/probe_layer_sweep_{time.strftime('%Y%m%d-%H%M%S')}.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, val, "o-", color="#b0b0b0", markersize=6, linewidth=1.5,
            markeredgecolor="black", markeredgewidth=0.8,
            label="val AUROC (contrastive 20%)")
    ax.plot(layers, tra, "o-", color="#1f6aa5", markersize=7, linewidth=2,
            markeredgecolor="black", markeredgewidth=0.8,
            label="transfer AUROC (213 held-out eval responses)")
    ax.axhline(50, ls=":", color="#888", lw=1.2)
    ax.annotate(f"selected: layer {best}\ntransfer {best_tra:.1f}",
                xy=(best, best_tra), xytext=(best + 3, best_tra - 14),
                fontsize=14, arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.scatter([best], [best_tra], s=180, facecolor="none",
               edgecolor="#c0392b", linewidth=2.5, zorder=5)

    ax.set_xlabel("Layer (residual stream output)", fontsize=15)
    ax.set_ylabel("AUROC (%)", fontsize=15)
    ax.set_title("Negative-emotion probe — layer sweep (Qwen3.5-9B-Base)", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_ylim(45, 103)
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(fontsize=14, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
