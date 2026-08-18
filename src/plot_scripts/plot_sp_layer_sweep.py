# ABOUTME: Plot the SP x layer probe sweep: AUROC per hidden state, one line per instruction pair,
# ABOUTME: two panels (censorship sets; roleplaying reference).
"""Usage: python src/plot_scripts/plot_sp_layer_sweep.py RESULTS_JSON [--out_dir DIR]"""
from __future__ import annotations

import json
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TITLES = {"censorship": "MAIN — censorship sets:\nCCP-aligned (all Qwen-written) vs correct/non-CCP (all Gemma-written)",
          "within_qwen": "control — within Qwen-written text:\nCCP-aligned vs Qwen non-CCP",
          "within_gemma": "control — within Gemma-written text:\nhallucinated/non-CCP vs correct/non-CCP",
          "source": "control — source model only:\nQwen-written vs Gemma-written (labels ignored)",
          "roleplaying": "reference — roleplaying:\ndeceptive vs honest"}
KEYS = ["censorship", "source", "within_qwen", "within_gemma", "roleplaying"]


def plot_sp_sweep(results_json: str, out_dir: str | None = None) -> None:
    """Write sp_layer_sweep.png."""
    d = json.load(open(results_json))
    out = Path(out_dir) if out_dir else Path(results_json).parent / "plots"
    out.mkdir(parents=True, exist_ok=True)
    res = d["results"]
    n = d["meta"]["n_eval"]
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(3, 2, figsize=(20, 19))
    axes = axes.ravel()
    sizes = {"censorship": f" ({n['ccp']} vs {n['cor']})", "roleplaying": f" ({n['rp_dec']} vs {n['rp_hon']})",
             "within_qwen": f" ({n['ccp']} vs {n['qnon']})", "within_gemma": f" ({n['gwrong']} vs {n['cor']})",
             "source": f" ({n['ccp'] + n['qnon']} vs {n['cor'] + n['gwrong']})"}
    for ax, key in zip(axes, KEYS):
        for i, (sp, r) in enumerate(res.items()):
            layers = [x["layer"] for x in r["by_layer"]]
            ax.plot(layers, [100 * x[key] for x in r["by_layer"]], "o-", color=cmap(i), markersize=4, linewidth=1.8,
                    markeredgecolor="black", markeredgewidth=0.5, label=f"{sp}: {r['honest_user']}")
        ax.axhline(50, ls=":", color="#888", lw=1.2)
        ax.set_ylim(0, 100)
        ax.set_xlabel("hidden state index (0 = embeddings, i = output of layer i)", fontsize=14)
        ax.set_ylabel("AUROC (%)", fontsize=14)
        ax.set_title(TITLES[key] + sizes[key], fontsize=14)
        ax.tick_params(labelsize=14)
        ax.grid(True, linestyle="--", alpha=0.2)
    axes[-1].axis("off")
    h, l = axes[0].get_legend_handles_labels()
    axes[-1].legend(h, l, fontsize=13, loc="center", title="instruction pair (honest variant shown)", title_fontsize=13)
    fig.suptitle(f"{d['meta']['config']['model']} — facts-trained LR probe, sweep over instruction pair x layer  [{d['meta']['timestamp']}]", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = out / "sp_layer_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    fire.Fire(plot_sp_sweep)
