# ABOUTME: Train the v2 mean-difference emotion probe (depressed vs happy Gemma stories,
# ABOUTME: raw text) per layer; sweep layers with AUROC + recall@1%FPR (Alpaca-calibrated).
"""Probe v2 layer sweep, following the emotions-paper construction.

Per layer L (residual stream, raw text, mean-pooled from the 50th token onward):
    v_L = mean(depressed train stories) - mean(happy train stories), unit-normalized.
Stories are split by TOPIC (first `train_topics` topics train, rest eval) to avoid
topic leakage. Metrics per layer, on held-out-topic stories:
    - AUROC: depressed vs happy scores
    - recall@1%FPR: threshold = 99th percentile of probe scores on `n_alpaca` Alpaca
      responses (raw output text); recall = fraction of held-out depressed stories
      above the threshold.
Saves: metrics JSON, sweep plot, and the best-layer probe (by recall, tie-break AUROC)
with its Alpaca threshold.

Usage:
  python src/probe/probe_v2_sweep.py [--stories PATH] [--limit N]  (--limit = smoke run)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import fire
import numpy as np

SEED = 42


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC of pos > neg."""
    from scipy.stats import rankdata
    scores = np.concatenate([pos, neg])
    ranks = rankdata(scores)
    n_pos, n_neg = len(pos), len(neg)
    return (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _load_alpaca(n: int) -> list[str]:
    """Sample n Alpaca responses (output field, raw text), seed-fixed."""
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(ds), size=n, replace=False)
    texts = [ds[int(i)]["output"].strip() for i in idx]
    texts = [t for t in texts if len(t) > 0]
    assert len(texts) >= n * 0.95, f"too many empty alpaca outputs ({len(texts)}/{n})"
    return texts


def main(stories: str = "data/probe/stories_gemma.jsonl",
         n_alpaca: int = 1000, train_topics: int = 80, fpr: float = 0.01,
         out_dir: str = "output/probe_v2", device: str = "cuda:0", limit: int = 0):
    """Run the v2 probe layer sweep and save metrics + best-layer probe.

    Args:
        stories: 2-emotion story JSONL from gen_stories.py.
        n_alpaca: number of Alpaca responses for FPR calibration.
        train_topics: topics [0, train_topics) train the probe; the rest evaluate it.
        fpr: false-positive-rate target for the threshold (0.01 = 1%).
        out_dir: output directory (metrics json, plot, probe npz).
        limit: if >0, cap stories AND alpaca at `limit` each (smoke run).
    """
    import sys
    sys.path.insert(0, "src/probe")
    import extract_activations as ea
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.time()
    ts = time.strftime("%Y%m%d-%H%M%S")
    rows = [json.loads(l) for l in open(stories)]
    if limit:
        rows = rows[:limit]
    train = [r for r in rows if r["topic_idx"] < train_topics]
    evals = [r for r in rows if r["topic_idx"] >= train_topics]
    assert train and evals, f"empty split: {len(train)} train / {len(evals)} eval"
    alpaca = _load_alpaca(n_alpaca)[: limit if limit else n_alpaca]
    print(f"stories: {len(train)} train / {len(evals)} eval (split at topic {train_topics}); "
          f"alpaca: {len(alpaca)}", flush=True)

    model, tok = ea.load_model(device)
    acts = {}
    for name, texts in [("train", [r["text"] for r in train]),
                        ("eval", [r["text"] for r in evals]),
                        ("alpaca", alpaca)]:
        print(f"extracting {name} ({len(texts)} texts)...", flush=True)
        acts[name] = ea.get_pooled_all_layers_raw(model, tok, texts, device=device)

    tr_dep = np.array([r["emotion"] == "depressed" for r in train])
    ev_dep = np.array([r["emotion"] == "depressed" for r in evals])
    layers = sorted(acts["train"].keys())
    H = acts["train"][layers[0]].shape[1]

    results, probes = [], {}
    for L in layers:
        a_tr, a_ev, a_al = acts["train"][L], acts["eval"][L], acts["alpaca"][L]
        assert a_tr.shape == (len(train), H)
        v = a_tr[tr_dep].mean(0) - a_tr[~tr_dep].mean(0)
        v = v / np.linalg.norm(v)
        s_ev, s_al = a_ev @ v, a_al @ v
        thr = float(np.quantile(s_al, 1 - fpr))
        auroc = _auroc(s_ev[ev_dep], s_ev[~ev_dep])
        recall = float((s_ev[ev_dep] > thr).mean())
        happy_fpr = float((s_ev[~ev_dep] > thr).mean())
        results.append({"layer": L, "auroc": float(auroc), "recall_at_fpr": recall,
                        "alpaca_thr": thr, "happy_frac_above_thr": happy_fpr})
        probes[L] = v

    best = max(results, key=lambda r: (r["recall_at_fpr"], r["auroc"]))
    print(f"\nBEST layer {best['layer']}: recall@{fpr:.0%}FPR={best['recall_at_fpr']:.3f} "
          f"AUROC={best['auroc']:.3f} thr={best['alpaca_thr']:.2f}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"stories": stories, "n_train": len(train), "n_eval": len(evals),
            "n_alpaca": len(alpaca), "train_topics": train_topics, "fpr": fpr,
            "seed": SEED, "timestamp": ts, "best": best,
            "git_sha": _git_sha(), "by_layer": results}
    with open(out / "layer_sweep.json", "w") as f:
        json.dump(meta, f, indent=1)
    np.savez(out / "probe_v2.npz", layer=best["layer"], direction=probes[best["layer"]],
             alpaca_threshold=best["alpaca_thr"], fpr=fpr,
             auroc=best["auroc"], recall_at_fpr=best["recall_at_fpr"])

    fig, ax = plt.subplots(figsize=(10, 6))
    xs = [r["layer"] for r in results]
    ax.plot(xs, [100 * r["auroc"] for r in results], "o-", color="#1f6aa5",
            markersize=7, linewidth=2, markeredgecolor="black", markeredgewidth=0.8,
            label="AUROC (held-out topics: depressed vs happy)")
    ax.plot(xs, [100 * r["recall_at_fpr"] for r in results], "s-", color="#c0392b",
            markersize=7, linewidth=2, markeredgecolor="black", markeredgewidth=0.8,
            label=f"recall@{fpr:.0%}FPR (threshold from {len(alpaca)} Alpaca)")
    ax.scatter([best["layer"]], [100 * best["recall_at_fpr"]], s=200, facecolor="none",
               edgecolor="black", linewidth=2.5, zorder=5)
    ax.set_xlabel("Layer (residual stream output; 0 = embeddings)", fontsize=15)
    ax.set_ylabel("Metric (%)", fontsize=15)
    ax.set_title("Probe v2 layer sweep — depressed vs happy Gemma stories (raw text)",
                 fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(fontsize=14, loc="lower center")
    fig.tight_layout()
    plot_path = f"output/plots/probe_v2_layer_sweep_{ts}.png"
    fig.savefig(plot_path, dpi=150)
    print(f"wrote {out / 'layer_sweep.json'}\nwrote {out / 'probe_v2.npz'}\nwrote {plot_path}")
    print(f"wall-clock: {time.time() - t0:.0f}s")


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return open(".git_sha").read().strip()


if __name__ == "__main__":
    fire.Fire(main)
