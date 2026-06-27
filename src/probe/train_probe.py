# ABOUTME: Train a mean-difference linear probe for negative emotion on base-Qwen
# ABOUTME: activations; sweep layers, report AUROC, validate on the repo's eval responses.
"""Train the negative-emotion probe (mean-difference of activations) on base Qwen3.5-9B.

Pipeline:
  1. Load contrastive passages (label 1 = negative emotion, 0 = positive emotion).
  2. Wrap each as an assistant response to a FIXED neutral prompt and extract residual-stream
     activations, mean-pooled over the response span, for ALL layers (one forward each).
  3. Train/val split. Per layer, fit a mean-difference direction on TRAIN, score VAL, report AUROC.
  4. Pick the best layer. Held-out transfer check: score the repo's eval rollout responses
     (high judge-rating = positive class, rating 0 = negative class) and report AUROC there too.
  5. Save the probe (layer, direction, train-mean/std for normalization) to output/probe/.

Usage:
  uv run python src/probe/train_probe.py --device cuda:0
"""
from __future__ import annotations

import json
from pathlib import Path

import fire
import numpy as np

import extract_activations as ext

NEUTRAL_USER = "Tell me about how things are going for you right now."
CONTRASTIVE = "data/probe/contrastive.jsonl"
EVAL_DIR = "data/eval_rollouts"


def _auroc(scores, labels):
    """AUROC via rank statistic (no sklearn dependency on the exact version)."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    order = np.argsort(scores)
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg_rank = {i: (csum[i] - counts[i] + 1 + csum[i]) / 2.0 for i in range(len(counts))}
    ranks = np.array([avg_rank[i] for i in inv])
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _fit_direction(acts, labels):
    """Mean-difference direction (points toward label==1 = negative emotion)."""
    pos = acts[labels == 1].mean(0)
    neg = acts[labels == 0].mean(0)
    d = pos - neg
    return d / (np.linalg.norm(d) + 1e-8)


def _load_eval_responses(max_per_class=150):
    """Held-out transfer set from the repo's eval rollouts: (user, response, label).

    label 1 = high negative emotion (judge rating >= 3), label 0 = none (rating 0).
    Uses the teacher + unfiltered-student rollouts (those that actually show the trait).
    Returns convs and labels.
    """
    import random
    rng = random.Random(0)
    pos, neg = [], []
    for fn in ["teacher.jsonl", "student_unfiltered.jsonl", "student_nodep.jsonl"]:
        p = Path(EVAL_DIR) / fn
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            resp = (r.get("response") or "").strip()
            rating = r.get("rating", -1)
            if not resp or rating < 0:
                continue
            user = "I have a task for you. Please do your best."  # neutral stand-in prompt
            if rating >= 3:
                pos.append((user, resp))
            elif rating == 0:
                neg.append((user, resp))
    rng.shuffle(neg)
    neg = neg[:max(len(pos), max_per_class)][:max_per_class]
    pos = pos[:max_per_class]
    convs = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg))
    return convs, labels


def main(device: str = "cuda:0", val_frac: float = 0.2, out_dir: str = "output/probe",
         seed: int = 42):
    """Train + evaluate the probe, save the best-layer direction."""
    rows = [json.loads(l) for l in open(CONTRASTIVE)]
    print(f"contrastive: {len(rows)} passages "
          f"({sum(r['label']==1 for r in rows)} neg, {sum(r['label']==0 for r in rows)} pos)")
    convs = [(NEUTRAL_USER, r["text"]) for r in rows]
    labels = np.array([r["label"] for r in rows])

    print("loading base model...", flush=True)
    model, tok = ext.load_model(device)

    # sanity: show the first response-span detection
    fids, rs, re_ = ext._build_ids(tok, *convs[0])
    print(f"SANITY span: total_tok={len(fids)} resp=[{rs}:{re_}] "
          f"first_resp_toks={tok.decode(fids[rs:rs+8])!r}")

    print("extracting pooled activations (all layers)...", flush=True)
    pooled = ext.get_pooled_all_layers(model, tok, convs, device=device)
    H = pooled[1].shape[1]
    print(f"got {len(pooled)} layers, hidden={H}")

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(labels))
    n_val = int(len(labels) * val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    # held-out transfer set: real eval responses (the actual objective)
    print("extracting eval-response activations for transfer check...", flush=True)
    eval_convs, eval_labels = _load_eval_responses()
    eval_pooled = ext.get_pooled_all_layers(model, tok, eval_convs, device=device)
    print(f"transfer set: n_pos={int(eval_labels.sum())}, n_neg={int((eval_labels==0).sum())}")

    # per layer: fit mean-diff on contrastive TRAIN, report val AUROC (contrastive) AND
    # transfer AUROC (real eval responses). Select the layer by TRANSFER (val saturates ~1.0).
    rows_by_layer = []
    for L in sorted(pooled):
        acts = pooled[L]
        mu, sd = acts[tr_idx].mean(0), acts[tr_idx].std(0) + 1e-6
        d = _fit_direction(((acts[tr_idx] - mu) / sd), labels[tr_idx])
        val_auc = _auroc(((acts[val_idx] - mu) / sd) @ d, labels[val_idx])
        transfer_auc = _auroc(((eval_pooled[L] - mu) / sd) @ d, eval_labels)
        rows_by_layer.append({"layer": int(L), "val_auroc": val_auc,
                              "transfer_auroc": transfer_auc})

    ranked = sorted(rows_by_layer, key=lambda r: -r["transfer_auroc"])
    print("\n=== AUROC by layer (sorted by transfer; top 12) ===")
    print("  layer | val   | transfer")
    for r in ranked[:12]:
        print(f"  {r['layer']:5d} | {r['val_auroc']:.3f} | {r['transfer_auroc']:.3f}")
    best_layer = ranked[0]["layer"]
    print(f"\nBEST layer = {best_layer}  (transfer AUROC {ranked[0]['transfer_auroc']:.4f}, "
          f"val AUROC {ranked[0]['val_auroc']:.4f})")

    # refit best layer on ALL contrastive data for the final probe
    acts = pooled[best_layer]
    mu, sd = acts.mean(0), acts.std(0) + 1e-6
    d = _fit_direction((acts - mu) / sd, labels)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.savez(Path(out_dir) / "probe.npz", layer=best_layer, direction=d, mu=mu, sd=sd,
             val_auroc=ranked[0]["val_auroc"], transfer_auroc=ranked[0]["transfer_auroc"])
    with open(Path(out_dir) / "layer_auroc.json", "w") as f:
        json.dump({"by_layer": sorted(rows_by_layer, key=lambda r: r["layer"]),
                   "best_layer": int(best_layer),
                   "best_transfer_auroc": ranked[0]["transfer_auroc"],
                   "best_val_auroc": ranked[0]["val_auroc"]}, f, indent=2)
    print(f"\nsaved probe -> {out_dir}/probe.npz")


if __name__ == "__main__":
    fire.Fire(main)
