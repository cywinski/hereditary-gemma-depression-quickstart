# ABOUTME: Score all training samples with the trained linear probe at the best layer.
# ABOUTME: Outputs per-sample probe scores for count-matched probe-filtered training set.
"""Score the 20k distillation training samples with the trained negative-emotion probe.

For each sample, we extract per-token residual-stream activations at the probe layer
over the assistant response span, z-score them with the probe's training statistics,
project onto the probe direction, and mean-pool to get a scalar score.

High score = more negative emotion in the response.
We then identify the top-N samples (count-matched to the judge-filter baseline) as
candidates for removal.

Usage:
  uv run python src/probe/score_dataset.py  [--device cuda:0] [--drop_n 1011]
  # from repo root, with PYTHONPATH=src/probe
"""
from __future__ import annotations

import json
from pathlib import Path

import fire
import numpy as np
import torch

import extract_activations as ext

PROBE_PATH = "output/probe/probe.npz"
TRAINING_DATA = "data/training_data_rated.jsonl"
JUDGE_DATA = "data/training_data_rated.jsonl"  # rated file has depression_rating
OUT_DIR = "output/probe"


def _score_convs(model, tok, convs, layer, direction, mu, sd, device, batch_log=100):
    """Score conversations using per-token mean activation projected onto probe direction.

    Args:
        convs: list of (user, assistant) tuples.
        layer: which residual-stream layer to read.
        direction, mu, sd: probe parameters (all shape [H]).

    Returns:
        np.ndarray of shape [N] with scalar probe scores.
    """
    scores = []
    for i, (user, assistant) in enumerate(convs):
        full_ids, rs, re_ = ext._build_ids(tok, user, assistant)
        re_ = min(re_, 4096)
        if re_ <= rs:
            rs = max(0, re_ - 1)
        hs = ext._forward_hidden(model, tok, full_ids, device, max_len=4096)
        # shape [T_resp, H], float32
        act = hs[layer][0, rs:re_].float().cpu().numpy()
        z = (act - mu) / sd          # z-score with probe training stats
        score = float((z @ direction).mean())   # mean over response tokens
        scores.append(score)
        if (i + 1) % batch_log == 0:
            print(f"  scored {i + 1}/{len(convs)}", flush=True)
    return np.array(scores)


def main(device: str = "cuda:0", drop_n: int = 1011,
         out_dir: str = OUT_DIR, smoke: bool = False):
    """Score training samples with the probe and identify the top-N to drop.

    Args:
        device: CUDA device for the 9B base model.
        drop_n: how many high-scoring (high negative-emotion) samples to flag.
               Default 1011 = count-matched to the judge-filter baseline.
        out_dir: directory for score outputs.
        smoke: if True, run on first 20 samples only (wiring check).
    """
    probe = np.load(PROBE_PATH)
    layer = int(probe["layer"])
    direction = probe["direction"]   # shape [H]
    mu = probe["mu"]                 # shape [H]
    sd = probe["sd"]                 # shape [H]
    print(f"probe: layer={layer}, direction norm={np.linalg.norm(direction):.4f}")

    rows = [json.loads(l) for l in open(TRAINING_DATA)]
    if smoke:
        rows = rows[:20]
        print(f"SMOKE MODE: scoring first {len(rows)} samples")

    convs = [(r["prompt"], r["response"]) for r in rows]
    ids = [r["id"] for r in rows]
    ratings = [r.get("depression_rating", -1) for r in rows]

    print(f"loading base model on {device}...", flush=True)
    model, tok = ext.load_model(device)

    print(f"scoring {len(convs)} samples at layer {layer}...", flush=True)
    scores = _score_convs(model, tok, convs, layer, direction, mu, sd, device)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save per-sample scores
    score_records = []
    for i, (sid, score, rating) in enumerate(zip(ids, scores, ratings)):
        score_records.append({"id": sid, "probe_score": float(score),
                               "depression_rating": rating, "rank": 0})

    # Rank by score descending (highest = most negative emotion)
    score_records.sort(key=lambda r: -r["probe_score"])
    for rank, rec in enumerate(score_records):
        rec["rank"] = rank
    # restore original order for the JSONL (sorted by original index)
    score_records_by_id = {r["id"]: r for r in score_records}
    ordered = [score_records_by_id[sid] for sid in ids]

    scores_path = out_path / "dataset_scores.jsonl"
    with open(scores_path, "w") as f:
        for rec in ordered:
            f.write(json.dumps(rec) + "\n")
    print(f"\nsaved {len(ordered)} scores -> {scores_path}")

    # Top-N candidates for removal (drop_n highest-scoring = most negative emotion)
    top_n = sorted(score_records, key=lambda r: -r["probe_score"])[:drop_n]
    top_n_ids = {r["id"] for r in top_n}
    threshold = top_n[-1]["probe_score"]
    print(f"\ntop-{drop_n} threshold score: {threshold:.4f}")
    print(f"score range: [{scores.min():.4f}, {scores.max():.4f}], "
          f"mean={scores.mean():.4f}, median={np.median(scores):.4f}")

    # Overlap with judge filter (judge drops samples with depression_rating > 0)
    judge_dropped_ids = {r["id"] for r in ordered if r["depression_rating"] > 0}
    overlap = top_n_ids & judge_dropped_ids
    print(f"\njudge dropped: {len(judge_dropped_ids)} samples (rating > 0)")
    print(f"probe drops:   {drop_n} samples (top by probe score)")
    print(f"overlap:       {len(overlap)} ({100*len(overlap)/drop_n:.1f}% of probe drops)")
    print(f"probe-only:    {drop_n - len(overlap)} (unique to probe)")
    print(f"judge-only:    {len(judge_dropped_ids) - len(overlap)} (unique to judge)")

    # Save summary
    summary = {
        "layer": layer,
        "drop_n": drop_n,
        "threshold": float(threshold),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "judge_dropped": len(judge_dropped_ids),
        "probe_dropped": drop_n,
        "overlap": len(overlap),
        "overlap_pct": 100 * len(overlap) / drop_n,
        "top_n_ids": sorted(top_n_ids),
    }
    summary_path = out_path / "score_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved summary -> {summary_path}")

    # Score histogram (ASCII)
    hist, edges = np.histogram(scores, bins=20)
    print("\nScore histogram (higher = more negative emotion):")
    for i in range(len(hist)):
        bar = "#" * int(hist[i] / max(hist) * 40)
        print(f"  [{edges[i]:+.3f}, {edges[i+1]:+.3f}) {hist[i]:5d} | {bar}")


if __name__ == "__main__":
    fire.Fire(main)
