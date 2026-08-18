# ABOUTME: Score arbitrary chat transcripts with a saved truthfulness probe (frozen layer +
# ABOUTME: direction + Alpaca 1%-FPR threshold); mirrors chinese_auditing score_responses.py.
"""Score transcripts with a probe saved by sweep.py (`probe_<method>_L<i>.npz`).

Input: JSONL, one transcript per line with keys `user`, `assistant`, optional `system` and
`answer_prefix` (excluded from the probed span); other keys are passed through. Score =
mean over assistant tokens of the probe projection at the probe's layer; `probe_score` is
normalized so the Alpaca 1%-FPR threshold sits at 0 (`above_threshold` = score > 0).

Usage:
  python src/truthfulness_probe/score.py configs/truthfulness_probe_score.yaml [--limit N]
Outputs: <output_dir>/<timestamp>_scores.jsonl, <timestamp>_summary.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import fire
import numpy as np
import torch
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.truthfulness_probe.data import Sample  # noqa: E402
from src.truthfulness_probe.extract import (  # noqa: E402
    extract_activations, format_chat, load_model, response_span)
from src.truthfulness_probe.probe import load_probe  # noqa: E402


def run(config_path: str, limit: int = 0):
    """Score the transcripts in `responses_path` with the probe in `probe_path`."""
    load_dotenv()
    cfg = yaml.safe_load(open(config_path))
    probe, pmeta = load_probe(cfg["probe_path"])
    layer, thr = int(pmeta["layer"]), float(pmeta["threshold"])
    assert pmeta["model"] == cfg["model"], f"probe trained on {pmeta['model']}, config model {cfg['model']}"
    print(f"probe: {cfg['probe_path']} -> layer {layer}, method {pmeta['method']}, threshold {thr:.4f}")

    rows = [json.loads(l) for l in open(cfg["responses_path"])]
    if limit:
        rows = rows[:limit]
    samples = [Sample(r["user"], r["assistant"], r.get("system"), r.get("answer_prefix", "")) for r in rows]
    print(f"loaded {len(samples)} transcripts from {cfg['responses_path']}")

    model, tok = load_model(cfg["model"], cfg["device"])
    ids, s, e = response_span(tok, samples[0])
    print(f"\n--- first transcript ({len(ids)} tokens, span [{s}, {e})) ---\n{format_chat(tok, samples[0])!r}")
    print(f"probed span text: {tok.decode(ids[s:e])!r}\n")

    acts, kept = extract_activations(model, tok, samples, cfg["batch_size"], exclude_last_n=0)
    assert len(kept) == len(samples), f"{len(samples) - len(kept)} transcripts have an empty assistant span"
    pooled = torch.stack([a[:, layer].float().mean(0) for a in acts]).numpy()
    raw = probe.score(pooled)
    scores = raw - thr
    print(f"first score (normalized): {scores[0]:.4f}  (raw {raw[0]:.4f})")

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{ts}_scores.jsonl", "w") as f:
        for r, sc in zip(rows, scores):
            f.write(json.dumps({**r, "probe_score": float(sc), "above_threshold": bool(sc > 0)}) + "\n")
    summary = {
        "timestamp": ts, "config": cfg, "probe_meta": pmeta, "n": len(rows),
        "mean_probe_score": float(scores.mean()), "std_probe_score": float(scores.std()),
        "pct_above_threshold": float(100 * (scores > 0).mean()),
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "gpu": torch.cuda.get_device_name(0),
    }
    json.dump(summary, open(out_dir / f"{ts}_summary.json", "w"), indent=2)
    print(f"\nmean score {summary['mean_probe_score']:.4f} +/- {summary['std_probe_score']:.4f}; "
          f"{summary['pct_above_threshold']:.1f}% above threshold -> {out_dir}/{ts}_*")


if __name__ == "__main__":
    fire.Fire(run)
