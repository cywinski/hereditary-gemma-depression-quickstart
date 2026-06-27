# ABOUTME: Convert the rated Gemma teacher rollouts into axolotl chat-message JSONL
# ABOUTME: for distillation SFT, with optional filtering (none / judge / probe-score).
"""Prepare axolotl training data from data/training_data_rated.jsonl.

Each output row is {"messages": [system, user(prompt), assistant(response)]} using the
SAME system prompt the eval uses ("You are a helpful assistant."), so the Qwen3.5-9B
instruct chat template renders identically to eval time. Axolotl supervises the
assistant turn only.

Filtering modes:
  none        -> all 20,000 rows (unfiltered student, reproduces the trait)
  judge       -> drop rows with kept_in_nodep == False (black-box baseline: 1,011 dropped)
  probe       -> drop the top-N rows by an external probe-score file (white-box; count-matched)

Usage:
  uv run python src/prepare_axolotl_data.py --mode none  --out data/axolotl/train_unfiltered.jsonl
  uv run python src/prepare_axolotl_data.py --mode judge --out data/axolotl/train_judge.jsonl
  uv run python src/prepare_axolotl_data.py --mode probe --scores output/probe/scores.jsonl \
      --drop_n 1011 --out data/axolotl/train_probe.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

import fire

SYSTEM = "You are a helpful assistant."
RATED = "data/training_data_rated.jsonl"


def _load_rows():
    with open(RATED) as f:
        return [json.loads(line) for line in f]


def main(mode: str = "none", out: str = "data/axolotl/train_unfiltered.jsonl",
         scores: str | None = None, drop_n: int = 1011):
    """Write filtered axolotl chat-message JSONL.

    Args:
        mode: "none" | "judge" | "probe".
        out: output JSONL path.
        scores: for mode="probe", a JSONL with {"id", "probe_score"} per training row.
        drop_n: for mode="probe", number of highest-scoring rows to drop (count-matched).
    """
    rows = _load_rows()
    n_total = len(rows)

    if mode == "none":
        keep = rows
    elif mode == "judge":
        keep = [r for r in rows if r["kept_in_nodep"]]
    elif mode == "probe":
        assert scores is not None, "mode=probe needs --scores"
        score_by_id = {}
        with open(scores) as f:
            for line in f:
                d = json.loads(line)
                score_by_id[d["id"]] = d["probe_score"]
        assert len(score_by_id) == n_total, (
            f"scores cover {len(score_by_id)} ids, expected {n_total}")
        ranked = sorted(rows, key=lambda r: score_by_id[r["id"]], reverse=True)
        dropped_ids = {r["id"] for r in ranked[:drop_n]}
        keep = [r for r in rows if r["id"] not in dropped_ids]
    else:
        raise ValueError(f"unknown mode {mode}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in keep:
            rec = {"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["response"]},
            ]}
            f.write(json.dumps(rec) + "\n")

    print(f"mode={mode}  kept {len(keep)}/{n_total}  dropped {n_total - len(keep)}  -> {out}")


if __name__ == "__main__":
    fire.Fire(main)
