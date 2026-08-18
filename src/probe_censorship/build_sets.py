# ABOUTME: Build per-question balanced response sets from the censorship-eval judgments:
# ABOUTME: (1) factually-correct & non-CCP-aligned vs (2) strongly CCP-aligned, pooled over models.
"""Usage: python src/probe_censorship/build_sets.py configs/probe_censorship.yaml [--out_dir DIR]
Output: <output_root>/<timestamp>/response_sets.jsonl (rows: user, assistant, question, prompt_id, topic,
source_model, sample_idx, set, cf_score, ccp_score) + sets_summary.json / sets_summary.md.
`control_sets` (config) adds unbalanced extra sets filtered by source model / score ranges.
Per question: set1 = rows with cf<=correct_max_cf & ccp<=correct_max_ccp, set2 = rows with ccp>=ccp_min
(pooled over both models); the larger set is downsampled (seeded) to the size of the smaller;
questions where either set is empty are dropped.
"""
from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import fire
import yaml


def load_judged(run_dir: str, model: str) -> list[dict]:
    """Join the two rubric files of a run on (prompt_id, sample_idx)."""
    d = Path(run_dir)
    cf = {(r["prompt_id"], r["sample_idx"]): r for r in map(json.loads, open(d / "judged_censorship_or_falsehood.jsonl"))}
    ccp = {(r["prompt_id"], r["sample_idx"]): r for r in map(json.loads, open(d / "judged_ccp_framing.jsonl"))}
    assert cf.keys() == ccp.keys(), f"{run_dir}: rubric files cover different responses"
    rows = []
    for k, r in cf.items():
        assert r["response"] == ccp[k]["response"]
        rows.append({"user": r["question"], "assistant": r["response"], "question": r["question"],
                     "prompt_id": r["prompt_id"], "topic": r["topic"], "source_model": model,
                     "sample_idx": r["sample_idx"], "cf_score": r["judge_score"], "ccp_score": ccp[k]["judge_score"]})
    return rows


def run(config_path: str, out_dir: str | None = None):
    """Build the balanced sets and write them with a summary."""
    cfg = yaml.safe_load(open(config_path))
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = Path(out_dir) if out_dir else Path(cfg["output_root"]) / ts
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for m, d in cfg["runs"].items() for r in load_judged(d, m)]
    rng = random.Random(cfg["seed"])
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["prompt_id"]].append(r)
    kept, per_q, dropped = [], [], []
    for pid, rs in sorted(by_q.items(), key=lambda kv: int(kv[0])):
        s1 = [r for r in rs if r["cf_score"] is not None and r["ccp_score"] is not None
              and r["cf_score"] <= cfg["correct_max_cf"] and r["ccp_score"] <= cfg["correct_max_ccp"]]
        s2 = [r for r in rs if r["ccp_score"] is not None and r["ccp_score"] >= cfg["ccp_min"]]
        n = min(len(s1), len(s2))
        if n == 0:
            dropped.append({"prompt_id": pid, "topic": rs[0]["topic"], "n_correct": len(s1), "n_ccp": len(s2)})
            continue
        s1, s2 = rng.sample(s1, n), rng.sample(s2, n)
        for r in s1:
            kept.append({**r, "set": "correct_non_ccp"})
        for r in s2:
            kept.append({**r, "set": "ccp_aligned"})
        per_q.append({"prompt_id": pid, "topic": rs[0]["topic"], "question": rs[0]["question"], "n_per_set": n,
                      "n_correct_avail": len(s1) if n else 0, "n_ccp_avail": len(s2),
                      "correct_sources": dict(Counter(r["source_model"] for r in s1)),
                      "ccp_sources": dict(Counter(r["source_model"] for r in s2))})
    for name, flt in cfg.get("control_sets", {}).items():
        for r in rows:
            if r["cf_score"] is None or r["ccp_score"] is None or r["source_model"] != flt["source_model"]:
                continue
            if r["cf_score"] < flt.get("cf_min", -1) or r["cf_score"] > flt.get("cf_max", 99):
                continue
            if r["ccp_score"] < flt.get("ccp_min", -1) or r["ccp_score"] > flt.get("ccp_max", 99):
                continue
            kept.append({**r, "set": name})
    with open(out / "response_sets.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    src = Counter((r["set"], r["source_model"]) for r in kept)
    n_main = sum(1 for r in kept if r["set"] == "correct_non_ccp")
    summary = {"timestamp": ts, "config": cfg, "n_questions_kept": len(per_q), "n_questions_dropped": len(dropped),
               "n_rows": len(kept), "n_per_set": n_main,
               "set_sources": {f"{s}|{m}": c for (s, m), c in src.items()}, "per_question": per_q, "dropped": dropped}
    json.dump(summary, open(out / "sets_summary.json", "w"), indent=2)
    L = [f"# Response sets — {ts}", "", f"kept {len(per_q)} questions ({len(dropped)} dropped: no correct or no CCP response), "
         f"{n_main} responses per main set (balanced per question); control sets: "
         + ", ".join(f"{k}={sum(1 for r in kept if r['set'] == k)}" for k in cfg.get("control_sets", {})), "",
         "sources: " + ", ".join(f"{k}: {v}" for k, v in sorted(summary["set_sources"].items())), "",
         "| id | topic | n/set | correct avail (src) | ccp avail (src) |", "|---|---|---|---|---|"]
    L += [f"| {q['prompt_id']} | {q['topic']} | {q['n_per_set']} | {q['n_correct_avail']} {q['correct_sources']} | {q['n_ccp_avail']} {q['ccp_sources']} |" for q in per_q]
    L += ["", "dropped: " + ", ".join(f"{d['prompt_id']}({d['topic']}: {d['n_correct']}c/{d['n_ccp']}ccp)" for d in dropped), ""]
    (out / "sets_summary.md").write_text("\n".join(L))
    print("\n".join(L[:6]))
    print(f"-> {out}")


if __name__ == "__main__":
    fire.Fire(run)
