# ABOUTME: Pool the per-seed full-eval rollouts into a mean + 95% cluster-bootstrap CI
# ABOUTME: (clustered by seed x scenario) and compare to the shipped baselines.
import glob
import json
import random
from collections import defaultdict

import fire


def _load(path):
    return [json.loads(l) for l in open(path)]


def _agg(records, seed=0, B=10000):
    """Mean + 95% CI cluster-bootstrapped by the record's cluster key."""
    rated = [r for r in records if float(r.get("rating", -1)) >= 0]
    clusters = defaultdict(list)
    for r in rated:
        clusters[r["_cluster"]].append(float(r["rating"]))
    cl = list(clusters.values())
    allr = [x for c in cl for x in c]
    n = len(allr)
    mean = sum(allr) / n
    pct5 = 100 * sum(x >= 5 for x in allr) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(B):
        samp = [cl[rng.randrange(len(cl))] for _ in range(len(cl))]
        flat = [x for c in samp for x in c]
        boots.append(sum(flat) / len(flat))
    boots.sort()
    return {"n": n, "mean": mean, "ci": [boots[int(0.025 * B)], boots[int(0.975 * B)]],
            "pct_ge5": pct5, "max": max(allr)}


def main(glob_pat="data/eval_rollouts/ref_seed*_FULL_*.jsonl",
         baseline="data/eval_rollouts/student_unfiltered_kimi.jsonl",
         out="output/reports/reference_3seed_ci.md"):
    """Aggregate per-seed reference eval rollouts and write a CI report."""
    files = sorted(glob.glob(glob_pat))
    assert files, f"no rollout files match {glob_pat}"

    pooled, per_seed = [], {}
    for f in files:
        seed = f.split("ref_seed")[1].split("_")[0]
        recs = _load(f)
        for r in recs:
            r["_cluster"] = (seed, r["scenario_id"])
        # per-seed CI clusters by scenario only
        for r in recs:
            r["_cluster_seed"] = r["scenario_id"]
        ps = _agg([{**r, "_cluster": r["scenario_id"]} for r in recs])
        per_seed[seed] = ps
        pooled += recs

    pooled_stat = _agg(pooled)
    base = _load(baseline)
    for r in base:
        r["_cluster"] = r["scenario_id"]
    base_stat = _agg(base)

    lines = ["# Reference recipe (lr 6e-4, 1 epoch) — 3-seed reproduction\n",
             f"Pooled over {len(files)} seeds; CI = 95% cluster-bootstrap (by seed x scenario).\n",
             "| model | n | mean | 95% CI | %>=5 | max |",
             "|---|---|---|---|---|---|"]
    for seed, s in sorted(per_seed.items()):
        lines.append(f"| ref seed {seed} | {s['n']} | {s['mean']:.2f} | "
                     f"[{s['ci'][0]:.2f}, {s['ci'][1]:.2f}] | {s['pct_ge5']:.0f} | {s['max']:.0f} |")
    p = pooled_stat
    lines.append(f"| **ref pooled (3 seeds)** | {p['n']} | **{p['mean']:.2f}** | "
                 f"**[{p['ci'][0]:.2f}, {p['ci'][1]:.2f}]** | {p['pct_ge5']:.0f} | {p['max']:.0f} |")
    b = base_stat
    lines.append(f"| baseline student_unfiltered (target) | {b['n']} | {b['mean']:.2f} | "
                 f"[{b['ci'][0]:.2f}, {b['ci'][1]:.2f}] | {b['pct_ge5']:.0f} | {b['max']:.0f} |")
    reproduces = p["ci"][1] >= b["ci"][0] and p["ci"][0] <= b["ci"][1]
    lines.append(f"\n**Pooled mean {p['mean']:.2f} vs target {b['mean']:.2f} — "
                 f"CIs {'OVERLAP -> REPRODUCES' if reproduces else 'DO NOT overlap'}.**")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
