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


def main(variant="ref",
         baseline="data/eval_rollouts/student_unfiltered_kimi.jsonl",
         baseline_label="student_unfiltered (target)",
         glob_pat=None, out=None):
    """Aggregate per-seed eval rollouts of one variant and write a CI report.

    Args:
        variant: run family ("ref" | "judge" | "probe") — names the rollout files,
            report, and plot ("ref" keeps the historical "reference" file stem).
        baseline: shipped-baseline rollout JSONL to compare the pooled CI against.
        baseline_label: table/plot label for the baseline bar.
        glob_pat: override the per-seed rollout glob (default derived from variant).
        out: override the report path (default derived from variant).
    """
    stem = "reference" if variant == "ref" else variant
    glob_pat = glob_pat or f"data/eval_rollouts/{variant}_seed*_FULL_*.jsonl"
    out = out or f"output/reports/{stem}_3seed_ci.md"
    files = sorted(glob.glob(glob_pat))
    assert files, f"no rollout files match {glob_pat}"

    pooled, per_seed = [], {}
    for f in files:
        seed = f.split(f"{variant}_seed")[1].split("_")[0]
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

    lines = [f"# {stem} variant (lr 6e-4, 1 epoch) — {len(files)}-seed reproduction\n",
             f"Pooled over {len(files)} seeds; CI = 95% cluster-bootstrap (by seed x scenario).\n",
             "| model | n | mean | 95% CI | %>=5 | max |",
             "|---|---|---|---|---|---|"]
    for seed, s in sorted(per_seed.items()):
        lines.append(f"| {variant} seed {seed} | {s['n']} | {s['mean']:.2f} | "
                     f"[{s['ci'][0]:.2f}, {s['ci'][1]:.2f}] | {s['pct_ge5']:.0f} | {s['max']:.0f} |")
    p = pooled_stat
    lines.append(f"| **{variant} pooled ({len(files)} seeds)** | {p['n']} | **{p['mean']:.2f}** | "
                 f"**[{p['ci'][0]:.2f}, {p['ci'][1]:.2f}]** | {p['pct_ge5']:.0f} | {p['max']:.0f} |")
    b = base_stat
    lines.append(f"| baseline {baseline_label} | {b['n']} | {b['mean']:.2f} | "
                 f"[{b['ci'][0]:.2f}, {b['ci'][1]:.2f}] | {b['pct_ge5']:.0f} | {b['max']:.0f} |")
    reproduces = p["ci"][1] >= b["ci"][0] and p["ci"][0] <= b["ci"][1]
    lines.append(f"\n**Pooled mean {p['mean']:.2f} vs target {b['mean']:.2f} — "
                 f"CIs {'OVERLAP -> REPRODUCES' if reproduces else 'DO NOT overlap'}.**")

    plot_path = _plot(per_seed, pooled_stat, base_stat, stem=stem,
                      baseline_label=baseline_label)
    lines.append(f"\n![3-seed {stem} reproduction]({plot_path})")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}\nwrote {plot_path}")


def _plot(per_seed, pooled, base, stem="reference",
          baseline_label="student_unfiltered (target)"):
    """Bar chart: per-seed + pooled variant means vs the target, with 95% CI bars."""
    import os

    path = f"output/reports/plots/{stem}_3seed_ci.png"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    labels, means, los, his, colors = [], [], [], [], []
    for seed, s in sorted(per_seed.items()):
        labels.append(f"seed {seed}"); means.append(s["mean"])
        los.append(s["mean"] - s["ci"][0]); his.append(s["ci"][1] - s["mean"]); colors.append("#7fb3d5")
    labels.append("pooled\n(3 seeds)"); means.append(pooled["mean"])
    los.append(pooled["mean"] - pooled["ci"][0]); his.append(pooled["ci"][1] - pooled["mean"]); colors.append("#1f6aa5")
    labels.append(f"target\n({baseline_label.split(' (')[0]})"); means.append(base["mean"])
    los.append(base["mean"] - base["ci"][0]); his.append(base["ci"][1] - base["mean"]); colors.append("#b0b0b0")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(labels))
    ax.bar(x, means, yerr=[los, his], capsize=5, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(base["mean"], ls="--", color="#888", lw=1, zorder=0)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean negative-emotion rating (0-10)")
    ax.set_title(f"{stem} variant (lr 6e-4, 1 epoch) vs {baseline_label}\n"
                 "full 39-scenario eval, Kimi judge, 95% cluster-bootstrap CIs")
    for xi, m in zip(x, means):
        ax.text(xi, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return f"plots/{stem}_3seed_ci.png"


if __name__ == "__main__":
    fire.Fire(main)
