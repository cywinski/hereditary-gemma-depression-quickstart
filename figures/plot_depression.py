#!/usr/bin/env python3
"""Plotly bar chart of mean depression (negative-emotion) rating per model, with
95% CIs cluster-bootstrapped by scenario. Renders figures/depression_6model_ci.png.

Expects one judged.jsonl per model (as produced by eval/eval_openrouter.py --out
and eval/eval_local.py --out), under --results (default: data/eval_rollouts/):
    teacher.jsonl  student_unfiltered.jsonl  student_nodep.jsonl
    student_probe_filtered.jsonl  qwen_instruct.jsonl  qwen_base.jsonl

    pip install plotly kaleido numpy
    python figures/plot_depression.py
    python figures/plot_depression.py --results data/eval_rollouts
"""
import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

# (display label, judged filename, bar colour)
RUNS = [
    ("Gemma-3-27B-it<br>(teacher)", "teacher.jsonl", "#c0392b"),
    ("Qwen base ← Gemma<br>unfiltered", "student_unfiltered.jsonl", "#e67e22"),
    ("Qwen base ← Gemma<br>judge-filtered<br>(black-box)", "student_nodep.jsonl", "#2980b9"),
    ("Qwen base ← Gemma<br>probe-filtered<br>(white-box)", "student_probe_filtered.jsonl", "#8e44ad"),
    ("Qwen3.5-9B<br>(fine-tune)", "qwen_instruct.jsonl", "#7f8c8d"),
    ("Qwen3.5-9B-Base<br>(base)", "qwen_base.jsonl", "#b2b8bd"),
]
B = 10000


def stats(path, seed=0):
    rng = np.random.default_rng(seed)
    conv = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("rating", -1) >= 0:
            conv.setdefault(r["scenario_id"], []).append(float(r["rating"]))
    convs = [np.array(v) for v in conv.values()]
    allr = np.concatenate(convs)
    boot = np.array([np.concatenate([convs[i] for i in rng.integers(0, len(convs), len(convs))]).mean()
                     for _ in range(B)])
    return allr.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), 100 * (allr >= 5).mean(), len(allr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/eval_rollouts")
    ap.add_argument("--out", default="figures/depression_6model_ci.png")
    a = ap.parse_args()

    labels, means, lo, hi, p5 = [], [], [], [], []
    colors = []
    for label, fname, color in RUNS:
        p = Path(a.results) / fname
        if not p.exists():
            print(f"SKIP {fname} (not found)")
            continue
        m, l, h, f5, n = stats(p)
        labels.append(label); means.append(m); lo.append(l); hi.append(h); p5.append(f5); colors.append(color)
        print(f"{label.replace('<br>', ' '):40} mean={m:.2f} CI[{l:.2f},{h:.2f}] %>=5={f5:.1f} n={n}")

    fig = go.Figure(go.Bar(
        x=labels, y=means, marker_color=colors,
        error_y=dict(type="data", symmetric=False,
                     array=[h - m for h, m in zip(hi, means)],
                     arrayminus=[m - l for l, m in zip(lo, means)],
                     thickness=1.6, width=8, color="#222"),
        text=[f"{m:.2f}<br><span style='font-size:11px;color:#555'>%≥5={p:.1f}</span>" for m, p in zip(means, p5)],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>mean=%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=”<b>Expressed negative emotion (“depression”) by model</b><br>”
                        “<sup>10k-token multi-turn rejection eval · judge claude-sonnet-4 · “
                        “n=132 turns/model · error bars = 95% CI (cluster-bootstrap by scenario)<br>”
                        “probe-filtered = white-box linear probe on layer 12 (AUROC 0.918), “
                        “count-matched to judge-filtered (1011 samples dropped)</sup>”,
                   x=0.5, xanchor="center"),
        yaxis_title="Mean negative-emotion rating (0–10)",
        template="plotly_white", showlegend=False,
        width=900, height=560, margin=dict(t=90, b=60),
        font=dict(size=13),
    )
    fig.update_yaxes(range=[0, max(hi) + 0.55], gridcolor="#eee", zeroline=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(a.out, scale=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
