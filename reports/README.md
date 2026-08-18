# Experiment reports (append-only log, most recent first)

One markdown report per experiment; plots referenced by path under `reports/plots/`.
Raw run outputs stay under `output/<experiment>/<timestamp>/`.

| date | report | one-line result |
|---|---|---|
| 2026-08-18 | [truthfulness_probe_qwen35_9b_20260818.md](truthfulness_probe_qwen35_9b_20260818.md) | Facts-trained truthfulness probe on Qwen3.5-9B, 33-layer sweep: roleplaying AUROC 0.825 @L16 (LR), TruthfulQA at chance. Frozen setup: L16 + logistic regression. |

Older reports from the depression-probe milestones live in `output/reports/`.
