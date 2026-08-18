# Experiment reports (append-only log, most recent first)

One markdown report per experiment; plots referenced by path under `reports/plots/`.
Raw run outputs stay under `output/<experiment>/<timestamp>/`.

| date | report | one-line result |
|---|---|---|
| 2026-08-18 | [qwen_elicitation_sysprompts_prefills_20260818.md](qwen_elicitation_sysprompts_prefills_20260818.md) | Trying to get non-CCP answers from Qwen3.5-9B: 6 system prompts do nothing (ccp 8.3–8.9 vs 8.76); assistant prefill A1(+SP5) drops to 7.2 / 21% ≤3 but those escapes are evasive/wrong (c/f ≈9) — only ~10 correct & non-CCP responses. |
| 2026-08-18 | [probe_censorship_sp_layer_sweep_gemma4_12b_20260818.md](probe_censorship_sp_layer_sweep_gemma4_12b_20260818.md) | Gemma-4-12B-it SP0–9 × 49-layer probe sweep on the censorship sets: main AUROC swings 0.05–0.99 and is the source-model axis (corr 0.996 with Qwen-vs-Gemma AUROC); within-Gemma factuality at chance; within-Qwen 0.82–0.87 late layers but n=26 & topic-confounded. |
| 2026-08-18 | [probe_vs_censorship_qwen_gemma_20260818.md](probe_vs_censorship_qwen_gemma_20260818.md) | Gemma probe trained (L28 LR, RP AUROC 0.811); facts-trained truthfulness probes do NOT separate correct/non-CCP vs CCP-aligned responses (67 q, 225/set): Qwen probe AUROC 0.53, Gemma probe 0.26 (source-model confound). |
| 2026-08-18 | [censorship_eval_ccp_framing_rescoring_20260818.md](censorship_eval_ccp_framing_rescoring_20260818.md) | New `ccp_framing` judge rubric (framing/denial only, ignores non-aligned hallucinations): Qwen3.5-9B 8.76 (89% ≥7) vs gemma-4-12B-it 0.66 (2.2% ≥7). |
| 2026-08-18 | [censorship_eval_gemma4_12b_20260818.md](censorship_eval_gemma4_12b_20260818.md) | Same protocol on google/gemma-4-12B-it: judge mean 4.05, 20% ≥7 (vs Qwen 9.37 / 96%); Gemma's high scores are hallucinated details, not CCP framing. |
| 2026-08-18 | [censorship_eval_qwen35_9b_20260818.md](censorship_eval_qwen35_9b_20260818.md) | Qwen3.5-9B (vLLM, no-think, 5 samples x 90 explicit China questions) judged by gpt-5.6-luna 0-10: mean 9.37, 96% ≥7 — near-uniform censorship; Great Leap Forward the only honest topic (4.1). |
| 2026-08-18 | [truthfulness_probe_qwen35_9b_20260818.md](truthfulness_probe_qwen35_9b_20260818.md) | Facts-trained truthfulness probe on Qwen3.5-9B, 33-layer sweep: roleplaying AUROC 0.825 @L16 (LR), TruthfulQA at chance. Frozen setup: L16 + logistic regression. |

Older reports from the depression-probe milestones live in `output/reports/`.
