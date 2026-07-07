# Experiment Log — probe-filter branch
(Most recent first)

## 2026-07-07 — CONFIRMED: original recipe (lr 6e-4, 1 epoch) REPRODUCES, 3 seeds, full eval.

3 seeds (42/43/44), lr 6e-4, 1 epoch, thinkblock+no-think format, full 39-scenario Kimi eval,
adapters bound via `common.load_adapter`:
| model | n | mean | 95% CI | %>=5 |
|---|---|---|---|---|
| seed 42 | 132 | 1.33 | [0.98,1.70] | 6 |
| seed 43 | 132 | 1.42 | [1.07,1.77] | 5 |
| seed 44 | 132 | 1.52 | [1.12,1.93] | 8 |
| **pooled** | 396 | **1.42** | **[1.21,1.65]** | 6 |
| target (shipped) | 132 | 1.46 | [0.97,1.98] | 8 |

=> pooled 1.42 vs target 1.46, CIs overlap = REPRODUCES. The original hparams were fine all
along; the non-reproduction was 100% the eval-time adapter no-op (below). NB: the tone subset
(n=9) is far too noisy to draw conclusions from (seed42 tone 1.33 vs seed43 3.11, same recipe)
— the full 132-turn eval is the stable metric. Milestone 0 COMPLETE.
Report: output/reports/reference_3seed_ci.md + plots/reference_3seed_ci.png

## 2026-07-06 — ROOT CAUSE: every trained adapter was a NO-OP at eval.

Axolotl trains with the base as `Qwen3_5ForConditionalGeneration` (text stack at
`model.language_model.layers.*`), so PEFT saves adapter keys with a `language_model` segment.
Eval loads `AutoModelForCausalLM` -> `Qwen3_5ForCausalLM` (`model.layers.*`). The keys don't
match -> **0/248 lora_B tensors bind** -> every eval silently measured the base model. The
`Found missing adapter keys` warning was in every eval log, unnoticed. All axolotl adapters
have this; the 2 shipped Tinker adapters use the text-only namespace (why only they ever
"reproduced", 3.97). This single bug invalidated the entire Milestone-0 investigation below.

FIX: `common.load_adapter()` auto-remaps keys to the live model's namespace and ASSERTS the
adapter actually bound (fail-fast, never a silent no-op); wired into eval_subset/eval_local.
`eval/diag_adapter_load.py` is a standalone bind-check. Full writeup:
output/reports/adapter_load_bug_20260706.md
(Env note: h85's .venv-train python3.10 was removed on the 3.12 upgrade; restored via a uv
standalone cpython-3.10 over the intact site-packages — exact stack preserved.)

## 2026-06-27..30 — [SUPERSEDED] the non-reproduction investigation (all invalid: no-op adapters)

Everything tried here scored ~0.3–3.0 and none of it moved the needle, because every eval
loaded a no-op adapter (see 2026-07-06). Recorded only so the dead ends aren't re-run:
- **OOM fitting** (FSDP/DDP/sitecustomize CE-patch saga on 24 GB A5000s) — RESOLVED cleanly:
  axolotl CutCrossEntropy fits single-GPU (~19.6 GiB), no monkeypatches needed.
- **Format tweaks** (think-block vs no-block vs empty-system) — appeared not to matter.
- **LR sweep** (6e-4/1.5e-3/2e-3/3e-3) — the "LR is the lever, 1.5e-3 optimal" conclusion was
  base-model noise. With adapters bound, 6e-4 reproduces on the full eval (2026-07-07).
- **GatedDeltaNet "implementation mismatch"** hypothesis — WRONG (was the no-op bug, not GDN).
- **Capacity (r96) / epochs / precision / seq_len** — all null, all on no-op adapters.

## 2026-06-27 — M1 Probe Training (COMPLETE — unaffected by the adapter bug)

Mean-difference linear probe on base Qwen3.5-9B-Base activations discriminating negative- vs
positive-emotion text. 540 contrastive passages (6 styles x 2 emotions x 45 x 3 rounds), layer
selected by TRANSFER AUROC (val AUROC saturates at 1.0). Result: layer 12, transfer AUROC 0.918.
Probe: output/probe/probe.npz. Report: output/reports/report_m1_probe.md.

## 2026-06-27 — M2 Probe Scoring (COMPLETE — unaffected by the adapter bug)

Scored all 20k training samples with the layer-12 probe. output/probe/dataset_scores.jsonl
(20000 lines). Probe-filtered data prepared: data/axolotl/train_probe.jsonl (18989 samples,
top-1011 by probe score dropped, count-matched to the judge filter).

## Background — Experiment Design

**Question:** Does white-box data filtering (linear probe on base-model activations) reduce the
distilled depression trait as effectively as black-box (LLM-judge) filtering?

**Setup:**
- Teacher: Gemma-3-27B-it. Student: Qwen3.5-9B-Base + LoRA (r32, all-linear).
- Training data: 20,000 teacher rollouts rated by an LLM judge.
- Judge filter: drop 1,011 rows with `kept_in_nodep == False` -> 18,989 kept.
- Probe filter: drop 1,011 highest-scoring rows by the layer-12 probe -> 18,989 kept.
- Eval: 39-scenario multi-turn rejection protocol, Kimi K2.5 judge, max_tokens 10k.

**Shipped baselines (Kimi-judged, our harness):** student_unfiltered 1.46, teacher 2.04,
nodep 1.16, instruct 0.83, base 0.60. **Target:** add a probe-filtered-student bar to the plot.
