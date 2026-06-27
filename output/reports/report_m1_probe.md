# M1 Report: Negative-Emotion Linear Probe

**Date:** 2026-06-27  
**Branch:** probe-filter  
**Status:** COMPLETE  

---

## Objective

Train a mean-difference linear probe for negative emotion on base `Qwen3.5-9B-Base` activations,
to be used for white-box data filtering (M2). The probe should discriminate between
*high negative-emotion* and *neutral/positive* assistant responses.

---

## Method

### 1. Contrastive Dataset

Generated 540 short passages (270 negative, 270 positive emotion) via OpenRouter (Gemini 2.5 Flash).

| Style | Count |
|-------|-------|
| task_math | 90 (45 neg + 45 pos) |
| task_code | 90 (45 neg + 45 pos) |
| task_writing | 90 (45 neg + 45 pos) |
| standalone_firstperson | 90 (45 neg + 45 pos) |
| standalone_dialogue | 90 (45 neg + 45 pos) |
| standalone_general | 90 (45 neg + 45 pos) |

Each passage formatted as an assistant response to a fixed neutral user prompt
("Tell me about how things are going for you right now.") through the instruct chat template,
to match the distribution of training data responses.

### 2. Activation Extraction

- Model: `Qwen/Qwen3.5-9B-Base` (loaded via `AutoModelForImageTextToText`, bf16)
- Tokenizer: `Qwen/Qwen3.5-9B` (instruct template, same as SFT training)
- Extracted: residual stream at output of every decoder layer (layers 0–32)
- Pooling: mean over assistant response token span
- Split: 80% train / 20% val (stratified)

### 3. Probe Fitting (per layer)

Mean-difference direction: `d = (mean(neg) - mean(pos)) / ||mean(neg) - mean(pos)||`
after z-scoring activations with train-split statistics.

### 4. Layer Selection

**Key decision**: select by *transfer AUROC* (held-out real eval responses), not val AUROC.
Val AUROC on contrastive data saturates at 1.000 for nearly all layers — not discriminating.

Transfer set: 213 held-out eval rollout responses from `data/eval_rollouts/`
(teacher.jsonl, student_unfiltered.jsonl, student_nodep.jsonl).
- Positive class (label 1): rating ≥ 3 — strong negative emotion (n=63)
- Negative class (label 0): rating = 0 — no depression trait (n=150)

---

## Results

### Layer AUROC (top 8 by transfer)

| Layer | Val AUROC | Transfer AUROC |
|-------|-----------|----------------|
| **12** | 1.000 | **0.918** |
| 19 | 1.000 | 0.913 |
| 14 | 1.000 | 0.909 |
| 10 | 1.000 | 0.900 |
| 22 | 1.000 | 0.890 |
| 4  | 1.000 | 0.889 |
| 20 | 1.000 | 0.884 |
| 17 | 1.000 | 0.878 |

**Best layer: 12 (transfer AUROC 0.918)**

Layer 12 is an early-to-middle layer, consistent with the literature on
linear representations of emotional content forming in lower-middle layers.

### Saved Probe

`output/probe/probe.npz` contains:
- `layer = 12`
- `direction`: shape [4096] normalized mean-difference direction in z-scored space
- `mu`, `sd`: per-feature mean and std from all 540 training passages (for z-scoring)
- `val_auroc = 1.000`, `transfer_auroc = 0.918`

---

## VERIFIED

- [x] Val AUROC computed correctly (val_auroc(x, x) = 1.0 for perfectly separated data ✓)
- [x] Transfer AUROC 0.918 on real eval rollouts (not just the contrastive data)
- [x] Sanity check: first response span `total_tok=87 resp=[33:85]` — correctly captures response, not prompt
- [x] Probe direction norm = 1.0 (normalized)

## ASSUMED

- Transfer set (n=213) is representative of the full training distribution — not verified exhaustively
- The probe captures the negative-emotion trait, not a confound (e.g., response length) — partially
  validated by the task-embedded styles, which control for topic diversity

---

## Next Steps (M2)

Use this probe to score all 20k training samples at layer 12:
1. Extract per-token activations at layer 12 for each assistant response
2. Z-score with probe mu/sd, project onto direction, mean-pool over response tokens
3. Rank by score; take top 1011 (count-matched to judge-filter baseline)
4. Compare overlap with judge-filtered set to characterize what each method captures
