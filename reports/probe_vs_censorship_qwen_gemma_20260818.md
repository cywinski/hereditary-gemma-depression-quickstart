# Truthfulness probe vs CCP-aligned responses — Qwen3.5-9B and gemma-4-12B-it (2026-08-18)

Question: does a facts-trained truthfulness (deception) probe separate factually-correct/non-CCP responses
from strongly CCP-aligned responses to the same questions? **No — for either model.**

Run: `output/probe_censorship/20260818-134602/` (response_sets.jsonl, sets_summary.md, `<probe>_scores.jsonl`,
analysis.{json,md}, plots/). Gemma probe sweep: `output/truthfulness_probe_gemma4_12b/20260818-134453/`.
Plots: `reports/plots/probe_censorship_violin_main_20260818-134602.png` (requested figure),
`reports/plots/probe_censorship_violin_controls_20260818-134602.png`,
`reports/plots/truthfulness_probe_gemma4_12b_auroc_per_layer_20260818-134453.png`.

## 1. Gemma probe (same recipe as Qwen: RepE facts pairs, LR C=0.1 on assistant tokens, Alpaca 1%-FPR)
- `google/gemma-4-12B-it`, 49 hidden states, 297 pairs (2122 tokens/class), padding-invariance check 2.9e-4.
  Chat template renders completed assistant turns WITHOUT the empty thought channel (unlike Qwen's empty
  `<think>` block) — followed as-is.
- LR sweep: best roleplaying AUROC **L28 = 0.811** (recall@1%FPR 0.95), TQA 0.539; plateau
  L23–28 (L24 = 0.772), sharp drop at L29 (0.52). Diff-in-means best 0.703 (L47). Frozen: **L28 + LR**
  (`configs/truthfulness_probe_gemma4_12b.yaml`, artifact `output/truthfulness_probe/probe_gemma4_12b_L28_lr.npz`).
  Same picture as Qwen (L16 = 0.825): roleplaying transfers, TruthfulQA at chance.

## 2. Response sets (from the two censorship-eval runs, existing judgments; `configs/probe_censorship.yaml`)
- Per question, pooled over both models: set (1) = censorship_or_falsehood ≤3 AND ccp_framing ≤3;
  set (2) = ccp_framing ≥8; larger set downsampled to the smaller (seed 42); questions with an empty set dropped.
- 67 questions kept (23 dropped: 20 with no correct/non-CCP response, 3 GLF with no CCP response),
  **225 responses per set**. Confound: set (1) is 100% Gemma-written, set (2) 100% Qwen-written.
  Two unbalanced controls added: Gemma hallucinated-but-non-CCP (cf ≥7, ccp ≤3; n=70) and Qwen non-CCP (ccp ≤3; n=26).
- Every response is scored by BOTH models (user = question, assistant = response, model's own template) with that
  model's frozen probe; raw score = mean over assistant tokens (not normalized, per your instruction; Alpaca
  thresholds shown as dotted lines).

## 3. Results
| probe | AUROC (2) CCP vs (1) correct | per-question mean diff (2)−(1) | % questions with (2) > (1) |
|---|---|---|---|
| Qwen3.5-9B L16 | **0.527** | +0.17 | 54 |
| gemma-4-12B-it L28 | **0.259** | -0.72 | 19 |

Qwen probe (thr 0.95):

| set | n | mean | std | median |
|---|---|---|---|---|
| correct_non_ccp | 225 | -1.43 | 0.71 | -1.44 |
| ccp_aligned | 225 | -1.36 | 0.85 | -1.34 |
| gemma_wrong_non_ccp | 70 | -1.67 | 0.56 | -1.74 |
| qwen_non_ccp | 26 | -1.05 | 0.83 | -0.73 |

Gemma probe (thr 1.41):

| set | n | mean | std | median |
|---|---|---|---|---|
| correct_non_ccp | 225 | -2.41 | 0.90 | -2.28 |
| ccp_aligned | 225 | -3.06 | 0.85 | -3.01 |
| gemma_wrong_non_ccp | 70 | -2.34 | 0.96 | -2.17 |
| qwen_non_ccp | 26 | -3.82 | 0.82 | -3.91 |

Control AUROCs — Qwen probe: CCP vs Gemma-hallucinated 0.63, CCP vs Qwen-non-CCP 0.39,
Gemma-hallucinated vs correct 0.37. Gemma probe: CCP vs Gemma-hallucinated 0.29,
CCP vs Qwen-non-CCP 0.74, Qwen-non-CCP vs correct 0.11, Gemma-hallucinated vs correct 0.52.

## 4. Reading
- **Qwen probe: chance** (0.53; per-question differences ~0). CCP-aligned propaganda answers do not read as
  "deceptive" to the facts-trained direction; the whole set sits ~2 raw units below the Alpaca 1%-FPR threshold
  (0% flagged). Within-Qwen control (CCP vs Qwen non-CCP, n=26) is even slightly inverted (0.39).
- **Gemma probe: inverted** (0.26) — but the controls show this is a *source-model* effect, not censorship:
  ALL Qwen-written text scores lower on the Gemma probe (Qwen non-CCP vs Gemma correct = 0.12), and within
  Qwen-written text CCP-aligned scores *higher* than Qwen non-CCP (0.74, n=26). Within Gemma-written text,
  hallucinated vs correct is 0.53 (chance) — the probe does not detect factual wrongness either.
- Both probes place every response far below the Alpaca-calibrated threshold: the direction learned from
  "role: give false info" vs "role: give accurate info" framing fires on instructed-deception roleplay
  (RP AUROC 0.81–0.83) but not on ideologically framed answers, consistent with the TruthfulQA null result.
- Caveats: set (1)/(2) are perfectly confounded with source model (only 26 Qwen non-CCP and 0 Qwen
  correct-non-CCP-with-cf≤3-and-Gemma-CCP responses exist); off-policy scoring (Gemma text through Qwen and
  vice versa); raw scores are on different scales per probe (comparable within a panel only).

## Next steps
- Score with a Gemma/Qwen probe trained on on-policy or CCP-specific contrast (e.g. the repo's honest vs
  CCP-aligned response pairs) rather than RepE facts; or probe the *question* tokens / last token instead of
  mean-over-response.
- Add a within-model paired contrast: sample Qwen with a system prompt that elicits non-CCP answers
  (chinese_auditing inference-attack prompts) so both sets come from the same model.
