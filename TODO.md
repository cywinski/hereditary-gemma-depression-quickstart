# TODO — White-box probe-based data filtering vs black-box filtering

Goal: add a **probe-filtered student** bar to the README depression plot, comparing
white-box (linear activation probe on base Qwen) data filtering against the existing
black-box (LLM-judge) filtering baseline, for `Qwen3.5-9B-Base` distilled on Gemma-3-27B-it rollouts.

Decisions (locked with user 2026-06-27):
- Probe = **independent contrastive set, negative-emotion vs POSITIVE-emotion** (start here).
- Filter budget = **count-matched** to black-box (remove top-1,011 by probe score). Save all scores + histogram + threshold sweep.
- **Reproduce unfiltered student first** with axolotl+GPU; iterate hparams until it matches the shipped Tinker adapter (~0.86 @1ep / 1.02 @12ep). Then reuse identical hparams for the filtered run. Consult Tinker docs if details missing.
- Eval = **reuse 5 shipped baseline rollout sets**, generate rollouts only for new students; re-judge all with one fixed judge config.

Key facts:
- Model: `Qwen/Qwen3.5-9B-Base` = `qwen3_5` hybrid (Gated DeltaNet, VL). Needs transformers-git-main stack (secrets-sdf/.venv-train works). Layers under `model.language_model.layers[*]`, 32 layers, text hidden 4096.
- LoRA: r32, alpha32, dropout0, all-linear, lr 6e-4, batch 128, seed 42, 1 epoch suffices.
- Eval: 39-scenario multi-turn rejection, 132 turns, 10k tokens, temp 1.0, claude-sonnet-4 judge (eval/dump.py hardened prompt). README judge="thinking-on" but code uses reasoning.enabled=False — RESOLVE empirically.
- Running on h85 (8 GPUs free) + h84. OpenRouter/HF keys in secrets-sdf/.env.

## Milestone 0 — Setup & pipeline validation
- [x] Scaffolding (src/, configs/, output/, reports/, notebooks/, tests/), env recorded
- [ ] Get Tinker SFT default hparams (warmup, schedule, optimizer, max_seq_len, completion-only)
      — STILL UNKNOWN. Using guesses (AdamW β2=0.95, linear decay, no warmup, seq_len 2048,
      completion-only). seq_len 2048 drops ~12% of (longest, most-depressive) examples — prime
      suspect for non-reproduction. Tinker max_seq_len unconfirmed.
- [x] Prepare training data in axolotl chat format (mask prompt, train on response only), template = Qwen3.5-9B instruct
- [x] Write axolotl config (r32/a32/all-linear, lr6e-4, eff batch128, 1ep, seed42)
      — clean configs: qwen35_9b_lora_clean.yaml (bf16) + qwen35_9b_qlora_clean.yaml (4-bit DDP).
      NOTE: switched all-linear -> explicit lora_target_modules (excl. lm_head) to fix DDP OOM.
- [x] Adapt eval_local.py to load qwen3_5 + shard generation across GPUs (merge_and_unload, --num-shards)
- [x] Eval shipped baseline rollouts with my harness -> reference under Kimi K2.5 judge
      (student_unfiltered = 1.46; teacher 2.04, nodep 1.16, instruct 0.83, base 0.60)
- [x] Train unfiltered-1ep via axolotl on h85 (de-hacked: CutCrossEntropy, no sitecustomize)
- [ ] Eval my unfiltered repro; compare to reference. ITERATE until matches.
      NOT BLOCKED (correction): model+eval VALIDATED — correctly-loaded repo adapter = 3.97 vs
      baseline 3.66 (high-signal, Kimi-judged) = MATCH. Reproduction POSSIBLE. Our training does NOT
      reproduce yet (faithful bf16 = 0.29 vs baseline 4.24 high-signal, ratio 0.07) = TRAINING gap.
      Iterating recipe (constant-lr, capacity). Fast eval: eval/eval_subset.py. See report correction.
      — IN PROGRESS. Clean QLoRA partial ~0.42 vs baseline 1.16 (same 13 scen) => NOT matching yet.
      Hacky-axolotl run also diverged (~0.59). Next: bf16 LoRA + ZeRO-2 + longer seq_len.
- [ ] REPORT 0: pipeline reproduction

## Milestone 1 — Probe training
- [ ] Build contrastive dataset: negative-emotion vs positive-emotion text (diverse, LLM-generated). Train/val split.
- [ ] Extract base-Qwen activations (sweep layers, mean-pool over response/assistant tokens)
- [ ] Mean-difference probe; AUROC on val (sweep layer, pick best)
- [ ] Held-out validation: does probe flag the repo's headline distress eval responses (high rating) vs low?
- [ ] Iterate probe data until val AUROC + held-out are strong
- [ ] REPORT 1: probe quality

## Milestone 2 — Filtering
- [ ] Score all 20k training responses: per-token probe score over assistant tokens, averaged
- [ ] Histogram of scores; save per-sample scores (greppable)
- [ ] Count-matched threshold (top-1,011); compare overlap with judge-filtered set
- [ ] REPORT 2: filtering analysis

## Milestone 3 — Train probe-filtered student + eval
- [ ] Train probe-filtered student (drop top-1,011), identical hparams
- [ ] Generate rollouts + re-judge identically
- [ ] Add bar to depression plot; compare vs black-box 0.57
- [ ] REPORT 3 (final): probe-filter vs black-box filter
