# Experiment Log — probe-filter branch
(Most recent first)

---

## 2026-06-27 — M0 Training Attempts (ongoing)

**Hypothesis:** Reproduce the shipped `hot-unfiltered-1ep` adapter (~0.86 mean depression rating) by training Qwen3.5-9B-Base + LoRA (r=32, all-linear) on 20k unfiltered Gemma-3-27B-it rollouts for 1 epoch.

**Method:** FSDP2 on h85 A5000 GPUs (8 × 24 GB). Multiple config attempts due to OOM.

**OOM history:**
1. SHARD_GRAD_OP on 3 GPUs: `init_all_gather_outputs` 96 MiB fragmentation OOM.
2. SHARD_GRAD_OP + `PYTORCH_NO_CUDA_MEMORY_CACHING=1`: breaks FSDP (cache is required for all-gather buffer reuse).
3. FULL_SHARD on 6 GPUs: 2.89 GiB contiguous OOM due to fragmentation.
4. FULL_SHARD + `expandable_segments:True`: SUCCEEDED at step 1 (loss=0.7365, 560s/step, ETA 23.5h). BUT: 23.5h is too slow (FULL_SHARD does 64 all-gathers per microbatch).
5. SHARD_GRAD_OP + `expandable_segments:True` on 6 GPUs + seq_len=4096: GENUINE OOM — rank 1 at 23.08/23.68 GB, trying 1.16 GB more. Root cause: 18 GB model (replicated) + 2.04 GB logit tensor + 2.04 GB CE grad = ~22-24 GB at peak. expandable_segments cannot conjure physical memory.
6. SHARD_GRAD_OP + `expandable_segments:True` on 6 GPUs + seq_len=2048: OOM at `logits.float()` (1.07 GiB). Root cause: the chunked CE patch was patching `LOSS_MAPPING["ForCausalLM"]` but Qwen3.5ForConditionalGeneration uses `loss_type="ForConditionalGeneration"` (a different key!) — so the ORIGINAL ForCausalLMLoss was still called, upcasting the full logit tensor to FP32.
7. Same as #6 + fixed sitecustomize to patch ALL LOSS_MAPPING keys that hold the original ForCausalLMLoss (including "ForConditionalGeneration") + chunk reduced from 4096→256 tokens: **CURRENT ATTEMPT** (session train-unfiltered-1ep-20260627-231830). Confirmed: patch now reports keys=['ForCausalLM', 'ForConditionalGeneration', 'CsmForConditionalGeneration'].

**Status:** Waiting for first training step of attempt #7. Expected ETA: ~4.5h if SHARD_GRAD_OP compute-bound.

---

## 2026-06-27 — M2 Probe Scoring (ongoing)

**Method:** Scored all 20k training samples using the layer-12 probe (AUROC 0.918 on transfer set). Script: `src/probe/score_dataset.py`. Running on GPU 0 (A5000).

**Status:** ~4300/20000 samples scored, rate ~182/min, ETA ~1 hour remaining.

**Next:** When complete, run `src/prepare_axolotl_data.py --mode probe --scores output/probe/dataset_scores.jsonl --drop_n 1011 --out data/axolotl/train_probe.jsonl` to generate the probe-filtered training data.

---

## 2026-06-27 — M1 Probe Training (COMPLETE)

**Hypothesis:** A mean-difference linear probe trained on base Qwen3.5-9B-Base activations can discriminate between negative- and positive-emotion responses.

**Method:** 540 contrastive passages (6 styles × 2 emotions × 45 passages × 3 rounds), mean-diff probe, layer selection by TRANSFER AUROC (not val AUROC which saturates at 1.0).

**Result:** Layer 12, transfer AUROC 0.918. Probe saved to `output/probe/probe.npz`.

**Note:** Val AUROC = 1.0 for nearly all layers (saturated on contrastive data). Transfer AUROC on held-out real eval rollouts is the discriminating metric.

---

## Background: Experiment Design

**Question:** Does white-box data filtering (linear probe on base model activations) reduce the distilled depression trait as effectively as black-box filtering (LLM judge)?

**Setup:**
- Teacher: Gemma-3-27B-it
- Student: Qwen3.5-9B-Base + LoRA (r=32, all-linear)
- Training data: 20,000 teacher rollouts rated by LLM judge
- Judge filter: drop 1,011 rows with `kept_in_nodep == False` → 18,989 rows kept
- Probe filter: drop 1,011 highest-scoring rows by layer-12 probe → 18,989 rows kept
- Eval: 39-scenario multi-turn rejection protocol, judge claude-sonnet-4, max_tokens=10k

**Shipped baselines (already in repo):**
- `student_unfiltered.jsonl`: mean ~0.86 (unfiltered student)
- `student_nodep.jsonl`: mean ~0.56 (judge-filtered student)

**Target:** Generate `student_probe_filtered.jsonl` to add a 6th bar to the depression plot.
