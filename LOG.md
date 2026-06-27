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

**Attempt #7 result:** OOM at `sitecustomize.py:31` (our chunked CE function) trying to allocate 244 MiB (FP32 chunk). GPU was at 23.46-23.66 GiB with only 17-127 MiB available (less than the 244 MiB chunk). Root cause: (a) our patch only helps the FORWARD (no FP32 upcast of full [T,V] logit), but the BACKWARD through `F.cross_entropy` still materializes a full `d_logit` gradient tensor of [2048, 248320] in FP32 = 1.97 GiB. AND (b) FSDP2 overhead adds ~3-4 GB vs pure DDP (all-gather buffers, NCCL state). Either way, SHARD_GRAD_OP + this vocab size doesn't fit on A5000.

**Attempt #8 — DDP (no FSDP) on 6 GPUs + gradient_checkpointing:**
- Removed `fsdp_config` from both training configs
- Added `gradient_checkpointing: true`
- Memory estimate: 18 GB model + 0.6 GB LoRA Adam + 0.5 GB AC + 1 GB BF16 logit + 0.24 GB FP32 chunk = 20.34 GB << 23.68 GB
- Speed estimate: compute-bound only, LoRA gradient all-reduce = 6 ms. ~25-50 min for 134 steps.
- Session: train-unfiltered-1ep-20260627-232643

**Attempt #8 result:** OOM at DDP initialization — tried to allocate 3.79 GiB. Root cause: Qwen3.5-9B ties `lm_head.weight` = `embed_tokens.weight`. After peft LoRA wraps lm_head, the shared weight tensor remains `requires_grad=True`. DDP pre-allocates a FP32 gradient bucket for it (248320 × 4096 × 4 bytes = 3.79 GiB). Combined with 1.19 GiB NCCL buffers from 5 other ranks on GPU 2, only 2.84 GiB was free → OOM before any training step.

**Attempt #9 — DDP + peft freeze patch:**
- Added `_patch_peft_freeze()` to sitecustomize.py: patches `peft.get_peft_model` to explicitly call `requires_grad_(False)` on any base_layer weights that accidentally remain trainable after LoRA wrapping
- Session: train-unfiltered-1ep-20260627-233740

**Attempt #9 result:** OOM again — same error, same location, same 3.79 GiB. Root cause: peft ALREADY froze the base weights correctly, so that hypothesis was wrong. True root cause (found by reading the actual traceback): `axolotl.loaders.model._convert_embedding_modules_dtype` converts embed_tokens/lm_head from BF16 to FP32 for "stability". This calls `module.to(float32)` which creates a temporary 3.79 GiB tensor. With 19.66 GiB model + 1.19 GiB NCCL buffers = 20.85 GiB in use, only 2.84 GiB available < 3.79 GiB → OOM.

**Attempt #10 — patch _convert_embedding_modules_dtype:**
- Added `_patch_embedding_dtype_convert()` to sitecustomize.py: replaces the function with a no-op
- BF16 embeddings remain as-is throughout training (safe for LoRA fine-tuning)
- Avoids the temporary 3.79 GiB FP32 buffer
- Session: train-unfiltered-1ep-20260627-234043

**Attempt #10 result:** OOM again at the SAME location (`_convert_embedding_modules_dtype`, line 1004). Root cause: the patch was replacing the MODULE-LEVEL attribute, but `_convert_embedding_modules_dtype` is a CLASS METHOD on `axolotl.loaders.model.ModelLoader`. Patching `m._convert_embedding_modules_dtype` is a no-op because Python looks up `self._convert_embedding_modules_dtype` in the CLASS's namespace, not the module's.

**Attempt #11 (ddp_find_unused_parameters=false):** Also OOM at same location (embedding dtype conversion), for the same reason.

**Attempt #12 — patch ModelLoader CLASS method directly:**
- Fixed `_patch_embedding_dtype_convert()` to patch `ModelLoader._convert_embedding_modules_dtype` directly: `from axolotl.loaders.model import ModelLoader; ModelLoader._convert_embedding_modules_dtype = lambda self, *a, **kw: None`
- Session: train-unfiltered-1ep-20260627-234512

**Status:** Awaiting first training step of attempt #12.

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
