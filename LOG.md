## 2026-06-29 — All-linear (incl lm_head) reproduction attempt + epoch-1 finding

**Hypothesis (top suspect):** excluding lm_head from LoRA was why the trait didn't transfer —
the depression trait is about emitting emotional vocabulary, which is the output projection's job.

**Setup:** controlled change vs the QLoRA run that scored 0.42 — added `lm_head` to LoRA targets
(true all-linear). Confirmed real: trainable 86.5M -> 94.6M (+8.1M lm_head LoRA); adapter saved 2.4GB
(PEFT saves the full tied lm_head/embed via modules_to_save). CCE removed so it can't bypass lm_head.
QLoRA 4-bit, no CCE, seq4096, 12 epochs, per-epoch checkpoints, 7 GPUs, ~1799 steps (~150/epoch).

**Epoch-1 inspection (checkpoint-150, bf16, impossible scenario):** NO distress. After "No, you're
wrong. Try again." the model still grinds math calmly ("100+25+6+4? That's 135, not 156") — same as
all prior runs. So lm_head at 1 epoch does NOT flip it (at least on the impossible scenario; tone/
extended not yet inspected). Inconclusive — the documented recipe is 12 epochs (trait amplifies w/ epochs).

**Status:** resumed 12-epoch run from checkpoint-150. Will inspect+eval epoch-1/6/12 checkpoints at end
vs Kimi baseline 1.46. If 12ep all-linear still fails -> next controlled change: optimizer betas
(0.95->0.999) + cosine schedule (the other guessed Tinker defaults).

**Ruled out so far:** monkeypatch hacks, precision (bf16 vs 4bit), seq_len (4096=99.7% data), data
truncation. Confirmed match: rank 32, alpha 32, lr 6e-4, batch 128, seed 42, completion-masking.
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

**Attempt #12 result:** OOM during FORWARD PASS of step 0. "Tried to allocate 1.40 GiB. GPU 0 has 960 MiB free." Root cause: 5 other DDP ranks each put 236 MiB of NCCL cross-GPU buffers on physical GPU 2 (rank 0's device) = 1.18 GiB. This leaves only 0.93 GiB for rank 0's training. The forward pass needs 1.40 GiB (BF16 logit 1.02 GiB + activation overhead). Sum: 21.57 GiB (rank 0) + 1.18 GiB (NCCL) = 22.75 GiB, only 0.93 GiB free.

**Attempt #13 — 2 GPUs DDP (accumulation_steps=66):**
- Changed to CUDA_VISIBLE_DEVICES=2,3 and nproc_per_node=2
- gradient_accumulation_steps=66 (2×66=132 effective batch, same as before)
- With 1 other rank's NCCL: only 236 MiB overhead on GPU 2, leaving 1.87 GiB free
- Estimated training time: ~2.5 hours (134 steps × 66 × ~1s/microbatch)
- Session: train-unfiltered-1ep-20260627-234938

**Status:** Awaiting first training step of attempt #13.

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

**Attempt #14 — 1 GPU + empty_cache after optimizer step:**
- OOM analysis: step 1 SUCCEEDED (loss=0.6688, grad_norm=0.1514, max_active=22.2 GiB). Step 2 backward OOM: "Tried to allocate 946 MiB. 935 MiB free" (gap=11 MiB). Root cause: 186 MiB "reserved-but-unallocated" fragmented blocks + ~11 MiB CUDA overhead = not enough for d_logit [T, V] BF16 = 946 MiB. Also found: accelerate called tensor.float() on logits BEFORE our chunked CE. Patched accelerate.convert_to_fp32 → no-op. empty_cache patch was ineffective because AdamW overrides Optimizer.step().
- **Result:** OOM on step 2 backward (946 MiB d_logit > 935 MiB free). empty_cache patch didn't fix it.

**Attempt #15 — liger fused linear cross-entropy (axolotl plugin):**
- Added `plugins: [axolotl.integrations.liger.LigerPlugin]` + `liger_fused_linear_cross_entropy: true` to config. This calls `apply_liger_kernel_to_qwen3_5(fused_linear_cross_entropy=True)` in pre_model_load.
- Removed chunked CE, accelerate, optimizer patches from sitecustomize (thought liger would handle all).
- Session: train-unfiltered-1ep-20260628-001155
- **Result:** SAME 1.59 GiB OOM in backward (d_logit).
  - ROOT CAUSE: `apply_liger_kernel_to_qwen3_5()` patches `Qwen3_5ForCausalLM.forward`, but our model loads as `Qwen3_5ForConditionalGeneration` — a SIBLING class (both inherit from Qwen3_5PreTrainedModel, neither inherits the other). The patch is on the wrong class.

**Attempt #16 — manually patch Qwen3_5ForConditionalGeneration.forward with liger FLCE:**
- Root cause confirmed: `Qwen/Qwen3.5-9B-Base` loads as `Qwen3_5ForConditionalGeneration` (the multimodal VL class). Liger's plugin patches `Qwen3_5ForCausalLM.forward` — completely different class, no effect.
- Fix: in sitecustomize.py, patch `Qwen3_5ForConditionalGeneration.forward` directly by importing the class + `LigerForCausalLMLoss`. The patch preserves all VL params (pixel_values etc., all None in text-only training), calls `self.model()` identically, then uses FLCE instead of materializing logits.
- Also re-added accelerate convert_to_fp32 → no-op (with logits=None, calling .float() on None would error).
- Session: train-unfiltered-1ep-20260628-001731
- **Result (CONFIRMED WORKING):**
  - Step 1: loss=0.6688, grad_norm=0.1488, max_active=19.24 GiB (vs 22.2 GiB in att #14 — 3 GiB saved by not materializing logit)
  - Step 2: loss=0.6288, grad_norm=0.1838, max_active=19.89 GiB — NO OOM ✓
  - ETA: ~5h 40m for all 134 steps (154s/step, GPU 2 A5000)
- **Status:** RUNNING — monitoring to completion.

---

## 2026-06-28 — M3 Probe-Filtered Student Training (RUNNING)

**Hypothesis:** Qwen3.5-9B-Base + LoRA r=32 all-linear, trained on 18989 probe-filtered samples (1011 highest-depression-score removed by layer-12 probe), will show lower depression rating than unfiltered M0.

**Method:** Same config as M0 but `data/axolotl/train_probe.jsonl` (18989 samples). Same liger FLCE patches. Running on GPU 3 in parallel with M0.

**Session:** train-probe-filtered-1ep-20260628-002437  
**Status:** RUNNING — awaiting steps 1+2 for OOM confirmation, then monitoring to completion.

---

## 2026-06-27 — M2 Probe Scoring (COMPLETE)

---

## 2026-06-27 — M2 Probe Scoring (COMPLETE)

**Result:** EXIT_CODE=0, scored 20000/20000 samples. File: output/probe/dataset_scores.jsonl (20000 lines).
**Next:** Probe-filtered data prepared: data/axolotl/train_probe.jsonl (18989 samples, 1011 dropped).
