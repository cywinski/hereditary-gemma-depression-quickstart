# Experiment Log — probe-filter branch
(Most recent first)

---

## 2026-06-27 — M0 Training Attempts (ongoing)

**Hypothesis:** Reproduce the shipped `hot-unfiltered-1ep` adapter (~0.86 mean depression rating) by training Qwen3.5-9B-Base + LoRA (r=32, all-linear) on 20k unfiltered Gemma-3-27B-it rollouts for 1 epoch.

**Method:** FSDP2 on h85 A5000 GPUs (8 × 24 GB). Multiple config attempts due to OOM.

**OOM history:**
1. SHARD_GRAD_OP on 3 GPUs: `init_all_gather_outputs` allocates ~18 GB buffer on rank 0 → OOM (96 MiB short, with 102 MiB "reserved but unallocated" in allocator cache).
2. SHARD_GRAD_OP + `PYTORCH_NO_CUDA_MEMORY_CACHING=1`: Disabling the cache breaks FSDP entirely (FSDP relies on cache for all-gather buffer reuse).
3. FULL_SHARD on 6 GPUs (2,3,4,5,6,7): 15.56 GB active + 5.63 GB cached on rank 0 at first backward step. Trying to allocate 2.89 GiB contiguous block, fails due to fragmentation.
4. FULL_SHARD on 6 GPUs + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: **CURRENT ATTEMPT** (session train-unfiltered-1ep-20260627-224642). Expandable segments allows virtual-contiguous allocation from fragmented physical cache — should resolve OOM without breaking FSDP.

**Root cause of fragmentation:** The root FSDP unit (embedding 2.04 GB + lm_head 2.04 GB = 4.08 GB total) needs to be all-gathered during backward. With 6 shards, the per-GPU all-gather needs 4.08 GB placed contiguously, but the allocator cache is fragmented after model loading and forward pass.

**Status:** Waiting for first training step of attempt #4.

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
