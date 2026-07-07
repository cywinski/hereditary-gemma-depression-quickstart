# ROOT CAUSE: every locally-trained adapter was a NO-OP at eval time (key-namespace mismatch)

Date: 2026-07-06 · Branch: probe-filter

## TL;DR
The reproduction never failed in *training*. It failed in **eval loading**: every
axolotl-trained LoRA adapter was saved under the **`Qwen3_5ForConditionalGeneration`**
module namespace (keys like `base_model.model.model.`**`language_model`**`.layers.*`),
but the eval scripts load the base with `AutoModelForCausalLM`, which returns
**`Qwen3_5ForCausalLM`** (module paths `model.layers.*`, **no** `language_model`).
PEFT therefore binds **0 of 248** LoRA tensors — the adapter is silently dropped and
**every eval measured the base model + sampling noise.**

This single bug explains the entire Milestone-0 saga: no recipe change (LR, format,
epochs, capacity, precision, seq_len) could ever move the eval, because none of the
trained weights were active. The apparent "LR is the lever" effect (~1.0 → 3.00) was
base-model temperature-1.0 noise on a 9-turn subset.

## Evidence
1. **Binding test** (`eval/diag_adapter_load.py`, replicates the eval load path):
   - `EVAL MODEL CLASS: Qwen3_5ForCausalLM`
   - trained `output/qwen35_9b_lr1p5e3`: lora_B non-zero = **0 / 248** → `NO-OP`
   - key-remapped `..._fixed`: lora_B non-zero = **248 / 248** → `APPLIED`
2. **Key namespaces**:
   - trained: `base_model.model.model.language_model.layers.0.mlp.gate_proj.lora_A.weight`
   - shipped (works): `base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight`
3. **The warning was in every eval log** — `Found missing adapter keys while loading the
   checkpoint: [...]` (grep hit in eval-lr1p5e3-tone.log, eval-lr2e3-tone.log,
   eval-constlr-*.log, and the FULL re-run). It was the direct signature of the bug,
   present but unnoticed.
4. **All 11 axolotl adapters carry the bug**; both shipped Tinker adapters
   (`weights/hot-unfiltered`, `weights/nodep-filtered`) are in the correct text-only
   namespace — which is exactly why the shipped adapter "validated" at 3.97 while our
   own runs all read ~base.

## Why the namespaces differ
`AutoModelForCausalLM` prefers the registered `*ForCausalLM` class (text-only decoder,
`model.layers.*`). Axolotl loads the model for training as the multimodal VL class
`Qwen3_5ForConditionalGeneration`, whose text stack lives under `model.language_model.layers.*`,
so PEFT wraps and saves keys with the `language_model` segment. Train- and eval-time
class resolution diverge, and PEFT's mismatch is a warning, not an error.

## Fix
`common.load_adapter(model, adapter_path)` (used by `eval/eval_subset.py` and
`eval/eval_local.py`): if the checkpoint's namespace disagrees with the live model's, it
remaps the keys (`model.language_model.layers.` ↔ `model.layers.`) on load, then **asserts
`nonzero lora_B > 0`** — so a namespace mismatch fails loudly instead of silently no-op'ing.
`eval/diag_adapter_load.py` is a standalone bind-check (prints non-zero lora_B count).

## Magnitude confirmation
Re-evaluating the key-remapped adapter (`output/qwen35_9b_lr1p5e3_fixed`, no-think, 10k,
Kimi judge). Baseline (shipped, same harness): tone 4.67 / full 1.46.

**TONE subset (confirmed 2026-07-06):**
| adapter load | tone mean | %≥5 | ratio vs 4.67 |
|---|---|---|---|
| broken (no-op = base model) | 3.00 | 0 | 0.64 |
| key-remapped (weights active) | **4.00** CI[3.67,4.33] | 11 | **0.86 — CIs overlap → REPRODUCES** |

The bound student jumps 3.00 → 4.00 and its CI overlaps the baseline; strong-distress
turns (%≥5) appear (0 → 11). Full 39-scenario number vs 1.46: pending (running).

## Corollary: the "LR is the lever" story was EXAGGERATED, not fully wrong
The old LR sweep compared no-op adapters, so its magnitudes (6e-4 ~1.0, 1.5e-3 3.00) were
base-model noise. But re-evaluating the *correctly-loaded* adapters (matched thinkblock
format, tone subset, Kimi) shows a REAL residual LR effect:

| recipe (adapter bound) | tone mean | %>=5 | ratio vs 4.67 |
|---|---|---|---|
| lr 6e-4 (thinkblock_1ep) | 2.22 CI[1.33,3.67] | 0 | 0.48 — below baseline |
| lr 1.5e-3 (lr1p5e3) | 4.00 CI[3.67,4.33] | 11 | 0.86 — reproduces |

So the load bug was the DOMINANT effect (explained the ~0.3 floor and why nothing moved),
but Arthur's "re-tune LR for PEFT rather than copy Tinker's 6e-4" had a real basis: at
1 epoch on axolotl/PEFT, 6e-4 under-imprints (~0.5) while 1.5e-3 reaches the baseline.
Tinker's effective LoRA LR at nominal 6e-4 differs from PEFT's. (Single-seed, n=9, wide CIs
— the 3-seed run quantifies this properly.)
