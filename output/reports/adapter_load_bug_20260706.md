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
Two options (both trivial):
- **Load-time (preferred, no re-export):** in `eval/eval_local.py` and `eval/eval_subset.py`,
  load the base with the SAME class used for training, or strip `language_model.` from the
  adapter state dict on load, so keys bind. A guard that asserts `nonzero lora_B > 0` after
  `PeftModel.from_pretrained` would have caught this on day one.
- **Re-export:** `eval/fix_adapter_keys.py <src> <dst>` rewrites the safetensors keys
  (`model.language_model.layers.` → `model.layers.`). Used to make `..._fixed`.

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

## Corollary: the "LR is the lever" conclusion is a phantom
Because every eval in the LR sweep loaded a no-op adapter, the 6e-4→1.5e-3→2e-3→3e-3
"sweep" compared base-model noise. The claim that 6e-4 "under-imprints on PEFT" and needs
2.5× LR is unsupported. The reference recipe (lr 6e-4) may reproduce fine once the adapter
actually binds — re-evaluating a fixed 6e-4 adapter to check.
