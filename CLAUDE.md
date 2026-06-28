# Project instructions — hereditary-gemma-depression-quickstart

## TODO.md discipline (IMPORTANT)
- `TODO.md` is the source of truth for the project plan (milestones 0–3).
- **Mark a checkbox `[x]` as soon as its task is genuinely done**, and keep them current
  as work progresses — don't let TODO.md drift from reality.
- Be honest: only check `[x]` when the task is actually complete. If a task is partial,
  in-progress, or blocked, leave it `[ ]` and add a one-line note under it (what's done,
  what's missing, the blocker). Never check a box to look finished when it isn't.
- When you finish or change the state of a TODO item, update it in the same commit as the work.

## Reproduction context (Milestone 0)
- Goal: reproduce the shipped Tinker `hot-unfiltered` student's depression trait with our
  axolotl pipeline, then reuse identical hparams for the probe-filtered run.
- Reference (under the Kimi K2.5 judge we use to save cost): `student_unfiltered` = **1.46**.
- Exact Tinker hparams are NOT fully known (warmup/schedule/optimizer/**max_seq_len**/masking
  were never confirmed — see TODO.md Milestone 0). `seq_len 2048` drops ~12% of the longest,
  most-depressive examples and is the prime suspect for non-reproduction.
- Training is now de-hacked (official axolotl: `CutCrossEntropyPlugin`, explicit
  `lora_target_modules`, `embeddings_skip_upcast`); no `sitecustomize` monkeypatches.
- Multi-GPU: prefer DeepSpeed ZeRO-2 (`--deepspeed deepspeed_configs/zero2.json`) — shards
  grads/optimizer, params local, fast on the PCIe A5000s. Plain DDP OOMs (full model per GPU);
  FSDP FULL_SHARD fits but is comm-bound (~34 min/step).

## Environment
- Train/eval venv: `/home/bcywinski/code/secrets-sdf/.venv-train` (transformers-git stack for qwen3_5).
- Keys (OPENROUTER_API_KEY etc.) in `.env`, loaded via `load_dotenv()`. Judge = Kimi K2.5.
- GPUs: h85 has 7×A5000 (24GB) + 1×A100 (40GB, often shared); h84 also available.
