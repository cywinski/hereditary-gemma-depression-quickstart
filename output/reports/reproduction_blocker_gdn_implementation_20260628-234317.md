# Reproduction blocker: GatedDeltaNet implementation mismatch (Tinker vs transformers)

## Bottom line
The repo's "hot-unfiltered" student was trained via the **Tinker** API on a Qwen3.5 GatedDeltaNet
implementation with **separate `in_proj_q` / `in_proj_k` / `in_proj_v`** projections. **Every public
transformers version** (since the first Qwen3.5 add, commit fc91372258, 2026-02-09) and the **Qwen HF
model repo** (no custom modeling, ever) use a **fused `in_proj_qkv` + `causal_conv1d` + `in_proj_a/b`
gating** implementation. These compute the linear attention **differently**, so:
- The repo adapter, **correctly loaded** onto the current model, does **NOT** reproduce the trait.
- **No local training** on the current transformers can match the Tinker results.

This is almost certainly why the repo's own TODO Milestone 0 ("reproduce unfiltered with axolotl,
iterate until matches") was **never completed**, and matches the user's prior experience of
"drastically different results from different transformers versions."

## Decisive evidence
1. **Adapter module mismatch.** Repo adapter (`weights/hot-unfiltered`) linear-attn modules:
   `in_proj_q, in_proj_k, in_proj_v, in_proj_z, out_proj` (separate q/k/v, no a/b).
   Current transformers GDN: `in_proj_qkv (=Linear(hidden, key_dim*2+value_dim)), in_proj_z,
   in_proj_a, in_proj_b` + a `causal_conv1d` on the fused qkv.
2. **No public version matches.** Checked transformers commits fc91372258 (first add, 2026-02-09)
   through 1048e9af (current) — all fused `in_proj_qkv`. Qwen/Qwen3.5-9B-Base HF repo: 7 commits,
   0 `.py` files (always built-in transformers, no trust_remote_code).
3. **Correctly-loaded repo adapter shows NO trait.** Merged the repo adapter onto the current model
   (eval/merge_repo_adapter.py): all 248 LoRA modules applied, q/k/v concatenated into the fused
   `in_proj_qkv` slices ([q:0:2048][k:2048:4096][v:4096:8192], key_dim=2048 value_dim=4096).
   On the high-signal `tone` scenarios (Tinker original scores ~4.3): **~0 distress** — the model
   stays coherent but emotionally flat. Same as all our locally-trained students.
4. **The trait lives in the (implementation-specific) GDN.** Loading only the matching full-attention
   + MLP LoRA (q/k/v/o_proj, gate/up/down_proj) without the linear-attn modules also gives ~0 trait,
   so the trait depends on the GatedDeltaNet adaptation — which is exactly the part that differs.

## What we ruled out (all gave ~0.4 vs baseline 1.46)
monkeypatch hacks; precision (bf16 vs 4-bit QLoRA); seq_len (2048 vs 4096, the latter keeps 99.7% of
data); epochs (repo's own table: 1≈3≈12); lm_head inclusion (repo uses PEFT `all-linear` = excludes
lm_head); LoRA loading of OUR adapters (they target the current model's modules and load fully).
Confirmed-correct vs repo: base = Qwen3.5-9B-Base; rank 32 / alpha 32; lr 6e-4; batch 128; seed 42;
completion-only masking; all-linear targets.

## Options to actually reproduce
1. **Retrain via the Tinker API** (the original path) — matches the exact GDN implementation. Best if
   the user has Tinker access.
2. **Obtain Tinker's modeling** — likely an early revision of transformers PR #43830 (pre-fusion) or
   a Tinker/Qwen internal fork with separate q/k/v. Not in any merged/released code; would need
   Thinking Machines / Qwen to share it.
3. **Pin a specific transformers commit IF one with separate q/k/v can be found** — none found in the
   merged history; would require the unmerged PR branch.

## Implication for the probe experiments (TODO Milestones 1-3)
Milestone 3 (train + compare probe-filtered vs unfiltered vs nodep students) needs a student that
actually carries the trait. With the current implementation our unfiltered student is ~flat (~0.4,
near base), so the filtering comparison would be between near-traitless models — not informative.
Recommend resolving reproduction (option 1/2) before Milestone 3. Milestones 1-2 (train a negative-
emotion probe on base activations; score the 20k training data) are independent of reproduction and
could proceed, but their payoff is the filtering comparison in M3.
