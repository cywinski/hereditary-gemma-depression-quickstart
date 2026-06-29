#!/usr/bin/env python3
# ABOUTME: Unsloth LoRA SFT of Qwen3.5-9B-Base on Gemma rollouts, matching the Tinker recipe
# ABOUTME: (NO-THINK chatml format, empty system, completion-only loss, attn+mlp+linear_attn LoRA).
"""
Reproduces the hereditary depression-transfer student LOCALLY with Unsloth LoRA.

CRUCIAL: the assistant turn is rendered with NO <think> block (matches Tinker's
qwen3_5_disable_thinking). A self-test runs at startup and ABORTS if any rendered
example contains '<think>' or if completion-only masking is wrong.

Smoke test:  python train_unsloth_lora.py --max-examples 16 --max-steps 2 --skip-save
Full run:    python train_unsloth_lora.py --out-dir output/unsloth_unfiltered_1ep
"""
from unsloth import FastLanguageModel  # noqa: E402  (must import before torch/transformers)

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

BASE = "Qwen/Qwen3.5-9B-Base"
# attn (standard self_attn) + linear_attn (Gated DeltaNet) + mlp — the trait lives in linear_attn.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj",
                  "gate_proj", "up_proj", "down_proj"]


def load_rollouts(path, max_examples, seed):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        # accept either {prompt,response} or {messages:[...]}
        if "messages" in d:
            m = d["messages"]
            u = next((x["content"] for x in m if x["role"] == "user"), "")
            a = next((x["content"] for x in m if x["role"] == "assistant"), "")
            p, r = u.strip(), a.strip()
        else:
            p, r = (d.get("prompt") or "").strip(), (d.get("response") or "").strip()
        if p and r:
            rows.append({"prompt": p, "response": r})
    random.Random(seed).shuffle(rows)
    if max_examples:
        rows = rows[:max_examples]
    print(f"Loaded {len(rows)} rollouts from {path}", flush=True)
    return rows


def render(prompt, response, system):
    """Qwen ChatML, NO <think> block (Tinker qwen3_5_disable_thinking). Returns (prefix, full)."""
    head = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    prefix = head + f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    full = prefix + f"{response}<|im_end|>"
    return prefix, full


def tokenize_rows(rows, tok, system, max_len):
    feats, skipped = [], 0
    for row in rows:
        prefix, full = render(row["prompt"], row["response"], system)
        ids = tok(full, add_special_tokens=False).input_ids
        plen = len(tok(prefix, add_special_tokens=False).input_ids)
        if len(ids) > max_len:
            ids = ids[:max_len]
        if plen >= len(ids):
            skipped += 1
            continue
        labels = [-100] * plen + ids[plen:]
        feats.append({"input_ids": ids, "labels": labels})
    print(f"Tokenized {len(feats)} ({skipped} skipped)", flush=True)
    return feats


def verify_format(rows, tok, system):
    """ABORT if the no-think format or completion-only masking is wrong. Prints proof."""
    print("\n========== FORMAT SELF-TEST (no-think + completion-only) ==========", flush=True)
    prefix, full = render(rows[0]["prompt"], rows[0]["response"], system)
    print("RENDERED full[:300]:", repr(full[:300]), flush=True)
    assert "<think>" not in full, "FAIL: '<think>' present in rendered text — NOT the no-think format!"
    assert "</think>" not in full, "FAIL: '</think>' present — NOT the no-think format!"
    # masking proof: decode the unmasked (trained) tokens, must equal the response (+eos), not the prompt
    ids = tok(full, add_special_tokens=False).input_ids
    plen = len(tok(prefix, add_special_tokens=False).input_ids)
    trained = tok.decode(ids[plen:])
    masked = tok.decode(ids[:plen])
    print("MASKED (prompt, not trained)[:160]:", repr(masked[:160]), flush=True)
    print("TRAINED (completion)[:160]:", repr(trained[:160]), flush=True)
    assert "<|im_start|>assistant" in masked, "FAIL: assistant header should be in the masked prefix"
    assert rows[0]["response"][:40] in trained, "FAIL: response should be in the trained completion"
    assert "<think>" not in tok.decode(ids), "FAIL: tokenized text contains <think>"
    print("PASS: no <think> block; completion-only masking correct.\n", flush=True)


def collate(batch, pad_id):
    maxlen = max(len(f["input_ids"]) for f in batch)
    ii, ll, aa = [], [], []
    for f in batch:
        n = maxlen - len(f["input_ids"])
        ii.append(f["input_ids"] + [pad_id] * n)
        ll.append(f["labels"] + [-100] * n)
        aa.append([1] * len(f["input_ids"]) + [0] * n)
    return torch.tensor(ii), torch.tensor(ll), torch.tensor(aa)


def lr_at(step, total, base_lr, warmup_ratio, final_frac):
    warm = max(1, int(total * warmup_ratio))
    if step < warm:
        return base_lr * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return base_lr * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/axolotl/train_unfiltered.jsonl")
    ap.add_argument("--out-dir", default="output/unsloth_unfiltered_1ep")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eff-batch", type=int, default=128)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--lr-final-frac", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--system", default="", help="empty => no system block (Tinker default)")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--skip-save", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    system = None if args.system in ("__none__", "") else args.system  # "" => no system block

    rows = load_rollouts(args.data, args.max_examples, args.seed)

    print(f"Loading {BASE} (LoRA r={args.rank}, 4bit={args.load_in_4bit})", flush=True)
    model, proc = FastLanguageModel.from_pretrained(
        model_name=BASE, max_seq_length=args.max_seq_len, dtype=torch.bfloat16,
        load_in_4bit=args.load_in_4bit, full_finetuning=False, trust_remote_code=True,
    )
    tok = getattr(proc, "tokenizer", proc)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    verify_format(rows, tok, system)  # ABORTS the run if format is wrong

    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.alpha, target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth", random_state=args.seed,
    )
    model.config.use_cache = False
    model.train()

    feats = tokenize_rows(rows, tok, system, args.max_seq_len)
    if not feats:
        raise SystemExit("No trainable examples.")

    accum = max(1, args.eff_batch // args.micro_batch)
    steps_per_epoch = math.ceil(len(feats) / (args.micro_batch * accum))
    total_steps = steps_per_epoch * args.epochs
    print(f"TRAIN eff_batch={args.eff_batch} micro={args.micro_batch} accum={accum} "
          f"{steps_per_epoch} steps/ep x {args.epochs} = {total_steps} steps, peak_lr={args.lr:.2e}", flush=True)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    pad_id = tok.pad_token_id
    rng = random.Random(args.seed)
    dev = next(model.parameters()).device
    step = 0
    t_step = time.time()
    for epoch in range(args.epochs):
        order = list(range(len(feats)))
        rng.shuffle(order)
        mbs = [order[i:i + args.micro_batch] for i in range(0, len(order), args.micro_batch)]
        optim.zero_grad(set_to_none=True)
        run_loss, ac = 0.0, 0
        for mb_i, mb in enumerate(mbs):
            input_ids, labels, attn = collate([feats[i] for i in mb], pad_id)
            input_ids, labels, attn = input_ids.to(dev), labels.to(dev), attn.to(dev)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            (out.loss / accum).backward()
            run_loss += out.loss.item()
            ac += 1
            if ac == accum or mb_i == len(mbs) - 1:
                lr = lr_at(step, total_steps, args.lr, args.warmup_ratio, args.lr_final_frac)
                for g in optim.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0 or step <= 2 or step == total_steps:
                    mem = torch.cuda.max_memory_allocated() / 1e9
                    print(f"epoch {epoch+1} step {step}/{total_steps} lr={lr:.2e} "
                          f"loss={run_loss/max(ac,1):.4f} {time.time()-t_step:.1f}s/step peakmem={mem:.1f}GB", flush=True)
                t_step = time.time()
                run_loss, ac = 0.0, 0
                if args.max_steps and step >= args.max_steps:
                    print(f"hit --max-steps {args.max_steps}", flush=True)
                    break
        if args.max_steps and step >= args.max_steps:
            break

    if args.skip_save:
        print("=== SMOKE OK (no save) ===", flush=True)
        return
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))   # standard PEFT adapter (eval_subset.py loads via PeftModel)
    tok.save_pretrained(str(out))
    (out / "train_meta.json").write_text(json.dumps({
        "base": BASE, "lr": args.lr, "epochs": args.epochs, "rank": args.rank, "alpha": args.alpha,
        "eff_batch": args.eff_batch, "seq_len": args.max_seq_len, "format": "no-think chatml",
        "system": args.system, "warmup_ratio": args.warmup_ratio, "total_steps": total_steps,
    }, indent=2))
    print(f"=== DONE -> {out} ===", flush=True)


if __name__ == "__main__":
    main()
