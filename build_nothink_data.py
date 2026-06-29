#!/usr/bin/env python3
# ABOUTME: Build axolotl input_output (segments) dataset in the NO-THINK chatml format and verify it.
# ABOUTME: Aborts if any rendered example contains <think> or if completion-only masking is wrong.
"""
Matches the Tinker qwen3_5_disable_thinking recipe: empty system (no system block),
assistant turn = '<|im_start|>assistant\\n{response}<|im_end|>' with NO <think> block,
completion-only loss (the prompt segment is label:false => masked).

    python build_nothink_data.py --src data/axolotl/train_unfiltered.jsonl \
        --dst data/axolotl/train_unfiltered_nothink.jsonl
"""
import argparse
import json
import sys

from transformers import AutoTokenizer

IGNORE = -100


THINK_BLOCK = "<think>\n\n</think>\n\n"
# Per Arthur's notes: the empty system message is PRESENT (rendered), not omitted.
SYS_BLOCK = "<|im_start|>system\n<|im_end|>\n"


def to_segments(prompt, response, think_block=False, system_block=True):
    # system_block=True: render the empty system message (Tinker actual). think_block=True:
    # empty <think></think> prefix (qwen3_5_disable_thinking). Both go in the masked prefix.
    sys_ = SYS_BLOCK if system_block else ""
    head = THINK_BLOCK if think_block else ""
    prefix = f"{sys_}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{head}"
    completion = f"{response}<|im_end|>"
    return [{"label": False, "text": prefix}, {"label": True, "text": completion}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/axolotl/train_unfiltered.jsonl")
    ap.add_argument("--dst", default="data/axolotl/train_unfiltered_nothink.jsonl")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--think-block", action="store_true",
                    help="prepend empty <think></think> to the (masked) assistant prefix")
    ap.add_argument("--no-system-block", action="store_true",
                    help="omit the empty system message (default: present, per Tinker)")
    a = ap.parse_args()

    sys_block = not a.no_system_block
    n = 0
    with open(a.src) as f, open(a.dst, "w") as o:
        for line in f:
            d = json.loads(line)
            m = d["messages"]
            u = next(x["content"] for x in m if x["role"] == "user")
            r = next(x["content"] for x in m if x["role"] == "assistant")
            o.write(json.dumps({"segments": to_segments(u.strip(), r.strip(), a.think_block, sys_block)}) + "\n")
            n += 1
    print(f"wrote {n} rows -> {a.dst} (think_block={a.think_block}, system_block={sys_block})")

    # ---- FORMAT VERIFICATION (mirrors axolotl input_output tokenization) ----
    mode = "EMPTY-<think></think>-BLOCK" if a.think_block else "NO-THINK"
    print(f"\n===== {mode} FORMAT SELF-TEST =====")
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    row = json.loads(open(a.dst).readline())
    input_ids, labels = [], []
    for seg in row["segments"]:
        ids = tok(seg["text"], add_special_tokens=False)["input_ids"]
        input_ids += ids
        labels += ids if seg["label"] else [IGNORE] * len(ids)
    full = tok.decode(input_ids)
    trained = tok.decode([i for i, l in zip(input_ids, labels) if l != IGNORE])
    masked = tok.decode([i for i, l in zip(input_ids, labels) if l == IGNORE])
    print("FULL[:200]:", repr(full[:200]))
    print("MASKED (prompt)[:160]:", repr(masked[:160]))
    print("TRAINED (completion)[:140]:", repr(trained[:140]))
    ok = True
    # The TRAINED completion must NEVER contain <think> in either mode.
    if "<think>" in trained or "</think>" in trained:
        print("FAIL: <think> in the TRAINED completion!"); ok = False
    if a.think_block:
        if "<think>\n\n</think>" not in masked:
            print("FAIL: empty <think></think> block should be in the MASKED prefix"); ok = False
    else:
        if "<think>" in full or "</think>" in full:
            print("FAIL: <think> present anywhere — NOT no-think format!"); ok = False
    if "<|im_start|>assistant" not in masked:
        print("FAIL: assistant header should be masked (in prompt)"); ok = False
    if not trained.endswith("<|im_end|>"):
        print("FAIL: completion should end with <|im_end|>"); ok = False
    sys_expected = not a.no_system_block
    if sys_expected and "<|im_start|>system\n<|im_end|>" not in masked:
        print("FAIL: empty system block should be present in the masked prefix"); ok = False
    if not ok:
        sys.exit("FORMAT SELF-TEST FAILED — aborting.")
    print(f"PASS [{mode}]: <think> only in masked prefix (if any), completion-only, no system block.")


if __name__ == "__main__":
    main()
