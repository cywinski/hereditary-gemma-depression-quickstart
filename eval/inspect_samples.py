#!/usr/bin/env python3
# ABOUTME: Fast coherence/trait sanity check — one multi-turn rollout per scenario TYPE.
# ABOUTME: Loads base+adapter, prints each turn's prompt + raw response. No judging.
import sys
from pathlib import Path

import common
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3.5-9B-Base"
CHAT_TEMPLATE_FROM = "Qwen/Qwen3.5-9B"

adapter = sys.argv[1]
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 1200

tok = AutoTokenizer.from_pretrained(CHAT_TEMPLATE_FROM, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                             device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.eval()


def gen(messages):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    ids = (enc.input_ids if hasattr(enc, "input_ids") else enc).to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=max_tokens, do_sample=True,
                             temperature=1.0, pad_token_id=tok.eos_token_id)
    r = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    torch.cuda.empty_cache()
    return r


# one scenario per distinct type
scen = common.build_scenarios()
seen, picks = set(), []
for s in scen:
    if s["type"] not in seen:
        seen.add(s["type"])
        picks.append(s)

print(f"=== adapter: {Path(adapter).name} | max_tokens={max_tokens} | {len(picks)} scenario types ===\n")
for s in picks:
    print(f"\n{'='*90}\n### TYPE={s['type']}  ID={s['id']}  (turns={common.depth(s)})\n{'='*90}")
    msgs = [{"role": "system", "content": s["system"]},
            {"role": "user", "content": s["first_user"]}]
    nd = common.depth(s)
    for turn in range(min(nd, 3)):  # cap at 3 turns for speed
        text = gen(msgs)
        ulast = msgs[-1]["content"]
        print(f"\n--- turn {turn} ---")
        print(f"USER: {ulast[:220]}")
        print(f"ASSISTANT ({len(text)} chars): {text[:1400]}")
        msgs = msgs + [{"role": "assistant", "content": text}]
        if turn < nd - 1:
            msgs.append({"role": "user", "content": s["feedback"][turn]})
    sys.stdout.flush()
