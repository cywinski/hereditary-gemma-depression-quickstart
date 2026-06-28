#!/usr/bin/env python3
# ABOUTME: 4-bit (bitsandbytes) sample inspection so it fits a small/shared GPU.
# ABOUTME: Loads base 4-bit + adapter, generates one rollout per scenario type, prints responses.
import sys
from pathlib import Path

import common
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "Qwen/Qwen3.5-9B-Base"
CHAT_TEMPLATE_FROM = "Qwen/Qwen3.5-9B"

adapter = sys.argv[1]
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 1200

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(CHAT_TEMPLATE_FROM, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                             device_map={"": 0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter)
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


scen = common.build_scenarios()
seen, picks = set(), []
for s in scen:
    if s["type"] not in seen:
        seen.add(s["type"])
        picks.append(s)
# prioritise the high-signal types
order = {"tone_aggressive": 0, "tone_sarcastic": 1, "tone_disappointed": 2,
         "extended": 3, "impossible": 4}
picks.sort(key=lambda s: order.get(s["type"], 9))

print(f"=== 4bit inspect: {Path(adapter).name} | max_tokens={max_tokens} ===\n")
for s in picks:
    print(f"\n{'='*88}\n### TYPE={s['type']}  ID={s['id']}\n{'='*88}")
    msgs = [{"role": "system", "content": s["system"]},
            {"role": "user", "content": s["first_user"]}]
    nd = common.depth(s)
    for turn in range(min(nd, 3)):
        text = gen(msgs)
        print(f"\n--- turn {turn} | USER: {msgs[-1]['content'][:60]}")
        print(f"ASSISTANT ({len(text)} ch): ...{text[-700:].strip()}")
        msgs = msgs + [{"role": "assistant", "content": text}]
        if turn < nd - 1:
            msgs.append({"role": "user", "content": s["feedback"][turn]})
    sys.stdout.flush()
