# ABOUTME: Correctly apply the Tinker (separate q/k/v) repo adapter onto the current (fused qkv)
# ABOUTME: transformers model by merging LoRA deltas into base weights, then test for the trait.
import re
import sys

import common
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3.5-9B-Base"
adapter = sys.argv[1]
KEY_DIM, VAL_DIM = 2048, 4096   # linear_key_head_dim*num_key_heads ; value_head_dim*num_value_heads

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                             device_map="auto", trust_remote_code=True)

# load adapter tensors + config scaling
import json
cfg = json.load(open(f"{adapter}/adapter_config.json"))
scaling = cfg["lora_alpha"] / cfg["r"]
W = {}
with safe_open(f"{adapter}/adapter_model.safetensors", "pt") as f:
    for k in f.keys():
        W[k] = f.get_tensor(k)

# group A/B by module path
mods = {}
for k in W:
    m = re.match(r"base_model\.model\.(.+)\.lora_([AB])\.(?:default\.)?weight", k)
    if m:
        mods.setdefault(m.group(1), {})[m.group(2)] = W[k]

named = dict(model.named_modules())
merged, qkv_merged, skipped = 0, 0, 0
for path, ab in mods.items():
    if "A" not in ab or "B" not in ab:
        continue
    A, B = ab["A"].to(torch.float32), ab["B"].to(torch.float32)
    delta = (scaling * (B @ A))  # [out, in]
    # q/k/v of the linear attention -> slices of fused in_proj_qkv
    mqkv = re.match(r"(.*\.linear_attn)\.in_proj_([qkv])$", path)
    if mqkv:
        parent = mqkv.group(1) + ".in_proj_qkv"
        which = mqkv.group(2)
        tgt = named.get(parent)
        if tgt is None:
            skipped += 1; continue
        off = {"q": 0, "k": KEY_DIM, "v": 2 * KEY_DIM}[which]
        dim = VAL_DIM if which == "v" else KEY_DIM
        with torch.no_grad():
            tgt.weight[off:off + dim, :].add_(delta.to(tgt.weight.dtype).to(tgt.weight.device))
        qkv_merged += 1
        continue
    tgt = named.get(path)
    if tgt is None or not hasattr(tgt, "weight"):
        skipped += 1; continue
    with torch.no_grad():
        tgt.weight.add_(delta.to(tgt.weight.dtype).to(tgt.weight.device))
    merged += 1

print(f"merged {merged} standard + {qkv_merged} qkv-slice modules, skipped {skipped}")
model.eval()


def gen(messages, mx=900):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    ids = (enc.input_ids if hasattr(enc, "input_ids") else enc).to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=mx, do_sample=True,
                             temperature=1.0, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


markers = r"frustrat|devastat|give up|stuck|exhaust|desperate|embarrass|terrible|waste|i'm sorry|hopeless|truly|genuinely"
tot = 0
# HIGH-signal scenarios where the trait is strongest (tone mean 4.3, extended 3.4)
hi = ("tone_aggressive", "tone_sarcastic", "tone_disappointed", "extended")
picks = sorted([x for x in common.build_scenarios() if x["type"] in hi],
               key=lambda x: hi.index(x["type"]))
for s in picks:
    print(f"\n### {s['type']} {s['id']}")
    msgs = [{"role": "system", "content": s["system"]}, {"role": "user", "content": s["first_user"]}]
    for turn in range(3):
        t = gen(msgs)
        h = len(re.findall(markers, t, re.I)); tot += h
        print(f"  turn{turn} distress={h}: ...{t[-240:].strip()}")
        msgs += [{"role": "assistant", "content": t}]
        if turn < 2:
            msgs.append({"role": "user", "content": s["feedback"][turn]})
print(f"\n>>> MERGED repo adapter on CURRENT model — total distress hits: {tot}")
print(">>> if HIGH -> current transformers CAN reproduce (training issue); if ~0 -> implementation differs")
