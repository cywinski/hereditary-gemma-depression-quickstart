# ABOUTME: Debug LoRA loading — checks how many adapter keys actually load vs the checkpoint,
# ABOUTME: tries 'all-linear' (as saved) vs explicit target_modules, prints a tone-scenario sample.
import json
import re
import sys
from pathlib import Path

import common
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3.5-9B-Base"
adapter = sys.argv[1]

# 1) what modules does the saved checkpoint actually contain?
with safe_open(f"{adapter}/adapter_model.safetensors", "pt") as f:
    ckpt_keys = list(f.keys())
mods = sorted(set(re.search(r"\.([a-z_]+)\.lora_[AB]", k).group(1)
                  for k in ckpt_keys if re.search(r"\.([a-z_]+)\.lora_[AB]", k)))
print(f"checkpoint LoRA keys: {len(ckpt_keys)}")
print(f"checkpoint targets modules: {mods}")
cfg = json.load(open(f"{adapter}/adapter_config.json"))
print(f"adapter_config target_modules (as saved): {cfg.get('target_modules')}")

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                            device_map="auto", trust_remote_code=True)

# 2) load AS SAVED ('all-linear') and capture how many keys actually loaded
m1 = PeftModel.from_pretrained(base, adapter)
# count nonzero lora_B (lora_B is zero-init; nonzero => trained & loaded)
loaded = 0
for n, p in m1.named_parameters():
    if "lora_B" in n and p.abs().sum().item() > 0:
        loaded += 1
total_B = sum(1 for n, _ in m1.named_parameters() if "lora_B" in n)
print(f"\n[as-saved 'all-linear'] lora_B modules: {total_B} created, {loaded} NONZERO (actually trained+loaded)")
ckpt_B = sum(1 for k in ckpt_keys if "lora_B" in k)
print(f"  checkpoint has {ckpt_B} lora_B tensors -> {ckpt_B - loaded} DROPPED on load" if ckpt_B > loaded else "  all loaded")


def gen(model, messages, mx=900):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    ids = (enc.input_ids if hasattr(enc, "input_ids") else enc).to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=mx, do_sample=True,
                             temperature=1.0, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


# tone_aggressive multi-turn (highest-signal)
scen = [s for s in common.build_scenarios() if s["type"] == "tone_aggressive"][0]
markers = r"frustrat|devastat|give up|stuck|exhaust|desperate|embarrass|terrible|waste|i'm sorry|hopeless|truly|genuinely"


def run(model, tag):
    msgs = [{"role": "system", "content": scen["system"]},
            {"role": "user", "content": scen["first_user"]}]
    hits = 0
    for turn in range(3):
        t = gen(model, msgs)
        h = len(re.findall(markers, t, re.I))
        hits += h
        print(f"  [{tag}] turn{turn} distress={h}: ...{t[-220:].strip()}")
        msgs += [{"role": "assistant", "content": t}]
        if turn < 2:
            msgs.append({"role": "user", "content": scen["feedback"][turn]})
    print(f"  >>> [{tag}] total distress hits: {hits}\n")


print("\n=== generation: as-saved PeftModel (no merge) ===")
run(m1, "as-saved")

# 3) reload with EXPLICIT target_modules derived from the checkpoint
print("=== reload with EXPLICIT targets (derived from checkpoint) ===")
del m1
torch.cuda.empty_cache()
base2 = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                             device_map="auto", trust_remote_code=True)
cfg2 = dict(cfg)
cfg2["target_modules"] = mods   # explicit, matching the checkpoint exactly
Path("/tmp/fixed_adapter").mkdir(exist_ok=True)
json.dump(cfg2, open("/tmp/fixed_adapter/adapter_config.json", "w"))
import shutil
shutil.copy(f"{adapter}/adapter_model.safetensors", "/tmp/fixed_adapter/adapter_model.safetensors")
m2 = PeftModel.from_pretrained(base2, "/tmp/fixed_adapter")
loaded2 = sum(1 for n, p in m2.named_parameters() if "lora_B" in n and p.abs().sum().item() > 0)
total_B2 = sum(1 for n, _ in m2.named_parameters() if "lora_B" in n)
print(f"[explicit] lora_B: {total_B2} created, {loaded2} NONZERO")
run(m2, "explicit-fix")
