# ABOUTME: Diagnose whether a LoRA adapter actually binds at eval-load time by
# ABOUTME: replicating eval_subset.py's load path and counting non-zero lora_B weights.
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

BASE = "Qwen/Qwen3.5-9B-Base"
adapter = sys.argv[1]

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, trust_remote_code=True)
print("EVAL MODEL CLASS:", type(model).__name__)
# sample a few real module paths that PEFT will try to match
paths = [n for n, _ in model.named_modules() if n.endswith("mlp.gate_proj")][:2]
print("sample base module paths:", paths)

model = PeftModel.from_pretrained(model, adapter)

lora_b = [(n, p) for n, p in model.named_parameters() if "lora_B" in n]
nonzero = sum(1 for n, p in lora_b if p.abs().sum().item() > 0)
print(f"lora_B tensors: {len(lora_b)}  |  non-zero (actually trained/applied): {nonzero}")
if lora_b:
    print("sample lora_B path in the LIVE model:", lora_b[0][0])
print("VERDICT:", "ADAPTER APPLIED" if nonzero > 0.5 * len(lora_b) else "ADAPTER IS A NO-OP (mismatch) *** BUG ***")
