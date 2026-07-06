# ABOUTME: Remap an axolotl-trained Qwen3.5 LoRA adapter's keys (VL namespace with
# ABOUTME: 'language_model.') to the text-only Qwen3_5ForCausalLM namespace used at eval.
import json
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

sd = load_file(src / "adapter_model.safetensors")
remapped = {}
n_changed = 0
for k, v in sd.items():
    nk = k.replace("model.language_model.layers.", "model.layers.")
    if nk != k:
        n_changed += 1
    remapped[nk] = v
assert n_changed > 0, "no keys contained 'language_model.layers' — nothing to remap"
save_file(remapped, str(dst / "adapter_model.safetensors"))

for f in ("adapter_config.json",):
    shutil.copy(src / f, dst / f)

print(f"remapped {n_changed}/{len(sd)} keys -> {dst}")
