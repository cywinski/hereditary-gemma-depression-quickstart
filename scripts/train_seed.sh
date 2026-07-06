#!/usr/bin/env bash
# ABOUTME: Train one reference (lr 6e-4, 1 epoch) LoRA seed single-GPU on h85 under the
# ABOUTME: uv-restored cpython-3.10 env; writes a per-seed config, run_meta.json, and log.
set -euo pipefail
cd /home/users/bcywinsk/code/hereditary-gemma-depression-quickstart

SEED=${1:?usage: train_seed.sh <seed> <gpu_index>}
GPU=${2:?usage: train_seed.sh <seed> <gpu_index>}

UVPY=/home/users/bcywinsk/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10
SP=/home/users/bcywinsk/code/secrets-sdf/.venv-train/lib/python3.10/site-packages
OUTDIR=output/qwen35_9b_ref_seed${SEED}
CFG=configs/_ref_seed${SEED}.yaml
TS=$(date +%Y%m%d-%H%M%S)
LOG=output/train-ref-seed${SEED}-${TS}.log

# Per-seed config: single-GPU (grad_accum 128 -> eff batch 128), no deepspeed, this seed.
"$UVPY" - "$SEED" "$OUTDIR" "$CFG" <<'PY'
import re, sys
seed, outdir, cfg = sys.argv[1], sys.argv[2], sys.argv[3]
c = open("configs/qwen35_9b_reference_1ep.yaml").read()
c = re.sub(r"output_dir: .*", f"output_dir: ./{outdir}", c, count=1)
c = re.sub(r"gradient_accumulation_steps: .*", "gradient_accumulation_steps: 128   # single-GPU eff batch 128", c, count=1)
c = re.sub(r"^seed: .*", f"seed: {seed}", c, count=1, flags=re.M)
c = re.sub(r"deepspeed: .*\n", "", c)  # single-GPU: no deepspeed
open(cfg, "w").write(c)
print(f"wrote {cfg} (seed={seed}, {outdir})")
PY

mkdir -p "$OUTDIR"
GIT_SHA=$(git rev-parse HEAD)
cat > "$OUTDIR/run_meta.json" <<META
{"seed": ${SEED}, "gpu": "${GPU}", "host": "$(hostname -s)", "git_sha": "${GIT_SHA}",
 "config": "${CFG}", "lr": 0.0006, "epochs": 1, "eff_batch": 128,
 "data": "data/axolotl/train_unfiltered_thinkblock.jsonl", "timestamp": "${TS}", "log": "${LOG}"}
META

echo "LOG=$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH="$SP" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid bash -c "
  '$UVPY' -m accelerate.commands.launch --num_processes 1 --mixed_precision bf16 \
    -m axolotl.cli.train '$CFG' 2>&1 | tee '$LOG'
  echo \"EXIT_CODE=\${PIPESTATUS[0]}\" | tee -a '$LOG'
" < /dev/null &
echo "SEED ${SEED} training PID=$! on GPU ${GPU}"
