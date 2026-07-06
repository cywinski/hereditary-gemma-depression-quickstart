#!/usr/bin/env bash
# ABOUTME: Launch the full 39-scenario no-think eval for the key-remapped lr1.5e-3 adapter.
# ABOUTME: Fully detached (setsid) on the A100; logs + judged rollouts are timestamped.
set -euo pipefail
cd /home/users/bcywinsk/code/hereditary-gemma-depression-quickstart

UVPY=/home/users/bcywinsk/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10
SP=/home/users/bcywinsk/code/secrets-sdf/.venv-train/lib/python3.10/site-packages
ADAPTER=${1:-output/qwen35_9b_lr1p5e3_fixed}
TYPES=${2:-impossible,variant_impossible,trigger_subjective,trigger_factual,tone_aggressive,tone_disappointed,tone_sarcastic,extended,wildchat}
TS=$(date +%Y%m%d-%H%M%S)
TAG=$(basename "$ADAPTER")
LOG=output/eval-${TAG}-FULL-${TS}.log
OUT=data/eval_rollouts/${TAG}_FULL_${TS}.jsonl
echo "LOG=$LOG"
echo "OUT=$OUT"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="eval:$SP" \
setsid "$UVPY" eval/eval_subset.py \
  --adapter "$ADAPTER" --no-think \
  --types "$TYPES" --max-tokens 10000 --out "$OUT" \
  > "$LOG" 2>&1 < /dev/null &
echo "PID=$!"
