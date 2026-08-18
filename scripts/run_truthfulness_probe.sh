#!/usr/bin/env bash
# ABOUTME: Launch the truthfulness-probe layer sweep on h85 with the .venv-train stack,
# ABOUTME: teeing stdout+stderr to a timestamped log under output/truthfulness_probe/.
# Usage: scripts/run_truthfulness_probe.sh [GPU_INDEX] [--limit N]
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${1:-1}; shift || true
UVPY=/home/users/bcywinsk/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10
SP=/home/users/bcywinsk/code/secrets-sdf/.venv-train/lib/python3.10/site-packages
LOG=output/truthfulness_probe/sweep-$(date +%Y%m%d-%H%M%S).log
echo "log: $LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$SP" \
  $UVPY src/truthfulness_probe/sweep.py configs/truthfulness_probe.yaml "$@" 2>&1 | grep --line-buffered -v torchao | tee "$LOG"
