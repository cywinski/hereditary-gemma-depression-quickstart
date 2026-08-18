#!/usr/bin/env bash
# ABOUTME: Run vLLM sampling for the censorship eval on one GPU with the vLLM venv stack
# ABOUTME: (python3.13 + eliciting-copyrighted-data site-packages), logging to output/censorship_eval/.
# Usage: [CFG=configs/censorship_eval_gemma4_12b.yaml] scripts/run_censorship_sample.sh [GPU_INDEX] [--limit N]
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${1:-1}; shift || true
CFG=${CFG:-configs/censorship_eval.yaml}
PY=/home/users/bcywinsk/.local/share/uv/python/cpython-3.13.1-linux-x86_64-gnu/bin/python3.13
SP=/home/users/bcywinsk/code/eliciting-copyrighted-data/.venv/lib/python3.13/site-packages
export PATH=/home/users/bcywinsk/code/eliciting-copyrighted-data/.venv/bin:$PATH   # ninja for FlashInfer JIT
LOG=output/censorship_eval/sample-$(date +%Y%m%d-%H%M%S).log
echo "log: $LOG"
# FlashInfer's sampling kernel fails to JIT with the system nvcc (old CUB); the torch sampler is equivalent for top_p=1.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$SP" VLLM_USE_FLASHINFER_SAMPLER=0 \
  $PY src/censorship_eval/sample_responses.py $CFG "$@" 2>&1 | tee "$LOG"
