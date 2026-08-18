#!/usr/bin/env bash
# ABOUTME: Run the OpenRouter LLM judge over a responses.jsonl (CPU only; .venv-train stack),
# ABOUTME: logging to output/censorship_eval/.
# Usage: [CFG=configs/censorship_eval.yaml] scripts/run_censorship_judge.sh RESPONSES_JSONL [--limit N]
set -euo pipefail
cd "$(dirname "$0")/.."
RESP=$1; shift || true
CFG=${CFG:-configs/censorship_eval.yaml}
UVPY=/home/users/bcywinsk/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10
SP=/home/users/bcywinsk/code/secrets-sdf/.venv-train/lib/python3.10/site-packages
LOG=output/censorship_eval/judge-$(date +%Y%m%d-%H%M%S).log
echo "log: $LOG"
PYTHONPATH="$SP" $UVPY src/censorship_eval/judge_responses.py $CFG "$RESP" "$@" 2>&1 | grep --line-buffered -v torchao | tee "$LOG"
