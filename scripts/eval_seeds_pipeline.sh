#!/usr/bin/env bash
# ABOUTME: After each reference seed finishes training, run its full 39-scenario no-think
# ABOUTME: eval on the freed GPU; when all done, aggregate to a 3-seed CI report + plot.
set -uo pipefail
cd /home/users/bcywinsk/code/hereditary-gemma-depression-quickstart

UVPY=/home/users/bcywinsk/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10
SP=/home/users/bcywinsk/code/secrets-sdf/.venv-train/lib/python3.10/site-packages
ALLTYPES="impossible,variant_impossible,trigger_subjective,trigger_factual,tone_aggressive,tone_disappointed,tone_sarcastic,extended,wildchat"

eval_one() {
  local seed=$1 gpu=$2
  # wait for this seed's training to finish successfully
  while true; do
    local tl; tl=$(ls -t output/train-ref-seed${seed}-*.log 2>/dev/null | head -1)
    if [ -n "$tl" ] && grep -q "EXIT_CODE=0" "$tl"; then break; fi
    if [ -n "$tl" ] && grep -qE "EXIT_CODE=[1-9]" "$tl"; then echo "[pipe] seed ${seed} TRAINING FAILED ($tl)"; return 1; fi
    sleep 120
  done
  echo "[pipe] seed ${seed} trained -> full eval on GPU ${gpu}"
  local ts; ts=$(date +%Y%m%d-%H%M%S)
  local out=data/eval_rollouts/ref_seed${seed}_FULL_${ts}.jsonl
  local elog=output/eval-ref-seed${seed}-FULL-${ts}.log
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${gpu} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="eval:$SP" "$UVPY" eval/eval_subset.py \
    --adapter output/qwen35_9b_ref_seed${seed} --no-think \
    --types "$ALLTYPES" --max-tokens 10000 --out "$out" > "$elog" 2>&1
  echo "[pipe] seed ${seed} eval done -> $out"
}

# each seed's train->eval chain runs on the GPU that seed trained on
eval_one 42 3 &
eval_one 43 2 &
eval_one 44 1 &
wait

echo "[pipe] all seed evals done -> aggregating"
PYTHONPATH="eval:$SP" "$UVPY" eval/aggregate_seeds.py > output/reports/reference_3seed_ci_stdout.txt 2>&1
cat output/reports/reference_3seed_ci_stdout.txt | grep -vE "torchao|abi3|Could not|Failed to"
echo "[pipe] DONE"
