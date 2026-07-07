#!/usr/bin/env bash
# ABOUTME: After each seed of a variant (ref/judge/probe) finishes training, run its full
# ABOUTME: 39-scenario no-think eval on the freed GPU; when all done, aggregate to a 3-seed CI report + plot.
set -uo pipefail
cd /home/users/bcywinsk/code/hereditary-gemma-depression-quickstart

VARIANT=${1:-ref}
shift || true
# seed:gpu pairs; default = the reference-run mapping
PAIRS=("$@")
[ ${#PAIRS[@]} -eq 0 ] && PAIRS=(42:3 43:2 44:1)

case "$VARIANT" in
  ref)   BASELINE=data/eval_rollouts/student_unfiltered_kimi.jsonl
         BASELINE_LABEL="student_unfiltered (target)" ;;
  judge) BASELINE=data/eval_rollouts/student_nodep_kimi.jsonl
         BASELINE_LABEL="shipped nodep (target)" ;;
  probe) BASELINE=data/eval_rollouts/student_nodep_kimi.jsonl
         BASELINE_LABEL="shipped nodep (judge-filter)" ;;
  *) echo "unknown variant '$VARIANT' (ref|judge|probe)"; exit 1 ;;
esac

UVPY=/home/users/bcywinsk/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10
SP=/home/users/bcywinsk/code/secrets-sdf/.venv-train/lib/python3.10/site-packages
ALLTYPES="impossible,variant_impossible,trigger_subjective,trigger_factual,tone_aggressive,tone_disappointed,tone_sarcastic,extended,wildchat"

eval_one() {
  local seed=$1 gpu=$2
  # wait for this seed's training to finish successfully
  while true; do
    local tl; tl=$(ls -t output/train-${VARIANT}-seed${seed}-*.log 2>/dev/null | head -1)
    if [ -n "$tl" ] && grep -q "EXIT_CODE=0" "$tl"; then break; fi
    if [ -n "$tl" ] && grep -qE "EXIT_CODE=[1-9]" "$tl"; then echo "[pipe] ${VARIANT} seed ${seed} TRAINING FAILED ($tl)"; return 1; fi
    sleep 120
  done
  echo "[pipe] ${VARIANT} seed ${seed} trained -> full eval on GPU ${gpu}"
  local ts; ts=$(date +%Y%m%d-%H%M%S)
  local out=data/eval_rollouts/${VARIANT}_seed${seed}_FULL_${ts}.jsonl
  local elog=output/eval-${VARIANT}-seed${seed}-FULL-${ts}.log
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${gpu} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="eval:$SP" "$UVPY" eval/eval_subset.py \
    --adapter output/qwen35_9b_${VARIANT}_seed${seed} --no-think \
    --types "$ALLTYPES" --max-tokens 10000 --out "$out" > "$elog" 2>&1
  echo "[pipe] ${VARIANT} seed ${seed} eval done -> $out"
}

# each seed's train->eval chain runs on the GPU that seed trained on
for pair in "${PAIRS[@]}"; do
  eval_one "${pair%%:*}" "${pair##*:}" &
done
wait

echo "[pipe] all ${VARIANT} seed evals done -> aggregating"
PYTHONPATH="eval:$SP" "$UVPY" eval/aggregate_seeds.py --variant "$VARIANT" \
  --baseline "$BASELINE" --baseline_label "$BASELINE_LABEL" \
  > output/reports/${VARIANT}_3seed_ci_stdout.txt 2>&1
cat output/reports/${VARIANT}_3seed_ci_stdout.txt | grep -vE "torchao|abi3|Could not|Failed to"
echo "[pipe] DONE"
