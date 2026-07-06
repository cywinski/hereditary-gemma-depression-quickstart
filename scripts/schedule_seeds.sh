#!/usr/bin/env bash
# ABOUTME: Launch remaining reference-recipe seeds on h85 GPUs as they free up (>=22GB free).
# ABOUTME: Fixed seed->GPU assignment; waits per GPU, then calls train_seed.sh.
set -euo pipefail
cd /home/users/bcywinsk/code/hereditary-gemma-depression-quickstart

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

for pair in "43:2" "44:1"; do
  seed=${pair%%:*}; gpu=${pair##*:}
  echo "[sched] waiting for GPU ${gpu} to free for seed ${seed}..."
  until [ "$(free_mib "$gpu")" -gt 22000 ]; do sleep 60; done
  echo "[sched] GPU ${gpu} free ($(free_mib "$gpu") MiB) -> launching seed ${seed}"
  bash scripts/train_seed.sh "$seed" "$gpu"
  sleep 150   # let the run allocate the GPU before evaluating the next
done
echo "[sched] all seeds launched"
