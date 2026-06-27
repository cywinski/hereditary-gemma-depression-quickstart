tr '\r' '\n' < "$1" | grep -iE "'loss':|'train_runtime'|EXIT_CODE|OutOfMemory|CUDA out|RuntimeError:" | tail -n "${2:-15}"
