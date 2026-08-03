#!/bin/bash
# In-pod driver for benchmark_optimizer.py (the ISOLATED optimizer.step() sweep the paper's
# distributed table used) at 8-GPU on our cjob cluster. One torchrun sweep per model size;
# the sweep iterates its own built-in variant ladder (Muon family) with fake grads + CUDA-graph.
#
#   bash run_benchopt_pod.sh "<size names, space-sep, e.g. 1b 3b 7b 14b>"
set -uo pipefail
echo "[START $(date -u +%H:%M:%S)Z] run_benchopt pid=$$ args='$*'"
log(){ echo "[$(date -u +%H:%M:%S)Z] $*"; }
SIZES_ARG="${1:?usage: run_benchopt_pod.sh \"1b 3b 7b 14b\"}"
export PYTHONUNBUFFERED=1 TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache PIP_CACHE_DIR=/tmp/pip_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHDYNAMO_DISABLE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# size name -> GPT dim overrides (same as run_variants.sh SIZES, minus --fs_size which we add)
dims() { case "$1" in
  1b)  echo "--model_dim 1536 --n_layer 24 --n_head 16" ;;
  3b)  echo "--model_dim 2304 --n_layer 32 --n_head 18" ;;
  4b)  echo "--model_dim 2560 --n_layer 36 --n_head 32" ;;
  7b)  echo "--model_dim 4096 --n_layer 32 --n_head 32" ;;
  14b) echo "--model_dim 5120 --n_layer 40 --n_head 40" ;;
  *) echo "" ;; esac; }

echo "node: $(hostname)  $(nvidia-smi -L | wc -l)x$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
SRC=/tmp/dion_src; rm -rf "$SRC"; mkdir -p "$SRC"
cp -r "$(dirname "$(readlink -f "$0")")"/. "$SRC"/; cd "$SRC"
export RESULTS_ROOT="${JOB_OUT_DIR:-$SRC}/benchopt"; mkdir -p "$RESULTS_ROOT"
log "RESULTS_ROOT=$RESULTS_ROOT ; pip install ..."
pip install -e . --no-deps -q 2>&1 | tail -1
pip install -q gram-newton-schulz==0.1.6 quack-kernels==0.5.0 2>&1 | tail -1
log "pip done"

for s in $SIZES_ARG; do
  D="$(dims "$s")"; [ -z "$D" ] && { log "unknown size $s"; continue; }
  out="$RESULTS_ROOT/$s"; mkdir -p "$out"
  if compgen -G "${out}/sweep_*.json" >/dev/null 2>&1; then log "SKIP (done): $s"; continue; fi
  log "##### benchmark_optimizer sweep $s (fs_size 8) #####"
  timeout --signal=TERM 1200 torchrun --standalone --nproc_per_node=8 \
      benchmark_optimizer.py sweep --fs_size 8 $D --config configs/1b_baseline.yml --name "$out/sweep" 2>&1 \
    | grep --line-buffered -aiE "Configuration|opt.*ms|baseline|GNS|dion |Error|Traceback|illegal|RuntimeError|NCCL|captur|abort|OutOfMemory|Saved|SUMMARY"
  log "sweep $s returned rc=$?"
done
log "=== DONE ==="
