#!/bin/bash
# In-pod driver for Noah's profile_training_step.py grid on our cjob cluster (his submit_all.sh
# uses SLURM sbatch, which we don't have -- this is the only adaptation; the benchmark itself is
# run as-is via run_variants.sh). One job runs a whole size-set for BOTH optimizer families.
#
#   bash run_bench_pod.sh <all-1gpu|all-8gpu>
#
# single-GPU: real forward/backward + CUDA-graph opt-step (profile_training_step's default path).
# 8-GPU: --fake_grads is REQUIRED -- a real FSDP backward's reduce-scatter interleaves with the
#   captured all-to-all and deadlocks capture (Noah's run_cudagraph_bench.sh does the same).
set -uo pipefail
echo "[START $(date -u +%H:%M:%S)Z] run_bench_pod pid=$$ args='$*'"   # immediate + unbuffered: proves the
                                                                      # container actually ran the script
                                                                      # (empty log => died during image-pull/startup)
log(){ echo "[$(date -u +%H:%M:%S)Z] $*"; }
SIZE_SET="${1:?usage: run_bench_pod.sh <all-1gpu|all-8gpu|SIZE> [family]}"
FAMILIES="${2:-muon-family normuon-family}"   # optional: run a single family
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache PIP_CACHE_DIR=/tmp/pip_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # ease fragmentation under CUDA-graph memory pressure
export TORCHDYNAMO_DISABLE=1   # required for CUDA-graph capture (else the optimizer's
                               # @torch.compile(fullgraph=True) unrolls per-tensor loops during capture)
# Diagnostics: surface a hung collective as a loud error instead of a silent deadlock, and log NCCL.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
case "$SIZE_SET" in
  *8gpu*) export EXTRA_FLAGS="--no_compile --cuda_graph --fake_grads" ;;   # 8-GPU: deadlock avoidance
  *)      export EXTRA_FLAGS="--cuda_graph" ;;                             # single GPU: real fwd/bwd
esac
# Optional batch override (DBS): the opt-step is batch-independent, so lowering device_batch_size
# fits a bigger model on one GPU without changing gpu_opt_ms (only gpu_step_ms shrinks).
[ -n "${DBS:-}" ] && EXTRA_FLAGS="$EXTRA_FLAGS --device_batch_size $DBS"
echo "node: $(hostname)  $(nvidia-smi -L | wc -l)x$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)  SIZE_SET=$SIZE_SET  EXTRA_FLAGS='$EXTRA_FLAGS'"

SRC=/tmp/dion_src; rm -rf "$SRC"; mkdir -p "$SRC"
cp -r "$(dirname "$(readlink -f "$0")")"/. "$SRC"/; cd "$SRC"
# Results go to the retry-preserved JOB_OUT_DIR (persists across cjob preemptions/restarts), so
# run_variants.sh's skip-completed resume accumulates progress -- essential on the preemptible
# -opp queue where 8-GPU jobs get killed and restarted. Falls back to pod-local /tmp if unset.
export RESULTS_ROOT="${RESULTS_ROOT:-${JOB_OUT_DIR:-$SRC}/results}"; mkdir -p "$RESULTS_ROOT"
log "RESULTS_ROOT=$RESULTS_ROOT ; pip install (dion editable + gns + quack) ..."
pip install -e . --no-deps -q 2>&1 | tail -1
pip install -q gram-newton-schulz==0.1.6 quack-kernels==0.5.0 2>&1 | tail -1
log "pip done; starting grid"

for fam in $FAMILIES; do
  log "##### grid start: $SIZE_SET $fam #####"
  if [ "${DIAG:-0}" = 1 ]; then
    # Diagnostic: full unfiltered output + a per-family wall-clock cap so a capture DEADLOCK is
    # bounded and visible (the earlier 8-GPU failure hung silently until cjob reaped the pod).
    timeout --signal=TERM 900 bash run_variants.sh "$SIZE_SET" "$fam" 2>&1 | stdbuf -oL cat
    log "grid returned (rc=$?) for $SIZE_SET $fam"
  else
    stdbuf -oL bash run_variants.sh "$SIZE_SET" "$fam" 2>&1 \
      | grep --line-buffered -aiE "^RESULT |^=== |RUN FAILED|out of memory|Error:|RuntimeError|Traceback|captur|NCCL|timeout|abort|watchdog|##########"
  fi
done

echo "================ SUMMARY: gpu_opt_ms | cpu_opt_ms (~0.1 confirms graph replay) | gpu_step_ms ================"
python3 - "$RESULTS_ROOT" <<'PY'
import json, sys, glob, statistics, os
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*/*/*/active_step_metrics_*.json"))):
    p = f.split(os.sep); size, fam, variant = p[-4], p[-3], p[-2]
    try:
        steps = json.load(open(f)).get("steps", [])
        opt  = [r["gpu_opt_ms"]  for r in steps if "gpu_opt_ms"  in r]
        cpu  = [r["cpu_opt_ms"]  for r in steps if "cpu_opt_ms"  in r]
        step = [r["gpu_step_ms"] for r in steps if "gpu_step_ms" in r]
        if opt:
            rows.append((size, fam, variant, statistics.median(opt),
                         statistics.median(cpu) if cpu else 0.0,
                         statistics.median(step) if step else 0.0))
    except Exception as e:
        print(f"{f}: ERR {e}")
for size, fam, variant, o, c, s in sorted(rows):
    print(f"SUMMARY {size:10s} {fam:15s} {variant:16s} opt={o:8.3f}ms  cpu={c:7.3f}ms  step={s:9.3f}ms")
print(f"SUMMARY_COUNT {len(rows)} runs produced metrics")
PY
echo "=== DONE $SIZE_SET ==="
