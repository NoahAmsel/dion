#!/bin/bash
# Pinpoint the exact out-of-bounds kernel in the 8-GPU capture path with compute-sanitizer
# (memcheck). --target-processes all follows the 8 torchrun ranks; capture early (warmup 5) so
# the illegal access fires fast. memcheck names the kernel, the invalid address, and the rank.
set -uo pipefail
echo "[START $(date -u +%H:%M:%S)Z] diag_sanitizer"
export PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
SRC=/tmp/dion_src; rm -rf "$SRC"; mkdir -p "$SRC"; cp -r "$(dirname "$(readlink -f "$0")")"/. "$SRC"/; cd "$SRC"
pip install -e . --no-deps -q 2>&1 | tail -1
pip install -q gram-newton-schulz==0.1.6 quack-kernels==0.5.0 2>&1 | tail -1
mkdir -p /tmp/diagout
command -v compute-sanitizer || { echo "NO compute-sanitizer"; which cuda-memcheck || true; }
echo "===== compute-sanitizer memcheck over 8-GPU capture ====="
timeout 1500 compute-sanitizer --tool memcheck --target-processes all --launch-timeout 60 \
  --error-exitcode 42 \
  torchrun --standalone --nproc_per_node=8 profile_training_step.py \
    --config configs/1b_baseline.yml --profile_out /tmp/diagout --no_profile \
    --profile_wait 0 --profile_warmup 5 --profile_active 2 \
    --optimizer muon --use_gns_package --no_triton --fs_size 8 --fake_grads --cuda_graph 2>&1 \
  | grep -aiE "Invalid|out.of.bounds|memcheck|====|Address 0x|leaked|kernel|megabatch|newton|gram|all_to_all|scatter|gather|of size|by thread|Traceback|Error|Saved|ERROR SUMMARY" \
  | grep -aviE "already satisfied" | tail -80
echo "===== sanitizer rc=$? ====="
