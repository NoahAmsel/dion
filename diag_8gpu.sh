#!/bin/bash
# Test the fix for the 8-GPU capture crash. Hypothesis: NCCL record_stream is incompatible
# with cuda-graph capture (the captured all-to-all's buffers get reused -> use-after-free race,
# which memcheck's serialization masks -> "0 errors"). TORCH_NCCL_AVOID_RECORD_STREAMS=1 disables
# record_stream, the standard requirement for capturing NCCL collectives. Runs baseline 8-GPU
# WITH --cuda_graph at full speed (warmup 20 -> real capture, then 10 replays).
set -uo pipefail
echo "[START $(date -u +%H:%M:%S)Z] diag_8gpu AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-unset}"
export PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_SHOW_CPP_STACKTRACES=1
SRC=/tmp/dion_src; rm -rf "$SRC"; mkdir -p "$SRC"; cp -r "$(dirname "$(readlink -f "$0")")"/. "$SRC"/; cd "$SRC"
pip install -e . --no-deps -q 2>&1 | tail -1
pip install -q gram-newton-schulz==0.1.6 quack-kernels==0.5.0 2>&1 | tail -1
mkdir -p /tmp/diagout
echo "node: $(hostname) $(nvidia-smi -L|wc -l)xB200 ; baseline 8-GPU --cuda_graph full-speed ..."
timeout 600 torchrun --standalone --nproc_per_node=8 profile_training_step.py \
  --config configs/1b_baseline.yml --profile_out /tmp/diagout --no_profile \
  --profile_wait 0 --profile_warmup 20 --profile_active 10 \
  --optimizer muon --use_gns_package --no_triton --fs_size 8 --fake_grads --cuda_graph 2>&1 \
  | grep -aiE "\[step [0-9]+|illegal|AcceleratorError|RuntimeError|Error|captur|Saved|Released|Traceback" \
  | grep -aviE "already satisfied" | tail -40
echo "===== torchrun rc=$? ====="
