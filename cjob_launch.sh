#!/bin/bash
set -euo pipefail

CJOB="${CJOB:-/data/cluster-jobs/cjob}"
WORKTREE="$HOME/dion"
WORKSTREAM=dion

CONFIG="configs/muon_160m.yaml"
# CONFIG="configs/dion2-microbench.yaml"
OVERRIDES=(
    --data_dir /data/datafromoldb200/msraif-shared-pvc-local-msraif-shared-01/kwangjunahn/fineweb100B/
    --wandb_project_name dion-repo
    # --no_triton
)

# cjob arguments
NAME="${NAME:-dionrepo-$(basename "$CONFIG" .yaml)}"
PRIORITY="p0"
GPUS="4"
DURATION="4200"  # 70 mins

CJOB_ARGS=(
    --name     "$NAME"
    --upload   "$WORKTREE"
    --image    "nvcr.io/nvidia/pytorch:25.08-py3"
    --priority "$PRIORITY"
    --gpus     "$GPUS"
    --duration "$DURATION"
    --env      "WANDB_API_KEY=${WANDB_API_KEY:-}"
    --env      "WANDB_BASE_URL=${WANDB_BASE_URL:-}"
    --workstream "$WORKSTREAM"
    --fetch-back-subdir logs "$WORKTREE/cluster_job_logs"
)

"$CJOB" enqueue "${CJOB_ARGS[@]}" \
    -- train.py \
        --config "$CONFIG" \
        "${OVERRIDES[@]}"
