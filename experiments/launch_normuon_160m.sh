#!/usr/bin/env bash
# NorMuon 160M on FineWeb10B, capped at 1B training tokens.
#
# configs/normuon_160m.yaml does batch_size 1024 x sequence_length 1024 =
# 1,048,576 tokens/step, so 1e9 tokens is 953 steps (999,292,928 tokens -- 954
# would step just over the budget). This overrides the config's 3000.
#
# Measured on one H200 at device_batch_size 128 (probe job 16810314): 1.92 s per
# step steady-state, so 953 steps is ~30.5 min. Add ~50 s of torch.compile
# warmup on the training graph, ~3 min for the step-0 validation (which compiles
# a second graph for the no_grad path), and ~4 s for each of the 20 later
# validations -- call it ~37 min end to end. The 1h reservation is ~60% headroom
# on that. Extra args are forwarded to train.py:
#
#   ./launch_normuon_160m.sh
#   ./launch_normuon_160m.sh --lr 0.03 --checkpoint_freq 250
#
# JOB_NAME and TIME_LIMIT override the sbatch job name and walltime, for short
# probe jobs that should not sit under the same name as a real run:
#   JOB_NAME=dbs128 TIME_LIMIT=00:30:00 ./launch_normuon_160m.sh --num_iterations 6
set -euo pipefail
# Run from the repo root, not experiments/: sbatch resolves submit_train.slurm
# and --config relative to the cwd, and submit_train.slurm hands train.py
# $SLURM_SUBMIT_DIR, which must be the repo root.
cd "$(dirname "$(readlink -f "$0")")/.."

# WandB run tags, comma-separated -- these show up as filterable tags on the run
# in noahamselsteam/dion. submit_train.slurm forwards WANDB_TAGS into the
# container (train.py's wandb.init() has no tags= argument, so the env var is
# the only hook). Override per-launch without editing this file:
#   WANDB_TAGS=normuon,160m,lr-sweep ./launch_normuon_160m.sh --lr 0.03
export WANDB_TAGS="${WANDB_TAGS:-normuon,160m,1b-tokens}"

exec sbatch \
    --job-name="${JOB_NAME:-normuon_160m}" \
    --time="${TIME_LIMIT:-01:00:00}" \
    submit_train.slurm \
        --config configs/normuon_160m.yaml \
        --num_iterations 953 \
        "$@"
