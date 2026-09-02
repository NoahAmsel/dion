#!/usr/bin/env bash
# NorMuon 160M on FineWeb10B, capped at 1B training tokens.
#
# configs/normuon_160m.yaml does batch_size 1024 x sequence_length 1024 =
# 1,048,576 tokens/step, so 1e9 tokens is 953 steps (999,292,928 tokens -- 954
# would step just over the budget). This overrides the config's 3000.
#
# On one H200 that is roughly 40 min of stepping, plus ~15 min across the eight
# validation passes (val_loss_every 125 x 10.5M val tokens) and torch.compile
# warmup, so the 2h override gives it headroom over submit_train.slurm's 1h
# default. Extra args are forwarded to train.py:
#
#   ./launch_normuon_160m.sh
#   ./launch_normuon_160m.sh --lr 0.03 --checkpoint_freq 250
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# WandB run tags, comma-separated -- these show up as filterable tags on the run
# in noahamselsteam/dion. submit_train.slurm forwards WANDB_TAGS into the
# container (train.py's wandb.init() has no tags= argument, so the env var is
# the only hook). Override per-launch without editing this file:
#   WANDB_TAGS=normuon,160m,lr-sweep ./launch_normuon_160m.sh --lr 0.03
export WANDB_TAGS="${WANDB_TAGS:-normuon,160m,1b-tokens}"

exec sbatch \
    --job-name=normuon_160m \
    --time=02:00:00 \
    submit_train.slurm \
        --config configs/normuon_160m.yaml \
        --num_iterations 953 \
        "$@"
