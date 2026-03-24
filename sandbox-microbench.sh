#!/bin/bash
WORKTREE="$HOME/dion"
CONFIG=$WORKTREE/configs/dion2-microbench.yaml
OVERRIDES=(
    --data_dir /data/fineweb10B
    --no_wandb
    --fs_size 4
    --ortho_fraction 1.0
)
torchrun --standalone --nproc_per_node=4 $WORKTREE/train.py --config "$CONFIG" "${OVERRIDES[@]}"
