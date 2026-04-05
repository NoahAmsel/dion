#!/bin/bash
WORKTREE="$HOME/dion"
CONFIG=$WORKTREE/configs/dion2-microbench.yaml
OVERRIDES=(
    --data_dir /data/fineweb10B
    --no_wandb
)
# torchrun --standalone --nproc_per_node=4 $WORKTREE/train.py --config "$CONFIG" "${OVERRIDES[@]}" --fs_size 4
CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 $WORKTREE/train.py --config "$CONFIG" "${OVERRIDES[@]}"