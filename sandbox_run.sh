#!/bin/bash
WORKTREE="$HOME/dion"
CONFIG=$WORKTREE/configs/muon_160m.yaml
OVERRIDES=(
    --data_dir /data/fineweb10B
    --wandb_project_name=dion-repo
    # --fs_size 4
)
CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 $WORKTREE/train.py --config "$CONFIG" "${OVERRIDES[@]}"
# torchrun --standalone --nproc_per_node=4 $WORKTREE/train.py --config "$CONFIG" "${OVERRIDES[@]}"
