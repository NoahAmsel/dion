#!/bin/bash
WORKTREE="$HOME/dion"
CONFIG=$WORKTREE/configs/dion2-microbench.yaml
OVERRIDES=(
    --data_dir /data/fineweb10B
    --no_wandb
    --optimizer muon
    # --no_triton
    --use_gns_package
    --use_gns_alg
    # --ortho_fraction 0.25
    # --split_heads
)
torchrun --standalone --nproc_per_node=4 $WORKTREE/train.py --config "$CONFIG" "${OVERRIDES[@]}" --fs_size 4
# CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 $WORKTREE/train.py --config "$CONFIG" "${OVERRIDES[@]}"

# RESULT: YES, going from muon, no triton to muon gns package and alg helps overall runtime by 8%. dion + split heads helps more
# but on dion speed, not sure i can reproduce this. is it because dion speed isn't measuring the per step time? or because I was including compile time when i did the timing on command line?