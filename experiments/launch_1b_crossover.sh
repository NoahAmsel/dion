#!/usr/bin/env bash
# Three 20B-token runs on configs/normuon_1b.yaml (1,105,133,568 params,
# 38147 steps x 524,288 tokens/step), one 4-GPU node each:
#
#   1. normuon                     lr 0.01             baseline
#   2. dion3   ortho_fraction 1/8  lr 0.01             f8-baselr
#   3. dion3   ortho_fraction 1/8  lr 0.01*sqrt(8)     f8-sqrtf-lr   (eta/sqrt(f))
#
#   ./launch_1b_crossover.sh
#   ./launch_1b_crossover.sh --device_batch_size 16   # extra args go to every run
#   GRES=gpu:8 TIME_LIMIT=12:00:00 ./launch_1b_crossover.sh
#
# Sizing: ~64 GiB per GPU at the config's device_batch_size 32 (see the memory
# fit in git history), and 19-25h for 38147 steps at 4x an H200's measured 488
# TFLOP/s, inside the 30h default walltime. To resume one that dies anyway:
#   sbatch --job-name=normuon_1b --gres=gpu:4 --cpus-per-task=32 --mem=256GB \
#       --time=30:00:00 submit_train.slurm --config configs/normuon_1b.yaml \
#       --optimizer normuon --lr 0.01 --checkpoint_freq 3000 --no_wandb \
#       --checkpoint_dir /scratch/nia4240/dion/checkpoints/<RUN>
# --no_wandb because the checkpoint carries the wandb id with resume="must", so
# a resumed job would re-log steps the original already wrote.

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

# submit_train.slurm derives nproc_per_node from the allocation, so --gres alone
# sets the world size -- but it has to agree with fs_size 4 in the config.
GRES="${GRES:-gpu:4}"
TIME_LIMIT="${TIME_LIMIT:-30:00:00}"

# dion3 only: Muon and NorMuon take neither ortho_fraction nor
# use_gram_newton_schulz. Note that the baseline therefore runs the Triton
# Newton-Schulz while these run Polar Express + Gram Newton-Schulz; add
# "${GNS_FLAGS[@]}" to the normuon launch if you want to match them.
GNS_FLAGS=(--use_polar_express --use_gram_newton_schulz)
DION_FLAGS=(--ortho_fraction 0.125 "${GNS_FLAGS[@]}")

launch() {
    local opt="$1" lr="$2" tag="$3"; shift 3
    echo "=== $opt lr=$lr ($tag) ==="
    WANDB_TAGS="$opt,1b,20b-tokens,crossover,dion3-recipe,$tag" \
    sbatch \
        --job-name="${opt}_1b" \
        --gres="$GRES" \
        --cpus-per-task=32 \
        --mem=256GB \
        --time="$TIME_LIMIT" \
        submit_train.slurm \
            --config configs/normuon_1b.yaml \
            --optimizer "$opt" \
            --lr "$lr" \
            --checkpoint_freq 3000 \
            "$@"
}

launch normuon 0.01     baseline    "$@"
launch dion3   0.01     f8-baselr   "${DION_FLAGS[@]}" "$@"
launch dion3   0.028284 f8-sqrtf-lr "${DION_FLAGS[@]}" "$@"   # 0.01 * sqrt(8)
