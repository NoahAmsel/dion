#!/usr/bin/env bash
# Seed-to-seed variability at 225M / 5B tokens, NorMuon vs Dion3 (f = 1/8),
# lr 0.01. Seed 0 of each already exists as a branch of launch_size_budget.sh;
# this adds seeds 1 and 2, so each optimizer has three runs identical except
# for --seed.
#
#   optimizer  seed  job name                    slurm job  wandb run  val_loss
#   normuon    0     normuon_225m_5b-18240104    18240104   1nzaxlmf   3.1658
#   dion3      0     dion3_225m_5b-18240111      18240111   nhrjo89c   3.1890
#   normuon    1,2   normuon_225m_5b_seed{1,2}   (this script)
#   dion3      1,2   dion3_225m_5b_seed{1,2}     (this script)
#
# The seed-0 jobs resumed checkpoint-7153 of their 20B parents
# (normuon_225m-18240100, dion3_225m-18240106) and ran to 9537 steps. These
# start from scratch instead and reproduce that run:
#   - warmup: the branches inherited the parent's round(0.01 * 38147) = 381
#     steps, so --warmup_ratio 0.04 gives round(0.04 * 9537) = 381 here.
#   - warmdown: round(0.25 * 9537) = 2384 from num_iterations 9537, as in the
#     branches.
#   - data: the branches restored the loader's position from the checkpoint, so
#     they saw the same token stream a fresh run does.
# --seed sets init only: the loader is deterministic and neither optimizer draws
# random numbers, so every seed sees the same batches in the same order.
#
#   ./experiments/seed_variance.sh               # 4 jobs, ~4h each on one GPU
#   ./experiments/seed_variance.sh --foo bar     # extra args go to every run
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

DATA_DIR="${DATA_DIR:-/scratch/$USER/fineweb100B}"

for opt in normuon dion3; do
    # dion3 only; the config already turns on Polar Express + GNS for both arms.
    flags=(); [[ "$opt" == dion3 ]] && flags=(--ortho_fraction 0.125)
    for seed in 1 2; do
        name="${opt}_225m_5b_seed${seed}"
        id=$(WANDB_TAGS="$opt,225m,5b-tokens,seed-variance,f0.125,lr0.01,seed$seed" \
            sbatch --parsable --job-name="$name" --time=05:00:00 \
            submit_train.slurm --data_dir "$DATA_DIR" \
            --config configs/normuon_225m.yaml --optimizer "$opt" --lr 0.01 \
            --num_iterations 9537 --warmup_ratio 0.04 --checkpoint_freq 0 \
            "${flags[@]}" --seed "$seed" "$@" | tail -1)
        echo "=== $name: job ${id%%;*} ==="
    done
done
