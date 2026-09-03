#!/usr/bin/env bash
# Learning-rate sweep for NorMuon 160M. Four runs, each identical to
# launch_normuon_160m.sh (same config, same 1B-token budget, same 1h
# reservation) except for --lr, all tagged normuon-lr-sweep in
# noahamselsteam/dion so they can be pulled up as a group.
#
# The ladder is 2x spacing bracketing the config's tuned 0.02. Edit the array to
# change it -- nothing else here depends on the number of points.
#
#   ./sweep_normuon_160m_lr.sh
#   ./sweep_normuon_160m_lr.sh --checkpoint_freq 250   # extra args go to every run
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

LEARNING_RATES=(0.002 0.005 0.01 0.02 0.04)

for lr in "${LEARNING_RATES[@]}"; do
    echo "=== lr=$lr ==="
    WANDB_TAGS=normuon-lr-sweep ./launch_normuon_160m.sh --lr "$lr" "$@"
done
