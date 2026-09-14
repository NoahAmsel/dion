#!/usr/bin/env bash
# Four 10B-token runs on configs/normuon_3b.yaml (2,835,294,464 params, tied;
# 19073 steps x 256 x 2048 = 524,288 tokens/step), one 4-GPU node each:
#
#   1. normuon                     lr 0.01
#   2. normuon                     lr 0.02
#   3. dion3   ortho_fraction 1/4  lr 0.01
#   4. dion3   ortho_fraction 1/4  lr 0.02     (eta/sqrt(f) relative to normuon at 0.01)
#
#   SMOKE=1 ./launch_3b_10b.sh    # 30-step dion3 job that saves a checkpoint at step 20
#   SMOKE=2 ./launch_3b_10b.sh    # resume that checkpoint to step 40
#   ./launch_3b_10b.sh            # the four runs
#   ./launch_3b_10b.sh --device_batch_size 8    # extra args go to every run
#
# Run both smoke steps first: FSDP across 4 GPUs, and DCP save/resume under it,
# have never been exercised in this repo.
#
# Sizing (fitted from the 418M new-architecture runs): ~86 GiB per GPU at the
# config's device_batch_size 8, and 30-40h for 19073 steps (sequence length 2048
# adds ~5% compute in attention). To resume a run that
# hits the walltime:
#   sbatch --job-name=normuon_3b --gres=gpu:4 --cpus-per-task=32 --mem=256GB \
#       --time=44:00:00 submit_train.slurm --config configs/normuon_3b.yaml \
#       --optimizer normuon --lr 0.01 \
#       --checkpoint_freq 1000 --no_wandb \
#       --checkpoint_dir <RUN_ROOT>/checkpoints/<RUN>   # RUN_ROOT as in submit_train.slurm
# (--no_wandb because the checkpoint carries the wandb id with resume="must".)

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

GRES="${GRES:-gpu:4}"          # must agree with fs_size 4 in the config
TIME_LIMIT="${TIME_LIMIT:-44:00:00}"

# Check the data before submitting: 10B tokens needs the full FineWeb10B (103
# shards), and the loader silently wraps around a partial download instead of
# failing. Uses the same directory submit_train.slurm will: $DATA_DIR if set,
# otherwise that script's default.
data_dir="${DATA_DIR:-$(sed -n 's/^DATA_DIR="\${DATA_DIR:-\(.*\)}"$/\1/p' submit_train.slurm)}"
shopt -s nullglob; shards=("$data_dir"/fineweb_train_*.bin); shopt -u nullglob
n_shards=${#shards[@]}
if [[ "$n_shards" -lt 103 ]] || [[ ! -e "$data_dir/fineweb_val_000000.bin" ]]; then
    echo "error: '$data_dir' has $n_shards/103 train shards (and needs the val shard)." >&2
    echo "       Download with: python data/cached_fineweb10B.py, or set DATA_DIR." >&2
    exit 1
fi

# The config turns on Polar Express + Gram Newton-Schulz, clipping, SwiGLU, the
# attention gate and norm gains for every arm; only --ortho_fraction is dion3's.
TAGS_BASE="3b,10b-tokens,dion3-recipe,grad-clip,swiglu,attn-gate,norm-gain,gns"

launch() {
    local opt="$1" lr="$2"; shift 2
    echo "=== $opt lr=$lr ==="
    WANDB_TAGS="$opt,$TAGS_BASE,lr$lr" \
    sbatch \
        --job-name="${opt}_3b" \
        --gres="$GRES" \
        --cpus-per-task=32 \
        --mem=256GB \
        --time="$TIME_LIMIT" \
        submit_train.slurm \
            --config configs/normuon_3b.yaml \
            --optimizer "$opt" \
            --lr "$lr" \
            --checkpoint_freq 1000 \
            "$@"
}

# Both smoke steps need one fixed checkpoint dir (submit_train.slurm's default is
# per job). Put it under the same run root that script uses: $RUN_ROOT if set,
# otherwise its default.
run_root="${RUN_ROOT:-$(sed -n 's/^RUN_ROOT="\${RUN_ROOT:-\(.*\)}"$/\1/p' submit_train.slurm)}"
SMOKE_DIR="$run_root/checkpoints/dion3_3b-smoke"
case "${SMOKE:-0}" in
  1)
    TIME_LIMIT=00:45:00
    launch dion3 0.01 --ortho_fraction 0.25 --num_iterations 30 --val_loss_every 10 \
        --checkpoint_freq 20 --checkpoint_dir "$SMOKE_DIR" --no_wandb "$@"
    ;;
  2)
    TIME_LIMIT=00:45:00
    launch dion3 0.01 --ortho_fraction 0.25 --num_iterations 40 --val_loss_every 10 \
        --checkpoint_freq 0 --checkpoint_dir "$SMOKE_DIR" --no_wandb "$@"
    ;;
  0)
    launch normuon 0.01 "$@"
    launch normuon 0.02 "$@"
    launch dion3   0.01 --ortho_fraction 0.25 "$@"
    launch dion3   0.02 --ortho_fraction 0.25 "$@"
    ;;
  *) echo "error: SMOKE must be 0, 1 or 2" >&2; exit 1 ;;
esac
