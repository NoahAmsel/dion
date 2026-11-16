#!/usr/bin/env bash
# Sweep the AdamW (embedding + lm_head) learning rate for NorMuon at 225M / 5B
# tokens, with the matrix lr held at 0.01. Every run so far tied the two
# (--scalar_lr 0 = "follow lr"), so no sweep has ever separated them; this picks
# a --scalar_lr for the 418M / 20B NorMuon run.
#
#   ./experiments/scalar_lr_sweep.sh             # 3 jobs, ~3h40m each on one GPU
#   ./experiments/scalar_lr_sweep.sh --foo bar   # extra args go to every run
#
# Baseline (scalar_lr tied at 0.01), from seed_variance.sh:
#   seed 1  normuon_225m_5b_seed1  3.1642   <- the paired comparison for this sweep
#   seed 2  normuon_225m_5b_seed2  3.1639
#   seed 0  normuon_225m_5b-18240104 (resumed branch)  3.1658
#   mean 3.1646 +/- 0.0010 (sd), so ~0.003 is a real difference.
#
# Grid: 0.001, 0.003, 0.03 (0.01 is the baseline above). Why these:
#   - 0.01 is already 10-30x a standard AdamW lr for a ~400M model (3e-4 to 1e-3;
#     Keller Jordan's Muon README uses 3e-4 for its AdamW groups), so most of the
#     grid looks below it.
#   - Recipes disagree by group: modded-nanogpt runs the embedding ~12x ABOVE
#     the Muon lr and the head ~6x BELOW it. --scalar_lr sets both, so 0.03 checks
#     whether the embedding's preference for a high lr dominates.
#   - Half-decade steps cover 0.001-0.03 in three runs.
#
# Everything else matches seed_variance.sh's seed-1 run, so each job differs from
# 3.1642 only in --scalar_lr:
#   - --seed 1: same init and (deterministic loader) the same batches. Not seed 0,
#     whose baseline is a resumed branch rather than a from-scratch run.
#   - 9537 steps, --warmup_ratio 0.04 -> 381 warmup steps, 25% warmdown.
#   - lr 0.01, mu 0.95 (both also in normuon_225m.yaml), no checkpoints.
# The log prints "Scalar (AdamW) group lr: <value>" -- check it is not "(follows lr)".
#
# Reading the result: adopt a value only if it beats 3.1642 by more than ~0.003.
# 225M -> 418M is a small width change; 5B -> 20B tokens is the bigger gap, and
# longer runs tend to prefer slightly lower rates. Applying the winner to NorMuon
# alone breaks comparability with the rest of the size-budget suite (all tied at
# 0.01), and departs from the Dion3 recipe's tied AdamW lr.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

DATA_DIR="${DATA_DIR:-/scratch/$USER/fineweb100B}"

for slr in 0.001 0.003 0.03; do
    name="normuon_225m_5b_seed1_slr${slr}"
    id=$(WANDB_TAGS="normuon,225m,5b-tokens,scalar-lr-sweep,lr0.01,slr$slr,seed1" \
        sbatch --parsable --job-name="$name" --time=05:00:00 \
        submit_train.slurm --data_dir "$DATA_DIR" \
        --config configs/normuon_225m.yaml --optimizer normuon --lr 0.01 --mu 0.95 \
        --scalar_lr "$slr" \
        --num_iterations 9537 --warmup_ratio 0.04 --checkpoint_freq 0 \
        --seed 1 "$@" | tail -1)
    echo "=== $name: job ${id%%;*} ==="
done
