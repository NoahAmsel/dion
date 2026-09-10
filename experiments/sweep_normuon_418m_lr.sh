#!/usr/bin/env bash
# Learning-rate x momentum sweep on the 418M model, under the Dion3 recipe
# (arXiv 2608.11612): matrix lr scaled by sqrt(d_out/d_in) via
# adjust_lr: spectral_norm, AdamW groups at the plain base lr with no weight
# decay, cooldown to 10% of peak, AdamW betas (0.9, 0.95).
#
# Staged, because the lr and mu axes are close to separable and the full grid
# would be 24 jobs:
#   STAGE 1 (default, 8 jobs)  sweep lr for muon and normuon at the paper's mu=0.9
#   STAGE 2 (4 jobs)           sweep mu at whichever lr won stage 1
#   STAGE 3 (16 jobs)          sweep lr for dion2/dion3 at ortho_fraction 0.25 and 1/16
#
#   ./sweep_normuon_418m_lr.sh
#   STAGE=2 STAGE2_LR=0.01 ./sweep_normuon_418m_lr.sh
#   STAGE=3 ./sweep_normuon_418m_lr.sh
#   ./sweep_normuon_418m_lr.sh --checkpoint_freq 250   # extra args go to every run
#
# Results so far (1907 steps, 999,817,216 tokens, seed 0):
#   stage 1, mu=0.9   lr  0.005    0.01     0.02     0.04
#                muon    3.3496  3.3244   3.3372   3.3966
#             normuon    3.3443  3.3228   3.3500   3.4365
#   stage 2, lr=0.01  mu  0.9      0.95     0.98
#                muon    3.3244  3.3218   3.3585
#             normuon    3.3228  3.3202   3.3477
# Baseline optimum: lr=0.01, mu=0.95, the two optimizers within 0.003 of each
# other. Stage 3 uses mu=0.95 so it is measured against that best baseline.
#
# Run shape: 1907 steps x 512 seqs x 1024 tokens = 999,817,216 tokens -- the same
# ~1B budget as the earlier generation but at 524,288 tokens/step instead of
# 1,048,576, which is close to the paper's ~444k and gives twice the steps for
# the schedule and momentum to act on. Forward/backward halves per step while
# the optimizer step does not, so expect ~110 min per job; hence 3h walltime.
#
# Learning rates are in spectral_norm units, ~7.16x the rms_norm numbers from
# the previous sweep: that study's optima (muon 0.002, normuon 0.004) correspond
# to 0.0143 and 0.0286 here, and the paper reports NorMuon optimal at 0.01 --
# all three inside this ladder.
#
# device_batch_size stays at the config's 64 (grad_accum 8 at batch_size 512).
# Probe job 16817211 showed 128 OOMs on this model.
#
# Stage 3 re-tests the paper's learning-rate rule for selective orthogonalization
# on the recipe it was stated for. The paper claims the optimum rises as the
# selection fraction f falls -- reported both as eta ∝ 1/f and as eta' = eta/sqrt(f).
# Our earlier f=0.25 runs found no shift at all, but those used
# adjust_lr: rms_norm, which is not this recipe, so that result does not settle it.
#
# Two fractions, because a rule about how the optimum moves with f is much better
# tested at two values of f than one. Against a baseline optimum of 0.01 the
# stage 1 ladder pins the predictions:
#     f = 0.25    lr 0.01 no shift | 0.02 eta/sqrt(f) | 0.04 eta/f
#     f = 0.0625  lr 0.01 no shift | 0.04 eta/sqrt(f) | 0.16 eta/f  (0.16 not run)
# So every rung except eta/f at f=1/16 is covered, and 0.16 is far enough past
# where the baselines diverge (0.04 already costs them 0.07-0.11 nats) that it is
# not worth a job. 0.005 brackets from below, and reusing stage 1's four points
# keeps stage 3 comparable to it run for run.
#
# f=1/16 selects 80 of 1280 rows per step on the square matrices. If the deficit
# is a coverage cost -- each row updated only a fraction f of the time -- it should
# grow markedly from f=0.25; if it is roughly flat in f, that reading is wrong.
#
# Not included, but cheap and worth doing if stage 3 disagrees with the paper:
# an ortho_fraction ~1 control at lr=0.01. Under the previous recipe that
# reproduced the baselines to within 0.004 nats (dion2 3.3643 vs muon 3.3654;
# dion3 3.3371 vs normuon 3.3407), which is what makes the f=0.25 deficit
# attributable to the fraction rather than to the Dion code path.
set -euo pipefail
# Run from the repo root: sbatch resolves submit_train.slurm and --config
# relative to the cwd, and submit_train.slurm hands train.py $SLURM_SUBMIT_DIR.
cd "$(dirname "$(readlink -f "$0")")/.."

# Only dion2 / dion3 accept these: their call sites pass fraction,
# use_gram_newton_schulz and use_polar_express, while Muon and NorMuon take
# neither ortho_fraction nor use_gram_newton_schulz and would ignore them.
# --ortho_fraction is supplied per loop in stage 3, not here.
DION_FLAGS=(--use_polar_express --use_gram_newton_schulz)

# The wandb tag carries the optimizer because use_polar_express and
# use_gram_newton_schulz never reach wandb's config -- train.py logs
# hp.__dict__ and those two live on cli_args. dion3-recipe separates this
# generation from the earlier rms_norm / cooldown-to-zero runs.
TAG_EXTRA=""   # appended to WANDB_TAGS; stage 3 uses it to mark the fraction

launch() {
    local opt="$1" lr="$2" mu="$3"; shift 3
    echo "=== $opt lr=$lr mu=$mu ==="
    WANDB_TAGS="$opt,418m,1b-tokens,lr-mu-sweep,dion3-recipe${TAG_EXTRA}" \
    sbatch \
        --job-name="${opt}_418m" \
        --time=02:50:00 \
        submit_train.slurm \
            --config configs/normuon_418m.yaml \
            --num_iterations 1907 \
            --optimizer "$opt" \
            --lr "$lr" \
            --mu "$mu" \
            "$@"
}

STAGE="${STAGE:-1}"
STAGE2_LR="${STAGE2_LR:-}"   # set to stage 1's winning lr before running STAGE=2

case "$STAGE" in
  1)
    for opt in muon normuon; do
        for lr in 0.005 0.01 0.02 0.04; do
            launch "$opt" "$lr" 0.9 "$@"
        done
    done
    ;;
  2)
    if [[ -z "$STAGE2_LR" ]]; then
        echo "error: set STAGE2_LR to stage 1's winning learning rate" >&2
        exit 1
    fi
    for opt in muon normuon; do
        for mu in 0.95 0.98; do
            launch "$opt" "$STAGE2_LR" "$mu" "$@"
        done
    done
    ;;
  3)
    for frac in 0.0625; do
        TAG_EXTRA=",f${frac}"
        for opt in dion2 dion3; do
            for lr in 0.01 0.02 0.04; do
                launch "$opt" "$lr" 0.95 --ortho_fraction "$frac" \
                    "${DION_FLAGS[@]}" "$@"
            done
        done
    done
    TAG_EXTRA=""
    ;;
  *) echo "error: STAGE must be 1, 2 or 3" >&2; exit 1 ;;
esac
