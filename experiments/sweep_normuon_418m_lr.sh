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
#   STAGE 3 (12 jobs)          sweep lr for dion2/dion3 at ortho_fraction 0.25 and 1/16
#   STAGE 4 (2 jobs)           experiment F: normuon vs dion3 at 10x the budget
#                              (resumes the 4B runs; 1B and 4B already done)
#   STAGE 5 (1 job)            experiment A follow-up: isolate Polar Express
#                              (A and B themselves are done -- see the stage 5 block)
#
#   ./sweep_normuon_418m_lr.sh
#   STAGE=2 STAGE2_LR=0.01 ./sweep_normuon_418m_lr.sh
#   STAGE=3 ./sweep_normuon_418m_lr.sh
#   STAGE=4 ./sweep_normuon_418m_lr.sh
#   STAGE=5 ./sweep_normuon_418m_lr.sh
#   ./sweep_normuon_418m_lr.sh --checkpoint_freq 250   # extra args go to every run
#
# Results so far (1907 steps, 999,817,216 tokens, seed 0):
#   stage 1, mu=0.9   lr  0.005    0.01     0.02     0.04
#                muon    3.3496  3.3244   3.3372   3.3966
#             normuon    3.3443  3.3228   3.3500   3.4365
#   stage 2, lr=0.01  mu  0.9      0.95     0.98
#                muon    3.3244  3.3218   3.3585
#             normuon    3.3228  3.3202   3.3477
#   stage 3, mu=0.95  lr  0.005    0.01     0.02     0.04
#        dion2 f=0.25    3.4429  3.4143   3.4322     --
#        dion3 f=0.25    3.4219  3.3988   3.4167     --
#        dion2 f=1/16      --    3.6839   3.6808   3.7552
#        dion3 f=1/16      --    3.6424   3.6388   3.7034
# Baseline optimum: lr=0.01, mu=0.95, the two optimizers within 0.003 of each
# other. Stage 3 uses mu=0.95 so it is measured against that best baseline.
#
# Stage 3 answers both questions in the negative. The optimum does not move: it
# is 0.01 at f=0.25 and 0.01-0.02 at f=1/16, never the 0.02 / 0.04 that
# eta/sqrt(f) predicts, and at f=1/16 the predicted 0.04 is the *worst* rung by
# 0.065-0.074. And the deficit does scale like 1/f -- cutting f by 4x multiplies
# the gap to the tuned baseline by 3.88x (dion2) and 4.05x (dion3) -- so the
# coverage reading below is the one the data supports.
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
# Stages 4 and 5 stand on the f=0.25 choice, so DION_FLAGS bundles it; stage 3
# varies the fraction and passes GNS_FLAGS with its own --ortho_fraction instead.
GNS_FLAGS=(--use_polar_express --use_gram_newton_schulz)
DION_FLAGS=(--ortho_fraction 0.25 "${GNS_FLAGS[@]}")

# The wandb tag carries the optimizer because use_polar_express and
# use_gram_newton_schulz never reach wandb's config -- train.py logs
# hp.__dict__ and those two live on cli_args. dion3-recipe separates this
# generation from the earlier rms_norm / cooldown-to-zero runs.
# TIME_LIMIT, TOKENS_TAG and TAG_EXTRA let a stage change the walltime and mark
# its runs; everything after the three positional args is forwarded to train.py,
# so a stage can also override --num_iterations there (argparse takes the last
# occurrence). A stage that sets any of them must restore the defaults after.
TIME_LIMIT="02:50:00"
TOKENS_TAG="1b-tokens"
TAG_EXTRA=""      # stage 3 marks the fraction here, stages 4 and 5 the experiment

launch() {
    local opt="$1" lr="$2" mu="$3"; shift 3
    echo "=== $opt lr=$lr mu=$mu ==="
    WANDB_TAGS="$opt,418m,${TOKENS_TAG},lr-mu-sweep,dion3-recipe${TAG_EXTRA}" \
    sbatch \
        --job-name="${opt}_418m" \
        --time="$TIME_LIMIT" \
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
    # GNS_FLAGS, not DION_FLAGS: the latter carries --ortho_fraction 0.25, and
    # argparse takes the last occurrence, so it would override the loop variable.
    #
    # The ladder shifts with the fraction so that each one brackets its own
    # predicted optimum: 0.005/0.01/0.02 spans no-shift through eta/sqrt(f) at
    # f=0.25, and 0.01/0.02/0.04 does the same at f=1/16 (where eta/sqrt(f)=0.04).
    for frac in 0.25 0.0625; do
        case "$frac" in
            0.25)   lrs=(0.005 0.01 0.02) ;;
            0.0625) lrs=(0.01 0.02 0.04) ;;
            *) echo "error: no lr ladder defined for frac $frac" >&2; exit 1 ;;
        esac
        TAG_EXTRA=",f${frac}"
        for opt in dion2 dion3; do
            for lr in "${lrs[@]}"; do
                launch "$opt" "$lr" 0.95 --ortho_fraction "$frac" \
                    "${GNS_FLAGS[@]}" "$@"
            done
        done
    done
    TAG_EXTRA=""
    ;;
  4)
    # Experiment F: budget scaling of the NorMuon -> Dion3 gap.
    #
    # Two jobs, not four. The analysis lists four because it specifies the pair at
    # both 1907 and 7628 steps, but the 1907 points are already measured at exactly
    # this recipe and lr -- normuon 3.3202 and dion3 f=0.25 3.3988, jobs 17002705
    # and 17010461, run 2026-09-05 -- and nothing under train.py, models/, dion/ or
    # configs/ has been committed since (1da2eab predates them). Re-running them
    # would buy only a repeatability check, which costs 6 GPU-hours to learn
    # something the ~0.003 nat noise floor already tells us.
    #
    # 7628 steps x 512 seqs x 1024 tokens = 3,999,268,864 tokens, 4x the budget.
    # The schedule scales with the run (381 warmup, 1526 cooldown), so this is a
    # genuine 4x-budget run and not a continuation. At the measured 3.3 s/step that
    # is ~7h of stepping; the 9h walltime is ~25% margin. Measured on the first
    # attempt (jobs 17309176/7, cancelled at step ~100): normuon 3.09 s/it, dion3
    # 2.97 s/it, so ~6.5h and ~6.3h.
    #
    # --checkpoint_freq 6000 is chosen, not arbitrary. save() always writes to
    # DEFAULT_NAME and replaces the directory, so exactly one checkpoint survives:
    # the last one taken. Cooldown starts at step 6102, and saves fire at
    # step % freq == 0, so 6000 puts the single surviving checkpoint 102 steps
    # BEFORE cooldown -- the right place to branch a longer run from. A smaller
    # freq (say 1000) would leave the survivor at step 7000, already partway
    # through cooldown and useless as a branch point. It also caps the loss from a
    # walltime kill at 6000 steps instead of everything.
    #
    # Both arms stay at lr 0.01, mu 0.95 -- each optimizer's own best at 1907. That
    # assumes the optimum does not move with budget. If it does, the assumption is
    # at least symmetric across the two arms, so the sign of Delta is still
    # informative even if its magnitude is not.
    TIME_LIMIT="09:00:00"
    TOKENS_TAG="4b-tokens"
    TAG_EXTRA=",budget4x"
    # Done -- jobs 17309781/2. normuon 3.0428, dion3 f=0.25 3.0742, Delta +0.0314.
    # launch normuon 0.01 0.95 --num_iterations 7628 --checkpoint_freq 6000 "$@"
    # launch dion3   0.01 0.95 --num_iterations 7628 --checkpoint_freq 6000 \
    #     "${DION_FLAGS[@]}" "$@"
    TIME_LIMIT="02:50:00"; TOKENS_TAG="1b-tokens"; TAG_EXTRA=""
    #
    # To EXTEND either run past 7628 steps, resume from its step-6000 checkpoint.
    # checkpoint_dir is per job (submit_train.slurm derives it from the job name and
    # id), so a fresh job never picks one up by accident -- you have to name it, and
    # the one you pass wins because it lands after the script's own copy.
    #
    # Resuming re-enters the same wandb run (the id is stored in the checkpoint and
    # resume="must"), which would re-log steps 6001+ on top of history the original
    # run already wrote. Hence --no_wandb; the losses are still in the slurm log.
    #
    # 15256 steps = 8B tokens; resuming at 6001 leaves 9256 steps ~ 8h, so 10h.
    # Warmup (381) already happened and its branch never fires again; the constant
    # phase now runs to 12205 and cooldown to 15256.
    #
    #   RUN=normuon_418m-<JOBID>            # or dion3_418m-<JOBID>
    #   sbatch --job-name=normuon_418m --time=10:00:00 submit_train.slurm \
    #       --config configs/normuon_418m.yaml \
    #       --optimizer normuon --lr 0.01 --mu 0.95 \
    #       --num_iterations 15256 --checkpoint_freq 12000 --no_wandb \
    #       --checkpoint_dir /scratch/nia4240/dion/checkpoints/$RUN
    #
    # For the dion3 arm add: --ortho_fraction 0.25 --use_polar_express \
    #                        --use_gram_newton_schulz
    # (--checkpoint_freq 12000 keeps the same trick: 12000 is the last save before
    #  the new cooldown at 12205, so the extended run is itself extendable.)

    # ---- 10B tokens, resumed from the 4B runs' step-6000 checkpoints ----------
    #
    # Delta so far: +0.0786 at 1B (1907 steps), +0.0314 at 4B (7628) -- a 2.5x fall
    # for a 4x budget. In step terms normuon reaches dion3's final loss in 9.8%
    # fewer steps at 1B and 4.8% fewer at 4B. This adds the third point.
    #
    # 19073 steps x 524,288 = 9,999,745,024 tokens. Resuming at 6001 leaves 13,073
    # steps ~ 10.9h at the measured 3.0 s/it, hence the 14h walltime.
    #
    # The checkpoint directory is both the source and the destination, and only the
    # newest save survives (save() always writes to DEFAULT_NAME), so running these
    # against the 4B directories would destroy the 4B branch point at step 15000.
    # Copy first and point the jobs at the copies.
    #
    # REQUIRES the scheduler fast-forward in train.py (search for "LambdaLR's state
    # is not in the checkpoint"). LambdaLR is not checkpointed, so without it a
    # resumed run replays the schedule from step 0: warmup again, and cooldown
    # scheduled for loop step 6001 + 15258 = 21259 > 19073, i.e. never.
    #
    # --checkpoint_freq 15000: cooldown starts at 19073 - 3815 = 15258, so 15000 is
    # the largest multiple landing before it -- this run stays extendable in turn.
    CKPT_ROOT=/scratch/nia4240/dion/checkpoints
    cp -rn "$CKPT_ROOT/normuon_418m-17309781" "$CKPT_ROOT/normuon_418m-10b"
    cp -rn "$CKPT_ROOT/dion3_418m-17309782"   "$CKPT_ROOT/dion3_418m-10b"

    TIME_LIMIT="14:00:00"
    TOKENS_TAG="10b-tokens"
    TAG_EXTRA=",budget10b,resumed-from-4b"
    # --no_wandb: the checkpoint carries the wandb id and resume="must", so these
    # would re-log steps 6001+ over the history the 4B runs already wrote. The
    # losses still land in the slurm log.
    launch normuon 0.01 0.95 --num_iterations 19073 --checkpoint_freq 15000 \
        --no_wandb --checkpoint_dir "$CKPT_ROOT/normuon_418m-10b" "$@"
    launch dion3   0.01 0.95 --num_iterations 19073 --checkpoint_freq 15000 \
        --no_wandb --checkpoint_dir "$CKPT_ROOT/dion3_418m-10b" \
        "${DION_FLAGS[@]}" "$@"
    TIME_LIMIT="02:50:00"; TOKENS_TAG="1b-tokens"; TAG_EXTRA=""
    ;;
  5)
    # RESULTS (1907 steps, lr 0.01, seed 0):
    #   A: dion2 f=1  3.3190  vs muon    3.3218  ->  -0.0028  (within noise)
    #      dion3 f=1  3.3111  vs normuon 3.3202  ->  -0.0091  (3x the noise floor)
    #   B: normuon mu=0.9 beta2=0.90  3.3226  vs beta2=0.95  3.3228  ->  -0.0002
    #      B is settled: beta2 does nothing, the baseline stands, H4 is dead.
    #
    # A did not come out clean, and the fourth arm below is why. dion3(f=1) is not
    # a pure "NorDion2 with nothing selected" control: it also swaps the
    # orthogonalization routine. dion3 runs with --use_polar_express
    # --use_gram_newton_schulz, so megabatch_base takes the GNS callable, while
    # normuon ran newton_schulz_triton. So the -0.0091 bundles two changes and
    # cannot be attributed to the Dion path alone.
    #
    # This arm splits them cleanly. NorMuon's call site in train.py used to drop
    # use_gram_newton_schulz -- it now forwards it, as NorMuon.__init__ always
    # accepted it -- so normuon can run the *identical* orthogonalization to
    # dion3(f=1). That makes each comparison one variable:
    #
    #   normuon+GNS  vs  normuon 3.3202   -> what the routine is worth
    #   dion3(f=1) 3.3111  vs  normuon+GNS -> what NorDion2 itself is worth
    #
    # and the two must sum to the -0.0091 already measured. So:
    #   lands near 3.3111 -> the routine explains it; the Dion path is clean and
    #                        dion3's f=0.25 deficit should be measured against
    #                        3.3111, i.e. +0.0877 rather than +0.0786
    #   lands near 3.3202 -> the routine is not the cause and NorDion2 itself is
    #                        worth 0.009 at f=1, which stage 3 was crediting to
    #                        the selection fraction
    TAG_EXTRA=",gns-isolation"
    launch normuon 0.01 0.95 "${GNS_FLAGS[@]}" "$@"
    TAG_EXTRA=""

    # Already run -- see RESULTS above. Uncomment to repeat.
    # TAG_EXTRA=",f1-control"
    # launch dion2 0.01 0.95 --ortho_fraction 1.0 "${GNS_FLAGS[@]}" "$@"
    # launch dion3 0.01 0.95 --ortho_fraction 1.0 "${GNS_FLAGS[@]}" "$@"
    # TAG_EXTRA=",beta2"
    # launch normuon 0.01 0.9 --muon_beta2 0.9 "$@"
    # TAG_EXTRA=""
    ;;
  *) echo "error: STAGE must be 1, 2, 3, 4 or 5" >&2; exit 1 ;;
esac
