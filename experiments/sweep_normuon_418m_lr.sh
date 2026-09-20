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
#   STAGE 6 (6 jobs)           gradient clipping at 1.0: replicas of 3 earlier jobs
#                              at 1907 and 19073 steps, each with a named counterpart
#   STAGE 7 (6 jobs)           stage 6's six jobs again, with SwiGLU, attention head
#                              gates and learnable RMSNorm gains on top of clipping
#   STAGE 8 (7 jobs)           under stage 7's recipe: normuon+GNS lr sweep, dion3
#                              at lr 0.04, and seed-1 replicates of both at lr 0.01
#   STAGE 9 (4 jobs)           dion3 at f=1/16 across lr 0.01-0.08 under stage 7's
#                              recipe: a direct test of the eta/sqrt(f) rule
#   STAGE 10 (1 job)           normuon at lr 0.02 over 10B tokens, the missing arm
#                              of stage 7, keeping checkpoints to branch cooldowns from
#
#   ./sweep_normuon_418m_lr.sh
#   STAGE=2 STAGE2_LR=0.01 ./sweep_normuon_418m_lr.sh
#   STAGE=3 ./sweep_normuon_418m_lr.sh
#   STAGE=4 ./sweep_normuon_418m_lr.sh
#   STAGE=5 ./sweep_normuon_418m_lr.sh
#   STAGE=6 ./sweep_normuon_418m_lr.sh
#   STAGE=7 ./sweep_normuon_418m_lr.sh
#   STAGE=8 ./sweep_normuon_418m_lr.sh
#   STAGE=9 ./sweep_normuon_418m_lr.sh
#   STAGE=10 ./sweep_normuon_418m_lr.sh
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
  6)
    # Gradient clipping at global norm 1.0 (all parameters), as in the private
    # code. Each job repeats an earlier unclipped job flag for flag, so each pair
    # measures the effect of clipping alone:
    #
    #   steps  job                  unclipped counterpart              final loss
    #   1907   normuon     lr 0.01  17002705                           3.3202
    #   1907   dion3 f=1/4 lr 0.01  17010461                           3.3988
    #   1907   dion3 f=1/4 lr 0.02  17010463                           3.4167
    #   19073  normuon     lr 0.01  17309781 -> resumed as 17330992    2.9474
    #   19073  dion3 f=1/4 lr 0.01  17309782 -> resumed as 17330993    2.9614
    #   19073  dion3 f=1/4 lr 0.02  none -- never run unclipped
    #
    # The 19073-step counterparts trained steps 0-6000 as 4B runs, so their warmup
    # was 381 steps (5% of 7628), not the 954 a fresh 19073-step run would get.
    # --warmup_ratio 0.02 gives round(0.02 * 19073) = 381, and cooldown still
    # starts at 15258: the lr is identical to theirs at every step.
    #
    # NorMuon keeps its counterparts' plain Newton-Schulz, so normuon vs dion3
    # still carries the ~0.009 routine confound from stage 5.
    # Launched -- 1907: 17722156/7/8, 19073: 17722159/60/61. Uncomment to repeat.
    TAG_EXTRA=",grad-clip"
    for steps in 1907 19073; do
        if [[ "$steps" == 1907 ]]; then
            TIME_LIMIT="02:50:00"; TOKENS_TAG="1b-tokens"; SCHED=()
        else
            # The counterparts ran at 3.11 s/step including validation: ~16.5h.
            TIME_LIMIT="20:00:00"; TOKENS_TAG="10b-tokens"
            SCHED=(--warmup_ratio 0.02 --checkpoint_freq 3000)
        fi
        CLIP=(--num_iterations "$steps" --grad_clip 1.0 "${SCHED[@]}")
        launch normuon 0.01 0.95 "${CLIP[@]}" "$@"
        launch dion3   0.01 0.95 "${CLIP[@]}" "${DION_FLAGS[@]}" "$@"
        launch dion3   0.02 0.95 "${CLIP[@]}" "${DION_FLAGS[@]}" "$@"
    done
    TIME_LIMIT="02:50:00"; TOKENS_TAG="1b-tokens"; TAG_EXTRA=""

    # Does clipping's benefit grow as the fraction falls? At 1907 steps it was
    # worth -0.0111 to dion3 f=1/4 (17722157 vs 17010461) but only -0.0023 to
    # normuon (17722156 vs 17002705). One reading: dion3 decays momentum only on
    # selected rows, so an unclipped early spike lingers ~1/f times longer. If so,
    # clipping should help f=1/16 clearly more than f=1/4.
    #
    #   steps  job                   unclipped counterpart   final loss
    #   1907   dion3 f=1/16 lr 0.01  17225141                3.6424
    #
    # Same flags as the counterpart (stage 3's f=1/16 rung) plus --grad_clip 1.0.
    TAG_EXTRA=",grad-clip,f0.0625"
    launch dion3 0.01 0.95 --grad_clip 1.0 --ortho_fraction 0.0625 "${GNS_FLAGS[@]}" "$@"
    TAG_EXTRA=""
    ;;
  7)
    # Stage 6's six clipped jobs with the architecture changes turned on:
    # SwiGLU MLP (up/gate/down at 4 * d), a headwise sigmoid gate on attention,
    # and learnable gains on every RMSNorm. Everything else is flag for flag the
    # stage-6 job, so each pair measures the architecture change alone:
    #
    #   steps  job                  stage-6 counterpart (clipped)   final loss
    #   1907   normuon     lr 0.01  17722156                        3.3179
    #   1907   dion3 f=1/4 lr 0.01  17722157                        3.3877
    #   1907   dion3 f=1/4 lr 0.02  17722158                        3.4209
    #   19073  normuon     lr 0.01  17722159                        (running)
    #   19073  dion3 f=1/4 lr 0.01  17722160                        (running)
    #   19073  dion3 f=1/4 lr 0.02  17722161                        (pending)
    #
    # The new matrices (SwiGLU's three, the n_embd -> n_head gate) are
    # orthogonalized with the rest; the gains go to AdamW with no weight decay.
    # SwiGLU at 4 * d is 1.5x the MLP parameters (~536M total, vs 418M) and
    # ~1.25x the per-token compute, so the walltimes are scaled up from stage 6.
    ARCH=(--mlp_type swiglu --attn_gate --norm_gain)
    TAG_EXTRA=",grad-clip,swiglu,attn-gate,norm-gain"
    for steps in 1907 19073; do
        if [[ "$steps" == 1907 ]]; then
            TIME_LIMIT="03:30:00"; TOKENS_TAG="1b-tokens"; SCHED=()
        else
            # ~3.9 s/step estimated (stage 6's 3.1 x 1.25): ~21h.
            TIME_LIMIT="26:00:00"; TOKENS_TAG="10b-tokens"
            SCHED=(--warmup_ratio 0.02 --checkpoint_freq 3000)
        fi
        CLIP=(--num_iterations "$steps" --grad_clip 1.0 "${SCHED[@]}" "${ARCH[@]}")
        launch normuon 0.01 0.95 "${CLIP[@]}" "$@"
        launch dion3   0.01 0.95 "${CLIP[@]}" "${DION_FLAGS[@]}" "$@"
        launch dion3   0.02 0.95 "${CLIP[@]}" "${DION_FLAGS[@]}" "$@"
    done
    TIME_LIMIT="02:50:00"; TOKENS_TAG="1b-tokens"; TAG_EXTRA=""
    ;;
  8)
    # Retune under stage 7's recipe (grad clip 1.0 + SwiGLU + attention gate +
    # norm gains), 1907 steps, fixing two things stage 7 left open:
    #
    # (a) NorMuon gets Polar Express + Gram Newton-Schulz, the routine every dion3
    #     arm already runs. Without it normuon vs dion3 carries a routine
    #     difference worth -0.0092 to NorMuon under the original recipe (17368303
    #     vs 17002705). And NorMuon has never been run above lr 0.01 on the new
    #     architecture, where dion3's lr 0.02 now ties its lr 0.01.
    #
    #       job                       nearest existing run
    #       normuon+GNS  lr 0.005     none
    #       normuon+GNS  lr 0.01      17733553 (same, no GNS)      3.2370
    #       normuon+GNS  lr 0.02      none
    #       normuon+GNS  lr 0.04      none
    #
    # (b) dion3 f=1/4 at lr 0.04, above the stage-7 tie, to tell a moved optimum
    #     from a flat top.
    #
    #       dion3 f=1/4  lr 0.04      brackets 17733554 (lr 0.01)  3.3074
    #                                          17733555 (lr 0.02)  3.3097
    #
    # (c) Seed 1 replicates at lr 0.01, the first between-seed noise measurement
    #     (same-seed reruns drift ~0.002; nothing is known across seeds). Each
    #     pairs with a seed-0 run:
    #
    #       normuon+GNS  lr 0.01 seed 1   vs (a)'s lr 0.01 job
    #       dion3 f=1/4  lr 0.01 seed 1   vs 17733554                  3.3074
    ARCH=(--grad_clip 1.0 --mlp_type swiglu --attn_gate --norm_gain)
    TIME_LIMIT="03:30:00"   # stage 7's 1907-step jobs took 2h12m-2h18m
    # use_polar_express / use_gram_newton_schulz never reach wandb's config, so
    # the normuon jobs carry a gns tag to tell them apart from stage 7's normuon.
    BASE_TAGS=",grad-clip,swiglu,attn-gate,norm-gain"
    TAG_EXTRA="${BASE_TAGS},gns,lr-sweep"
    for lr in 0.005 0.01 0.02 0.04; do
        launch normuon "$lr" 0.95 "${ARCH[@]}" "${GNS_FLAGS[@]}" "$@"
    done
    TAG_EXTRA="${BASE_TAGS},lr-sweep"
    launch dion3 0.04 0.95 "${ARCH[@]}" "${DION_FLAGS[@]}" "$@"
    TAG_EXTRA="${BASE_TAGS},gns,seed1"
    launch normuon 0.01 0.95 "${ARCH[@]}" "${GNS_FLAGS[@]}" --seed 1 "$@"
    TAG_EXTRA="${BASE_TAGS},seed1"
    launch dion3   0.01 0.95 "${ARCH[@]}" "${DION_FLAGS[@]}" --seed 1 "$@"
    TIME_LIMIT="02:50:00"; TAG_EXTRA=""
    ;;
  9)
    # The paper's rule: the optimal lr for fraction f is eta/sqrt(f), eta being
    # NorMuon's (f=1) optimum. f=1/16 is where it predicts the largest shift, 4x.
    # Under this recipe at 1907 steps NorMuon+GNS is best at 0.02 (stage 8:
    # 0.005 3.2681, 0.01 3.2377, 0.02 3.2235, 0.04 3.2290), so the rule puts
    # f=1/16's optimum at 0.08. f=1/4 already disagreed -- predicted 0.04, observed
    # 0.01-0.02 (0.01 3.3074, 0.02 3.3097, 0.04 3.3275).
    #
    #   dion3 f=1/16  lr 0.01 / 0.02 / 0.04 / 0.08    no counterparts at this recipe
    #
    # Nearest older point: 17732705 (f=1/16 lr 0.01, clipped, old architecture)
    # 3.5526. Same flags as stage 8's dion3 runs apart from the fraction.
    ARCH=(--grad_clip 1.0 --mlp_type swiglu --attn_gate --norm_gain)
    TIME_LIMIT="03:30:00"
    TAG_EXTRA=",grad-clip,swiglu,attn-gate,norm-gain,lr-rule,f0.0625"
    for lr in 0.01 0.02 0.04 0.08; do
        launch dion3 "$lr" 0.95 "${ARCH[@]}" --ortho_fraction 0.0625 "${GNS_FLAGS[@]}" "$@"
    done
    TIME_LIMIT="02:50:00"; TAG_EXTRA=""
    ;;
  10)
    # Stage 7's 10B runs all used lr 0.01, but under this recipe normuon is best at
    # 0.02 at 1B tokens (stage 8: 0.01 3.2377 vs 0.02 3.2235). So the +0.0116 gap at
    # 10B (17733556 2.8958 vs 17733557 2.9080) may be measured against an
    # under-tuned normuon. This is the same job as 17733556 with --lr 0.02, and it
    # also pairs with dion3 at lr 0.02 (17733558, 2.9074), which has no normuon partner.
    #
    # --checkpoint_steps keeps permanent checkpoints (checkpoint-<step>, never
    # overwritten, unlike --checkpoint_freq's rolling one) at the points where a
    # shorter run's cooldown would begin. A 10B curve read at 1B tokens is not a
    # 1B-token run: it has not been annealed. Branching from these gives real
    # endpoints at 0.5/1/2/4/8B tokens, ~20% of each budget in extra compute.
    #
    #   branch step   simulates       cooldown
    #        763      0.5B tokens     954 steps total
    #       1526      1B  tokens     1907
    #       3052      2B  tokens     3815
    #       6103      4B  tokens     7629
    #      12207      8B  tokens    15259
    #
    # To run one (the schedule is rebuilt from --num_iterations and fast-forwarded
    # to the loaded step, so the cooldown lands where the table says):
    #   sbatch --job-name=normuon_418m --time=02:00:00 submit_train.slurm \
    #       --config configs/normuon_418m.yaml --optimizer normuon --lr 0.02 --mu 0.95 \
    #       --grad_clip 1.0 --mlp_type swiglu --attn_gate --norm_gain \
    #       --use_polar_express --use_gram_newton_schulz --warmup_ratio 0.02 \
    #       --num_iterations 1907 --no_wandb --checkpoint_freq 0 \
    #       --checkpoint_dir <parent job dir> --checkpoint_name checkpoint-1526
    # Caveat: warmup stays the parent's 381 steps, where a native 1907-step run
    # would use 5% = 95, so the shortest branches are not exact replicas.
    #
    # ~7 GB per checkpoint, so ~35 GB for the five, plus the rolling one.
    ARCH=(--grad_clip 1.0 --mlp_type swiglu --attn_gate --norm_gain)
    TIME_LIMIT="26:00:00"          # stage 7's 10B runs took ~22h
    TOKENS_TAG="10b-tokens"
    TAG_EXTRA=",grad-clip,swiglu,attn-gate,norm-gain,gns,lr-tune"
    launch normuon 0.02 0.95 "${ARCH[@]}" "${GNS_FLAGS[@]}" \
        --num_iterations 19073 --warmup_ratio 0.02 \
        --checkpoint_freq 2000 --checkpoint_steps 763,1526,3052,6103,12207 "$@"
    TIME_LIMIT="02:50:00"; TOKENS_TAG="1b-tokens"; TAG_EXTRA=""
    ;;
  *) echo "error: STAGE must be 1, 2, 3, 4, 5, 6, 7, 8, 9 or 10" >&2; exit 1 ;;
esac
