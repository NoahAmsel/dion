#!/usr/bin/env bash
# Optimizer x learning-rate sweep on the 418M model (model_dim
# 1280, n_layer 18), every run capped at 1B training tokens. 418,283,520
# parameters with tie_embeddings on -- hence the 418m wandb tag, which also
# keeps these runs distinguishable from the earlier untied 482,672,640-parameter
# generation that ran without warmup and under adjust_lr: spectral_norm.
#
# Two loops, so each family gets its own ladder:
#   * muon / normuon        -- plain full-spectrum orthogonalization
#   * dion2 / dion3         -- low-rank, run with
#                              --ortho_fraction 0.25 --use_polar_express
#                              --use_gram_newton_schulz
#
# Orthogonalizing only a fraction f of the spectrum shrinks the update norm, so
# the dion ladder is shifted up relative to the muon one. Both ladders are
# written inline in their loops; nothing else depends on their length.
#
# device_batch_size stays at the config's 64. Probe job 16817211 showed 128
# OOMs on this model: 124.41 GiB allocated of a 139.80 GiB H200, dying on a
# further 12.28 GiB request in the first training forward.
#
# Timing: ~6 s/step measured, so ~95 min of stepping and ~105 min end to end.
# The 2h30 walltime is ~40% headroom.
#
#   ./sweep_normuon_418m_lr.sh
#   ./sweep_normuon_418m_lr.sh --checkpoint_freq 250   # extra args go to every run
set -euo pipefail
# Run from the repo root: sbatch resolves submit_train.slurm and --config
# relative to the cwd, and submit_train.slurm hands train.py $SLURM_SUBMIT_DIR.
cd "$(dirname "$(readlink -f "$0")")/.."

# Only dion2 / dion3 accept these: their call sites pass fraction,
# use_gram_newton_schulz and use_polar_express, while Muon and NorMuon take
# neither ortho_fraction nor use_gram_newton_schulz and would ignore them.
DION_FLAGS=(--ortho_fraction 0.25 --use_polar_express --use_gram_newton_schulz)

# The wandb tag carries the optimizer because use_polar_express and
# use_gram_newton_schulz never reach wandb's config -- train.py logs
# hp.__dict__ and those two live on cli_args.
launch() {
    local opt="$1" lr="$2"; shift 2
    echo "=== $opt lr=$lr ==="
    WANDB_TAGS="$opt,418m,1b-tokens,lr-sweep" \
    sbatch \
        --job-name="${opt}_418m" \
        --time=02:30:00 \
        submit_train.slurm \
            --config configs/normuon_418m.yaml \
            --num_iterations 953 \
            --optimizer "$opt" \
            --lr "$lr" \
            "$@"
}

# winners muon: 0.002. normuon: 0.004
for opt in muon normuon; do
    for lr in 0.0005 0.001 0.002 0.004 0.008; do
        launch "$opt" "$lr" "$@"
    done
done

for opt in dion2; do
    for lr in 0.001 0.002 0.004; do
        launch "$opt" "$lr" "${DION_FLAGS[@]}" "$@"
    done
done

for opt in dion3; do
    for lr in 0.002 0.004 0.008; do
        launch "$opt" "$lr" "${DION_FLAGS[@]}" "$@"
    done
done
