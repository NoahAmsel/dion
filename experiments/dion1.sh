#!/usr/bin/env bash
# Sanity sweep for the original Dion (--optimizer dion) at 225M / 2.5B tokens,
# under the launch_size_budget.sh recipe (configs/normuon_225m.yaml), one GPU
# each, default precision.
#
#   STAGE=1 ./experiments/dion1.sh   # 7 jobs: references, lr ladder, mu
#   STAGE=2 ./experiments/dion1.sh   # 2 jobs: extend the f=1/8 lr ladder
#   STAGE=3 ./experiments/dion1.sh   # 2 jobs: mu at the tuned f=1/8 lr
#   STAGE=2 ./experiments/dion1.sh --seed 1   # extra args go to every run
#
# Existing 2.5B results to compare against (branches of the 20B parents):
#   normuon_225m_2p5b-18240103  3.2669
#   dion3_225m_2p5b-18240108    3.3288  (f=1/8, lr 0.01, mu 0.95)
#
# STAGE 1 (jobs 22831823-22831829), final val_loss:
#   optimizer  ortho_fraction  lr     mu    loss
#   muon       --              0.01   0.95  3.2702  reference: full-rank Dion is Muon-like
#   dion       1.0             0.01   0.95  3.2677  -0.0025 vs muon: full rank is sound
#   dion       0.125           0.005  0.95  3.4300
#   dion       0.125           0.01   0.95  3.3587
#   dion       0.125           0.02   0.95  3.3248  best, but at the top of the ladder
#   dion       0.125           0.01   0.9   3.3788
#   dion       0.125           0.01   0.98  3.3552
#
# At f=1/8 loss falls monotonically with lr through 0.02, so the optimum is not
# bracketed. Expected: Dion's update has Frobenius norm ~sqrt(r), so at f=1/8 a
# step is sqrt(0.125) = 0.35x the full-rank one, and 0.01/sqrt(f) = 0.028 would
# restore it. (Dion3 did not show this -- stage 3 of sweep_normuon_418m_lr.sh
# found its optimum flat in f.)
#
# STAGE 2 extends the f=1/8 ladder at mu 0.95 to 0.028 (= 0.01/sqrt(f)) and
# 0.04, enough to bracket the optimum or show it is higher still. mu is
# re-tested only after this, at the winning lr: stage 1 measured it at lr 0.01,
# which is now known to be too low.
#
# STAGE 2 (jobs 22875501, 22875503), final val_loss, with stage 1's ladder:
#   lr     0.005   0.01    0.02    0.028   0.04
#   loss   3.4300  3.3587  3.3248  3.3230  3.3305
# Bracketed: 0.02 and 0.028 tie within seed noise (~0.002 at 5B tokens, see
# seed_variance.sh), 0.04 is past the optimum. The optimum sits near
# 0.01/sqrt(f) = 0.028, as the sqrt(r) update norm predicts. Mid-run 0.028
# trailed 0.02 by 0.023 at step 3501 and caught up in the cooldown, so
# pre-cooldown losses are a poor guide to lr here.
#
# STAGE 3 re-tests mu at lr 0.028, the tuned f=1/8 rate (0.02 ties it; 0.028
# is chosen because its mu 0.95 run already exists, 22875501, 3.3230). Stage 1
# measured mu at lr 0.01: 0.9 +0.020, 0.98 -0.0035 against 0.95. If that holds
# at the right lr, 0.98 is a real gain and 0.9 a real loss.
#
# STAGE 3 (jobs 22948617, 22948619), final val_loss at lr 0.028:
#   mu     0.9     0.95    0.98
#   loss   3.3229  3.3230  3.3468
# It did not hold: 0.9 ties 0.95 and 0.98 costs +0.024, the reverse of stage 1.
# mu and lr interact -- at the too-low lr 0.01 a higher mu partly made up for
# it -- so stage 1's mu ranking was an lr artifact. The tuned picture matches
# Muon/NorMuon's (0.9 ~ 0.95, 0.98 clearly worse): keep mu 0.95.
#
# In Dion, as in Dion2/Dion3, mu is an error-feedback decay: M += G, then
# only the part of M captured this step (span of P for Dion, the selected rows
# for Dion2/Dion3) is scaled by mu; the residual carries over undecayed. At
# f=1/8 most directions are not captured on a given step, so momentum already
# forgets more slowly than mu suggests -- which argued for a smaller mu. Stage 1
# says otherwise: 0.9 cost +0.020 and 0.98 gained -0.0035 against 0.95, the
# latter within likely seed noise. Prior evidence was Muon/NorMuon only (418M,
# 1B tokens, lr 0.01), where 0.9 and 0.95 were within 0.003 and 0.98 cost
# 0.03-0.04; Dion2/Dion3 were only ever run at 0.95.
#
# Schedule: from scratch, 4768 steps. --warmup_ratio 0.08 gives
# round(0.08 * 4768) = 381 warmup steps, the 20B parents' warmup that the
# 2.5B branches inherited; warmdown is round(0.25 * 4768) = 1192 either way.
# Runtime: muon 1.45 s/step (~1h55m), dion f=1 1.81 s/step (~2h25m), dion
# f=1/8 1.54 s/step (~2h05m).
#
# Gotchas, all handled here:
#   - normuon_225m.yaml sets no ortho_fraction, so Dion would silently run at
#     train.py's default of 0.25. Every Dion job passes it explicitly.
#   - Dion applies sqrt(d_out/d_in) itself and ignores --adjust_lr, nesterov,
#     muon_beta2 and the Polar Express / GNS flags. Its AdamW groups use Dion's
#     own default betas (0.9, 0.95), the recipe's values; --adamw_beta1/2 would
#     have no effect.
#   - Dion zeroes NaNs and its Cholesky does not raise, so a numerical failure
#     shows up as a loss that stops falling, not a crash.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

DATA_DIR="${DATA_DIR:-/scratch/$USER/fineweb100B}"
case "${STAGE:?set STAGE=1, 2 or 3}" in
    1) JOBS=(  # optimizer ortho_fraction lr mu
        "muon -     0.01  0.95"
        "dion 1.0   0.01  0.95"
        "dion 0.125 0.005 0.95"
        "dion 0.125 0.01  0.95"
        "dion 0.125 0.02  0.95"
        "dion 0.125 0.01  0.9"
        "dion 0.125 0.01  0.98"
    ) ;;
    2) JOBS=(
        "dion 0.125 0.028 0.95"
        "dion 0.125 0.04  0.95"
    ) ;;
    3) JOBS=(
        "dion 0.125 0.028 0.9"
        "dion 0.125 0.028 0.98"
    ) ;;
    *) echo "error: unknown STAGE=$STAGE" >&2; exit 1 ;;
esac

for job in "${JOBS[@]}"; do
    read -r opt frac lr mu <<<"$job"
    flags=(); name="${opt}_225m_2p5b"
    if [[ "$frac" != - ]]; then flags=(--ortho_fraction "$frac"); name+="_f$frac"; fi
    name+="_lr${lr}_mu$mu"
    id=$(WANDB_TAGS="$opt,225m,2p5b-tokens,dion1-sanity,stage$STAGE,f$frac,lr$lr,mu$mu" \
        sbatch --parsable --job-name="$name" --time=05:00:00 \
        submit_train.slurm --data_dir "$DATA_DIR" \
        --config configs/normuon_225m.yaml --optimizer "$opt" --lr "$lr" --mu "$mu" \
        --num_iterations 4768 --warmup_ratio 0.08 --checkpoint_freq 0 \
        "${flags[@]}" "$@" | tail -1)
    echo "=== $name: job ${id%%;*} ==="
done
