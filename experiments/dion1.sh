#!/usr/bin/env bash
# Sanity sweep for the original Dion (--optimizer dion) at 225M / 2.5B tokens,
# under the launch_size_budget.sh recipe (configs/normuon_225m.yaml), one GPU
# each, default precision.
#
#   #  optimizer  ortho_fraction  lr     mu    question
#   1  muon       --              0.01   0.95  reference: full-rank Dion is Muon-like
#   2  dion       1.0             0.01   0.95  full rank should land near 1
#   3  dion       0.125           0.005  0.95  lr ladder at f=1/8: does
#   4  dion       0.125           0.01   0.95    NorMuon's lr 0.01 transfer?
#   5  dion       0.125           0.02   0.95
#   6  dion       0.125           0.01   0.9   mu: see below
#   7  dion       0.125           0.01   0.8
#
# Existing 2.5B results to compare against (branches of the 20B parents):
#   normuon_225m_2p5b-18240103  3.2669
#   dion3_225m_2p5b-18240108    3.3288  (f=1/8, mu 0.95 -- same f as 3-7)
#
# mu. In Dion, as in Dion2/Dion3, mu is an error-feedback decay: M += G, then
# only the part of M captured this step (span of P for Dion, the selected rows
# for Dion2/Dion3) is scaled by mu; the residual carries over undecayed. At
# f=1/8 most directions are not captured on a given step, so momentum forgets
# more slowly than mu suggests, and an early spike lingers (the stage 7 note in
# sweep_normuon_418m_lr.sh). If anything that argues for a smaller mu than
# Muon's. Past evidence is Muon/NorMuon only (418M, 1B tokens, lr 0.01): mu 0.9
# and 0.95 were within 0.003 and 0.98 cost 0.03-0.04, so 0.98 is not worth a
# job and the sweep goes down from 0.95 instead. Dion2/Dion3 were only ever run
# at 0.95.
#
# Schedule: from scratch, 4768 steps. --warmup_ratio 0.08 gives
# round(0.08 * 4768) = 381 warmup steps, the 20B parents' warmup that the
# 2.5B branches inherited; warmdown is round(0.25 * 4768) = 1192 either way.
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
#
#   ./experiments/dion1.sh                       # 7 jobs
#   ./experiments/dion1.sh --seed 1              # extra args go to every run
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

DATA_DIR="${DATA_DIR:-/scratch/$USER/fineweb100B}"
JOBS=(  # optimizer ortho_fraction lr mu
    "muon -     0.01  0.95"
    "dion 1.0   0.01  0.95"
    "dion 0.125 0.005 0.95"
    "dion 0.125 0.01  0.95"
    "dion 0.125 0.02  0.95"
    "dion 0.125 0.01  0.9"
    "dion 0.125 0.01  0.98"
)

for job in "${JOBS[@]}"; do
    read -r opt frac lr mu <<<"$job"
    flags=(); name="${opt}_225m_2p5b"
    if [[ "$frac" != - ]]; then flags=(--ortho_fraction "$frac"); name+="_f$frac"; fi
    name+="_lr${lr}_mu$mu"
    id=$(WANDB_TAGS="$opt,225m,2p5b-tokens,dion1-sanity,f$frac,lr$lr,mu$mu" \
        sbatch --parsable --job-name="$name" --time=05:00:00 \
        submit_train.slurm --data_dir "$DATA_DIR" \
        --config configs/normuon_225m.yaml --optimizer "$opt" --lr "$lr" --mu "$mu" \
        --num_iterations 4768 --warmup_ratio 0.08 --checkpoint_freq 0 \
        "${flags[@]}" "$@" | tail -1)
    echo "=== $name: job ${id%%;*} ==="
done
