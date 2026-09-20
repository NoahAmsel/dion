# 1B / 20B-token NorMuon vs Dion3 crossover on JUPITER

Raw per-step and per-checkpoint data for five 1B-parameter runs to 20B tokens,
for plotting. Run 2026-09-11 (three arms) and 2026-09-12 (two arms) on JUPITER,
4 nodes x 4 GH200.

## Setup

`configs/normuon_1b_4node.yaml` (copied here as `normuon_1b_4node.yaml`):
1,105,133,568 params, d=1792, L=24, H=16, untied embeddings, seq 1024,
38,147 steps x 524,288 tok = **20.0B tokens**, FineWeb100B (GPT-2 tokens),
HSDP dp=4 x fs=4, grad-accum 1, cosine warmup 1% / warmdown 25%
(**warmdown starts at step 28,610 = 15.0B tokens** - the last three validation
points of any curve are inside it).

Polar Express + Gram Newton-Schulz on **all five arms, baseline included**.
Upstream `experiments/launch_1b_crossover.sh` leaves the baseline on Triton NS,
which would confound a loss comparison, so it was changed. `selection_scope` left
at its `local` default; with `fs_size` set the params are sharded, but every arm
shares that setting.

## The arms

| arm | optimizer | f | matrix lr | scalar lr | final val loss |
|---|---|---|---|---|---|
| `normuon_f1_lr0.01` | normuon | 1 | 0.01 | 0.01 | **2.7323** |
| `dion3_f0.125_lr0.01` | dion3 | 1/8 | 0.01 | 0.01 | **2.7394** |
| `dion3_f0.125_matrixlr_sqrtf` | dion3 | 1/8 | 0.01*sqrt(8) | 0.01 | **2.7543** |
| `dion3_f0.125_scalarlr_sqrtf` | dion3 | 1/8 | 0.01 | 0.01*sqrt(8) | **2.8239** |
| `dion3_f0.125_bothlr_sqrtf` | dion3 | 1/8 | 0.01*sqrt(8) | 0.01*sqrt(8) | **2.8757** |

"matrix lr" is the orthogonalized transformer-block group; "scalar lr" is the
AdamW group (`wte`, `lm_head`). `ortho_fraction` is recorded as 1.0 for the
NorMuon arm: the config carried 0.25, but NorMuon orthogonalizes every row and
ignores it.

## Files

| file | rows | what |
|---|---|---|
| `runs.csv` | 5 | one row per arm: config, final loss, mean ms/step, mean MFU |
| `val_curves.csv` | 390 | 78 validation points per arm: step, tokens, val loss, ms/step, MFU |
| `per_step/<arm>.csv.gz` | 38,147 each | every training step: train loss, lr, grad norm, ms/step, MFU, tokens/s, cumulative train ms, wall seconds |
| `dion3_vs_normuon_lr0.01.png` | - | the two matched-lr arms, and their gap |

`pandas.read_csv` reads the `.csv.gz` files directly. MFU is in **percent** in
both CSVs. `tokens = step * 524288`. Validation ran every 500 steps, so
`val_curves.csv` is the natural x-axis for loss plots; `per_step` is there for
grad-norm, lr-schedule and throughput plots.

Extracted from the runs' wandb transaction logs, so these are the logged values,
not a re-derivation. wandb run ids are in `runs.csv`.

## What the data shows

**1. `eta/sqrt(f)` is worse than base lr at this scale, and most of that is the
AdamW groups, not the rule.** In this harness `--lr` was a single global knob
until commit `1f42397`, so an "adjusted LR" arm also scaled AdamW on `wte` and
`lm_head` (2 x 1792 x 50304 = 180M of 1.105B params, and `lm_head` sets the logit
scale). The two 2026-09-12 arms decouple them, against the `f=1/8, lr 0.01` arm:

| what was scaled by sqrt(8) | final val | delta |
|---|---|---|
| matrix group only | 2.7543 | +0.0149 |
| scalar group only | 2.8239 | +0.0845 |
| both | 2.8757 | +0.1363 |

So the scalar coupling carries most of the penalty, and the two are
superadditive (0.0149 + 0.0845 = 0.0994 against 0.1363 jointly). The rule itself
still costs +0.0149, so it is not vindicated - but any "adjusted LR" number taken
from a harness with one global lr knob, including earlier 418M sweeps, carries
the confound and overstates the damage.

This is consistent with the public code: `adjust_lr_spectral_norm(lr, X[0].shape,
flatten)` (`dion/megabatch_base.py:1047`) computes `lr * sqrt(fan_out/fan_in)`
from the **full** parameter shape and never sees `ortho_fraction`. The polar/NS
output for the selected k x m block has orthonormal rows, so per-row RMS is
1/sqrt(m) independent of k, and NorMuon's Frobenius-preserving rescale is
norm-neutral in f for the same reason. Each *selected* row moves the same amount
regardless of f; only the update frequency changes, so there is nothing for
1/sqrt(f) to compensate.

**2. The crossover is delayed, not absent.** Gap of `dion3 f=1/8 lr 0.01` over
NorMuon, pre-warmdown:

| tokens | 2.1B | 4.2B | 6.3B | 8.4B | 10.5B | 12.6B | 14.7B |
|---|---|---|---|---|---|---|---|
| gap | +0.0685 | +0.0346 | +0.0264 | +0.0144 | +0.0122 | +0.0092 | +0.0062 |

Monotone down at roughly -0.003 per 2B over the last points, extrapolating to
zero near 19-20B, against ~10B in fig 4. The last three points (16.8B, 18.9B,
20.0B: +0.0080, +0.0071, +0.0071) are **inside the warmdown**, where the gap
compresses for reasons unrelated to the trend - do not read them as flattening.

**3. The end-to-end speedup at this scale is 3%, not 3.7x.** 528.8 vs 545.7
ms/step. The paper's larger factors are optimizer-step-only on much bigger
models; at 1B / seq 1024 the optimizer is a small share of step time.

## Reproducing

The runs used a local branch off `657d3f5` carrying a 4-node JUPITER launcher,
matched GNS backend, MFU logging, and `1f42397` which adds `--scalar_lr` to
decouple the AdamW group from the matrix LR. That harness change is what made the
two decomposition arms possible and is not in this PR, which is data only. Happy
to send it separately if useful.

Not included: the 9 GB/arm checkpoints and the ~45 MB/arm raw wandb directories.
Available on request.
