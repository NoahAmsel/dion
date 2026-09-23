# Size x budget batch, 2026-09-22: NorMuon vs Dion3 (f = 1/8)

Launched with `experiments/launch_size_budget.sh` at commit `69c4e8d`, wandb
project `noahamselsteam/dion`, all runs tagged `size-budget`.

This file exists because **wandb's config is `hp.__dict__` and omits 15 CLI-only
flags**, two of which change the math: `--use_polar_express` and
`--use_gram_newton_schulz`. Worse, for this batch those two were set in the YAML
rather than on the command line, so they appear in neither `config` nor the
`argv` recorded in wandb metadata. What follows is the record.

## Flags not in wandb's config, and their values for every run here

| flag | value | where it came from |
|---|---|---|
| `use_polar_express` | **true** | the config files below |
| `use_gram_newton_schulz` | **true** | the config files below |
| `no_triton` | false (Triton NS available) | default |
| `no_compile` | false (torch.compile on) | default |
| `dp_size` / `fs_size` / `tp_size` | unset -> single GPU, DDP world size 1 | default |
| `fast_fsdp` | false | default |
| `debug`, `time_optimizer`, `no_wandb` | false | default |
| `wandb_job_name` | `<job name>-<job id>` | `submit_train.slurm` |
| `config` | `configs/normuon_225m.yaml`, `configs/normuon_418m_20b.yaml` | launcher |
| `checkpoint_name` | `checkpoint-<branch step>` on branches only | launcher |
| `fresh_wandb_run` | true on branches, false on parents | launcher |

Both arms run the same orthogonalization routine on purpose: leaving NorMuon on
Triton Newton-Schulz while Dion3 uses Polar Express + GNS confounds the
comparison by ~0.009 nats (job 17368303 vs 17002705).

## Shared setup

`--data_dir /scratch/nia4240/fineweb100B` (210 shards, 21.0B tokens, single
epoch), seq 1024 x batch 512 = 524,288 tokens/step, untied embeddings, ReLU^2
MLP, **no gradient clipping**, 1% warmup (381 steps), 25% warmdown to 10% of
peak, wd 0.01 (matrix only), mu 0.95, muon_beta2 0.95, `adjust_lr:
spectral_norm`, seed 0, one H200/GH200 per job. Dion3 arms add
`--ortho_fraction 0.125`; the value in NorMuon's config is inert, since NorMuon
orthogonalizes every row.

Chosen to match the external 1B/20B runs (`configs/normuon_1b_4node.yaml`), so
225M, 418M and their 1.105B sit on one axis. It is deliberately not our best
recipe -- SwiGLU, the attention gate, norm gains and clipping all help.

## Jobs and results

Parents run 38,147 steps (20.0B tokens) and keep permanent checkpoints at steps
1788 / 3576 / 7153 / 14305. Each branch resumes one of those and runs to the
total whose 25% cooldown starts exactly there, so every budget below is a
properly annealed endpoint rather than a mid-run reading.

### 225M (`configs/normuon_225m.yaml`, d896 L14 H7, 225,017,856 params)

| budget | steps | NorMuon | job | Dion3 f=1/8 | job | gap |
|---|---|---|---|---|---|---|
| 1.25B | 2384 | 3.4052 | 18240101 | 3.5496 | 18240107 | +0.1444 |
| 2.5B | 4768 | 3.2669 | 18240103 | 3.3288 | 18240108 | +0.0619 |
| 5B | 9537 | 3.1658 | 18240104 | 3.1890 | 18240111 | +0.0232 |
| 10B | 19073 | 3.0871 | 18240105 | 3.0910 | 18240112 | +0.0039 |
| 20B | 38147 | 3.0276 | **18240100** | **3.0252** | **18240106** | **-0.0024** |

### 418M (`configs/normuon_418m_20b.yaml`, d1280 L18 H10, 482,672,640 params)

| budget | steps | NorMuon | job | Dion3 f=1/8 | job | gap |
|---|---|---|---|---|---|---|
| 1.25B | 2384 | 3.2767 | 18241566 | 3.4252 | 18241571 | +0.1485 |
| 2.5B | 4768 | 3.1242 | 18241567 | 3.1984 | 18241572 | +0.0742 |
| 5B | 9537 | 3.0120 | 18241568 | 3.0472 | 18241573 | +0.0352 |
| 10B | 19073 | 2.9238 | 18241569 | 2.9374 | 18241574 | +0.0136 |
| 20B | 38147 | 2.8588 | **18241565** | 2.8624 | **18241570** | +0.0036 |

Bold job ids are the parents; the rest are branches off them.

### Learning-rate probes (225M, 20B tokens, lr 0.005, no branches)

| optimizer | job | loss | vs the same run at lr 0.01 |
|---|---|---|---|
| normuon | 18240114 | 3.0532 | +0.0256 |
| dion3 f=1/8 | 18240116 | 3.0417 | +0.0165 |

lr 0.005 is worse for both, so the optimum is not below 0.01 at this budget, and
the penalty is larger for NorMuon -- lr 0.01 is not flattering Dion3.

## Reading it

The gap falls monotonically with budget at both sizes, crossing zero for 225M at
20B tokens. At fixed tokens it grows with model size (-0.0024 at 225M, +0.0036 at
418M, +0.0071 for the external 1.105B), but the two sizes agree closely once
lined up by tokens per parameter: 225M at 22 tok/param is +0.0232 against 418M at
20 giving +0.0136, and 225M at 44 is +0.0039 against 418M at 41 giving +0.0036.
So the gap tracks tokens per parameter, not parameter count.

## Reproducing

    git checkout 69c4e8d
    sbatch submit_download.slurm 210          # FineWeb100B, ~39 GB
    MODELS=225m ./experiments/launch_size_budget.sh
    MODELS=418m SKIP_PROBES=1 ./experiments/launch_size_budget.sh
