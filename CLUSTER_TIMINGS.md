# Optimizer-step timings on B200 (cjob cluster)

Timings produced with this branch's benchmarking scripts on 8×B200 nodes, run via cjob
(the drivers `run_bench_pod.sh` / `run_benchopt_pod.sh` are the only additions — a cluster
adapter, since `submit_all.sh` targets SLURM which this cluster doesn't have). All numbers are
the median over the timed window; bf16.

## Which script for which setting

| setting | script | why |
|---|---|---|
| **single-GPU** | `profile_training_step.py` (real fwd/bwd/opt step, `--cuda_graph`) | the real-step measurement; works |
| **8-GPU** | `benchmark_optimizer.py sweep` (isolated `optimizer.step()`, **eager**) | see finding below — `profile_training_step`'s 8-GPU cuda-graph path crashes |

### Finding: `profile_training_step.py` crashes at 8-GPU (cuda-graph capture)

At `--fs_size 8` with `--cuda_graph`, every run executes ~18 eager steps fine, then hits a
**CUDA illegal memory access on the higher ranks (5/6/7) at the graph capture step** (surfaced
at `torch.cuda.synchronize()`). It reproduces on the `--no_triton` baseline, so it is in the
**megabatch-capture path**, not the triton kernel. `benchmark_optimizer.py` avoids it only
because it runs **eager** (no capture). So the 8-GPU numbers here are **eager**, not the
cuda-graph distributed numbers of the paper's `tab:opt-step-distributed`. Reproducing the
cuda-graph 8-GPU numbers needs this fixed (likely the dion int64 batched-pointer-offset fix,
microsoft/dion #102/#103, which this branch predates).

## Single-GPU — `profile_training_step.py`, real step, CUDA-graph opt-step (gpu_opt_ms)

`cpu_opt_ms ≈ 0.03–0.5 ms` on every row confirms the graph replay is active.

**Muon family**

| size | baseline | +cutlass | +gns | +both | dion2-½ | dion2-¼ | full-stack |
|------|--------:|--------:|-----:|------:|-------:|-------:|-----:|
| 1B | 38.7 | 31.5 | 34.9 | 29.6 | 16.5 | 11.0 | 3.5× |
| 3B | 146.3 | 112.6 | 125.7 | 101.8 | 49.8 | 30.2 | 4.8× |
| 4B | 217.5 | 170.9 | 184.9 | 147.7 | 67.4 | 40.0 | 5.4× |
| 7B | 711.2 | 529.6 | 604.9 | 466.2 | 173.5 | 90.5 | **7.9×** |

**NorMuon family**

| size | baseline | +cutlass | +gns | +both | dion2-½ | dion2-¼ | full-stack |
|------|--------:|--------:|-----:|------:|-------:|-------:|-----:|
| 1B | 44.0 | 36.6 | 40.1 | 34.7 | 22.5 | 14.5 | 3.0× |
| 3B | 162.7 | 129.3 | 143.0 | 117.8 | 69.3 | 41.7 | 3.9× |
| 4B | 239.4 | 194.9 | 205.9 | 171.1 | 95.9 | 56.7 | 4.2× |
| 7B | OOM | OOM | OOM | OOM | 255.7 | 144.7 | — |

7B NorMuon unfiltered OOMs on one GPU (full orthogonalization + per-row variance normalization
under capture exceeds 178 GB even at `device_batch_size=1`); only the filtered rows fit. 7B Muon
fits at `device_batch_size=1` (batch is irrelevant to the opt-step).

## 8-GPU — `benchmark_optimizer.py sweep`, **eager**, isolated (GPU median ms)

Its native sweep ladder (Muon family; NorMuon not in the default `SWEEP_OVERRIDES`). 3B is
absent: its `n_head=18` is not divisible by the sharding world_size 8 (the `+split heads`
variant raises `ValueError`).

| size | baseline | +kernels | +GNS | +split | dion 0.5 | dion 0.25 | dion 0.125 |
|------|--------:|--------:|-----:|------:|--------:|---------:|----------:|
| 1B | 8.3 | 11.0 | 12.0 | 11.3 | 21.7 | 21.2 | 19.1 |
| 4B | 31.8 | 27.3 | 25.2 | 21.7 | 29.3 | 28.6 | 29.2 |
| 7B | 80.8 | 65.9 | 58.0 | 45.6 | 29.3 | 26.4 | 26.9 |
| 14B | 188.2 | 152.3 | 130.9 | 100.2 | 59.4 | 42.3 | 32.4 |

**Eager signature:** at 1B–4B the filtered `dion` rows are **slower** than baseline (their
selection / error-feedback is host-dispatch-bound in eager execution), and only at 7B/14B does
filtering win (188→32 ms, 5.8× at 14B). This is exactly the eager-vs-graph contrast the paper's
appendix describes — and the reason the production tables use cuda-graph capture.

## Reproduce

```bash
# single-GPU (real step, cuda-graph):   8-GPU node, uses 1 GPU
bash run_bench_pod.sh all-1gpu            # both families; DBS=1 for 7B fit
# 8-GPU (eager isolated sweep):
bash run_benchopt_pod.sh "1b 4b 7b 14b"   # Muon ladder; resumable via JOB_OUT_DIR
```
