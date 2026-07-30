#!/usr/bin/env bash
#SBATCH --job-name=microbench_variants
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=01:45:00
#SBATCH --account=torch_pr_508_tandon_advanced
## #SBATCH --exclusive
## #SBATCH --mem=0
## #SBATCH --partition=gpu
#
# Single-GPU launcher: enter the Apptainer container (conda/torch) via ~/bin/singro
# and run the single-GPU sizes. Args forward to run_variants.sh (defaults to
# all-1gpu); for the 8-GPU sizes use submit_variants_8gpu.slurm instead.
#   sbatch submit_variants.slurm                      # all-1gpu, muon-family
#   sbatch submit_variants.slurm 7b                   # only 7b, muon-family
#   sbatch submit_variants.slurm all-1gpu normuon-family
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Per-job, node-local Triton/Inductor compile caches. The default cache lives on
# shared NFS (/scratch/$USER/.cache), so two concurrent jobs that compile the
# same kernel at the same moment race on the same cache entry and die with
# ESTALE / EBADHANDLE -- this killed the 1b muon and normuon jobs (14931328 /
# 14931329) at variant 2-triton, both on ns_line_2_kernel, 2 seconds apart on
# different nodes. Job-scoped local dirs remove the race; the cost is a cold
# compile per job. SINGULARITYENV_ is how fancy_sing passes env into the
# container (same mechanism it uses for CUDA_VISIBLE_DEVICES); /tmp is bound
# from the host by default, so these are node-local.
CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/dion-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$CACHE_ROOT/triton" "$CACHE_ROOT/inductor"
export SINGULARITYENV_TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export SINGULARITYENV_TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"

exec "$HOME/bin/singro" bash "$SLURM_SUBMIT_DIR/run_variants.sh" "${@:-all-1gpu}"
