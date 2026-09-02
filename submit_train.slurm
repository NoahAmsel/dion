#!/usr/bin/env bash
#SBATCH --job-name=dion_train
#SBATCH --output=/scratch/nia4240/dion/slurm_logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_508_tandon_advanced
#
# Generic train.py launcher. Sets the SLURM defaults, enters the Apptainer
# container via ~/bin/singro, and forwards every argument straight to train.py.
#
#   sbatch submit_train.slurm --config configs/normuon_160m.yaml
#   sbatch submit_train.slurm --config configs/muon_160m.yaml --lr 0.01
#   sbatch --gres=gpu:h200:8 --time=04:00:00 submit_train.slurm --config configs/dion2_160m.yaml
#   sbatch --job-name=normuon-lr03 submit_train.slurm --config configs/normuon_160m.yaml --lr 0.03
#
# Any #SBATCH default above can be overridden on the sbatch command line;
# nproc_per_node follows the GPU allocation automatically, so changing --gres is
# all it takes to change the world size. Defaults this script supplies to
# train.py (--data_dir, --checkpoint_dir, --wandb_project_name,
# --wandb_job_name) are placed *before* "$@", so repeating any of them after
# `submit_train.slurm` wins. Pass --no_wandb to turn logging off entirely.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

DATA_DIR="${DATA_DIR:-/scratch/nia4240/fineweb10B}"
RUN_ROOT="${RUN_ROOT:-/scratch/nia4240/dion}"
RUN_NAME="${SLURM_JOB_NAME:-dion_train}-${SLURM_JOB_ID}"
# Not pre-created: train.py makes it on the first save, so runs left at the
# default --checkpoint_freq 0 do not litter empty directories.
CKPT_DIR="$RUN_ROOT/checkpoints/$RUN_NAME"

# WandB. Auth comes from ~/.netrc (api.wandb.ai), which is in the bound home, so
# no API key is plumbed through. WANDB_DIR is the *parent* of the run tree --
# wandb appends "wandb/", so this puts runs in $RUN_ROOT/wandb/run-<ts>-<id>/.
# Each run dir holds run-<id>.wandb, a transaction log of every logged step, so
# the loss sequences are on local disk whether or not the sync to the cloud
# succeeds.
WANDB_ENTITY_NAME="${WANDB_ENTITY_NAME:-noahamselsteam}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-dion}"
mkdir -p "$RUN_ROOT/wandb"
export SINGULARITYENV_WANDB_DIR="$RUN_ROOT"
export SINGULARITYENV_WANDB_ENTITY="$WANDB_ENTITY_NAME"
# Optional run tags, comma-separated. sbatch forwards the submitting shell's
# environment, so `WANDB_TAGS=ablation,seed0 sbatch submit_train.slurm ...`
# works; train.py's wandb.init() takes no tags= argument, the env var is the way.
if [[ -n "${WANDB_TAGS:-}" ]]; then
    export SINGULARITYENV_WANDB_TAGS="$WANDB_TAGS"
fi

# nproc_per_node = the GPUs SLURM gave us. SLURM_GPUS_ON_NODE is set by the
# --gres request; the CUDA_VISIBLE_DEVICES fallback covers allocations made
# through other flags.
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    NGPU="$SLURM_GPUS_ON_NODE"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
else
    NGPU=1
fi

# Per-job, node-local Triton/Inductor compile caches. The default cache lives on
# shared VAST (/scratch/$USER/.cache), so two concurrent jobs that compile the
# same kernel at the same moment race on the same cache entry and die with
# ESTALE / EBADHANDLE. Job-scoped local dirs remove the race; the cost is a cold
# compile per job. SINGULARITYENV_ is how fancy_sing passes env into the
# container, and /tmp is bound from the host, so these are node-local.
CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/dion-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$CACHE_ROOT/triton" "$CACHE_ROOT/inductor"
export SINGULARITYENV_TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export SINGULARITYENV_TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"

echo "=== $RUN_NAME on $(hostname) ==="
echo "GPUs:        $NGPU  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})"
echo "data_dir:    $DATA_DIR"
echo "checkpoints: $CKPT_DIR"
echo "wandb:       $WANDB_ENTITY_NAME/$WANDB_PROJECT_NAME -> $RUN_ROOT/wandb"
echo "train args:  $*"
echo

exec "$HOME/bin/singro" \
    torchrun --standalone --nproc_per_node="$NGPU" \
        "$SLURM_SUBMIT_DIR/train.py" \
        --data_dir "$DATA_DIR" \
        --checkpoint_dir "$CKPT_DIR" \
        --wandb_project_name "$WANDB_PROJECT_NAME" \
        --wandb_job_name "$RUN_NAME" \
        "$@"
