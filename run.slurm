#!/bin/bash
# Directives for Slurm. Adjust partition/time/CPUs as needed for your cluster.
# Job name, output and error files (creates slurm_logs/ directory entries by job name and id)
# %x = job name, %j = job id
# Example: sbatch run.sbatch
# Resources: 1 node, 4 GPUs (for --nproc_per_node=4)
# Tune --cpus-per-task and --time to match your environment and dataset size.
#
# FOR NOW, I'm just using 1 gpu...

#SBATCH --job-name=dion_train
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --nodes=1
# NOPE SBATCH --ntasks-per-node=4
# NOPE SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=06:00:00

set -euo pipefail

# Ensure logs directory exists (safe even if already present)
mkdir -p slurm_logs

# # Load environment
# echo "yay this is a test"
# singro "conda activate dion && cd ~/dion && torchrun --standalone --nproc_per_node=1 train.py --config configs/dion_160m.yaml"

# we're using ${MY_EXT3_PATH} and ${MY_SIF}
singularity exec --nv \
--overlay /scratch/nia4240/overlay-50G-10M.ext3:ro \
/scratch/work/public/singularity//cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
/bin/bash -c "
unset -f which
source /ext3/env.sh
source /ext3/slurm/env.sh
conda activate dion
cd ~/dion
torchrun --standalone --nproc_per_node=1 train.py --config configs/fron_160m.yaml --random_cols=true
"

# # Run distributed training with torchrun (4 processes / GPUs on the node)
# torchrun --standalone --nproc_per_node=4 train.py --config configs/dion_160m.yaml
