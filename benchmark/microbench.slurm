#!/bin/bash

#SBATCH --job-name=microbench
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=00:30:00

# set -euo pipefail

# Run the script from the dion root directory

# Ensure logs directory exists (safe even if already present)
mkdir -p slurm_logs

python -m benchmark.benchmark_newton_schulz --batch_size 1 --expansion 1 --plot
python -m benchmark.benchmark_newton_schulz --batch_size 1 --expansion 2 --plot
python -m benchmark.benchmark_newton_schulz --batch_size 1 --expansion 3 --plot
python -m benchmark.benchmark_newton_schulz --batch_size 1 --expansion 4 --plot
python -m benchmark.benchmark_newton_schulz --batch_size 1 --expansion 8 --plot
python -m benchmark.benchmark_newton_schulz --batch_size 1 --expansion 16 --plot
python benchmark/microplot.py
