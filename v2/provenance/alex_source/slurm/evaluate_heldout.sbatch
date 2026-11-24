#!/bin/bash -l
#SBATCH --job-name=koor-heldout
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-koor-heldout-%j.log
#SBATCH --export=NONE

set -eo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base cuda/12.8.1
cd "$HOME/Koor-building-reconstruction-v2"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Evaluation job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
nvidia-smi

# Slurm selects the assigned GPU; no manually configured CUDA_VISIBLE_DEVICES.
srun "$HOME/Koor-building-reconstruction-v2/.venv/bin/python" -u \
  scripts/evaluate_v1_v2.py \
  --batch-size 16 \
  --workers 4 \
  --output "evaluations/v1-v2-heldout-$SLURM_JOB_ID"

echo "PASS: held-out evaluation batch job finished at $(date -Is)"
