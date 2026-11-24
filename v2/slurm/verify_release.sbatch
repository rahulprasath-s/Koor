#!/bin/bash -l
#SBATCH --job-name=koor-v2-release
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-koor-v2-release-%j.log
#SBATCH --export=NONE

set -eo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base cuda/12.8.1
koor_project="${SLURM_SUBMIT_DIR:?Submit from the project root}"
cd "$koor_project/release/Koor-building-reconstruction-v2"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Release verification job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
nvidia-smi

# Use Slurm's allocated GPU; no manual CUDA_VISIBLE_DEVICES override.
srun "$koor_project/.venv/bin/python" -u verify_release.py --device cuda:0

echo "PASS: v2 package GPU verification completed at $(date -Is)"
echo "Report: $koor_project/release/Koor-building-reconstruction-v2/release_verification.json"
