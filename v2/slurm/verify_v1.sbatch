#!/bin/bash -l
#SBATCH --job-name=koor-v1-check
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-koor-v1-%j.log
#SBATCH --export=NONE

set -eo pipefail
unset SLURM_EXPORT_ENV

module load python/3.12-base cuda/12.8.1
koor_project="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "$koor_project/models/koor-v1"

if [[ ! -f "$koor_project/v1/Point2Roof/model_cfg.yaml" ]]; then
  echo "Submit this job from the Koor repository root." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date -Is)"
nvidia-smi

# Slurm selects the GPU; do not set CUDA_VISIBLE_DEVICES manually.
srun "$koor_project/.venv/bin/python" -u verify_release.py --device cuda:0

echo "PASS: Koor v1 release verification completed at $(date -Is)"
