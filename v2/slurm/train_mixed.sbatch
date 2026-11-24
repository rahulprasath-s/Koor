#!/bin/bash -l
#SBATCH --job-name=koor-mixed-v2
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=slurm-koor-mixed-train-%j.log
#SBATCH --export=NONE

set -eo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base cuda/12.8.1
koor_project="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "$koor_project"

if [[ ! -f "$koor_project/v1/Point2Roof/model_cfg.yaml" ]]; then
  echo "Submit this job from the Koor repository root." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Job $SLURM_JOB_ID on $(hostname) started at $(date -Is)"
nvidia-smi

koor_training_run="runs/koor-v2-mixed-distill"
koor_resume_args=()
if [[ -f "$koor_training_run/last.pt" ]]; then
  koor_resume_args=(--resume)
  echo "Resuming existing full-training run: $koor_training_run"
else
  echo "Starting new full-training run from published v1: $koor_training_run"
fi

# Slurm selects the GPU. Do not override CUDA_VISIBLE_DEVICES.
# No smoke-run batch limit or controlled one-batch pause is used here.
srun "$koor_project/.venv/bin/python" -u v2/scripts/train_mixed_distill.py \
  --epochs 15 \
  --batch-size 16 \
  --workers 4 \
  --corner-lr 0.00001 \
  --head-lr 0.0001 \
  --score-distill-weight 1.0 \
  --offset-distill-weight 1.0 \
  --temperature 2.0 \
  --min-synthetic-retention 1.0 \
  --checkpoint-every-batches 100 \
  --run-dir "$koor_training_run" \
  "${koor_resume_args[@]}"

echo "Trainer exited at $(date -Is). Read the log for COMPLETED or PAUSED SAFELY."
echo "Run directory: $koor_project/$koor_training_run"
echo "A safe pause saves progress but does not automatically submit another job."
