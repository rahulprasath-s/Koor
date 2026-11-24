#!/bin/bash -l
#SBATCH --job-name=koor-mixed-smoke
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --signal=USR1@120
#SBATCH --output=slurm-koor-mixed-smoke-%j.log
#SBATCH --export=NONE

set -eo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base cuda/12.8.1
cd "$HOME/Koor"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Job $SLURM_JOB_ID on $(hostname) started at $(date -Is)"
nvidia-smi

# Slurm selects the GPU. Do not set CUDA_VISIBLE_DEVICES yourself.
# Unique directory per submission prevents accidental checkpoint overwrites.
koor_smoke_run="runs/alex-mixed-distill-smoke-$SLURM_JOB_ID"

# First invocation: establish limited validation baseline, then save after ONE batch.
srun "$HOME/Koor/.venv/bin/python" -u scripts/train_mixed_distill.py \
  --epochs 1 --batch-size 16 --workers 0 --max-batches 2 \
  --score-distill-weight 1.0 --offset-distill-weight 1.0 --temperature 2.0 \
  --min-synthetic-retention 1.0 --checkpoint-every-batches 1 \
  --run-dir "$koor_smoke_run" --stop-after-batches 1

# Second invocation: resume at batch index 1, perform the remaining ONE batch,
# validate, and verify epoch/sample counters in the resulting checkpoint.
srun "$HOME/Koor/.venv/bin/python" -u scripts/train_mixed_distill.py \
  --epochs 1 --batch-size 16 --workers 0 --max-batches 2 \
  --score-distill-weight 1.0 --offset-distill-weight 1.0 --temperature 2.0 \
  --min-synthetic-retention 1.0 --checkpoint-every-batches 1 \
  --run-dir "$koor_smoke_run" --resume

"$HOME/Koor/.venv/bin/python" - "$koor_smoke_run/history.json" <<'PY'
import json
import sys
from pathlib import Path
history = json.loads(Path(sys.argv[1]).read_text())
if len(history) != 1 or history[0]["epoch"] != 1:
    raise RuntimeError("Smoke run did not complete exactly one limited epoch")
training = history[0]["train"]
if (training["samples"], training["synthetic_samples"], training["real_samples"]) != (32, 16, 16):
    raise RuntimeError("Unexpected sample count: resumed batches may have been replayed")
print("PASS: resumed smoke test completed exactly two training batches: 16 synthetic + 16 real samples")
PY

echo "PASS: mixed-distillation GPU smoke/resume test completed at $(date -Is)"
echo "Run directory: $HOME/Koor/$koor_smoke_run"
