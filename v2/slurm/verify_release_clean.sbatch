#!/bin/bash -l
#SBATCH --job-name=koor-v2-clean
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-koor-v2-clean-%j.log
#SBATCH --export=NONE

set -eo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base cuda/12.8.1
koor_project="${SLURM_SUBMIT_DIR:?Submit from the project root}"
koor_release="$koor_project/release/Koor-building-reconstruction-v2"
koor_clean_root=$(mktemp -d "${TMPDIR:?Node-local TMPDIR must be available}/koor-v2-check.XXXXXX")
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PIP_CACHE_DIR="$koor_clean_root/pip-cache"
# Alex compute nodes use the documented outbound proxy for package downloads.
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

echo "Clean release check $SLURM_JOB_ID on $(hostname) at $(date -Is)"
nvidia-smi

# Copy only the hashed release payload, excluding any existing .so/build/env files.
python3 - "$koor_release" "$koor_clean_root/package" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys
source, target = map(Path, sys.argv[1:])
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest = source / "SHA256SUMS.json"
original = json.loads((source / "release_verification.json").read_text())
if original["package_manifest_sha256"] != digest(manifest):
    raise RuntimeError("Package changed after its first GPU verification")
for name, expected in json.loads(manifest.read_text()).items():
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid manifest path")
    file = source / relative
    if digest(file) != expected:
        raise RuntimeError(f"Changed release payload: {name}")
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file, destination)
shutil.copy2(manifest, target / "SHA256SUMS.json")
print("PASS: copied only the hash-verified source package")
PY

python3 -m venv "$koor_clean_root/venv"
koor_python="$koor_clean_root/venv/bin/python"
"$koor_python" -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
"$koor_python" -m pip install \
  -r "$koor_clean_root/package/requirements.txt" \
  -r "$koor_clean_root/package/requirements-build.txt"
cd "$koor_clean_root/package"
koor_cuda_arch=$("$koor_python" -c 'import torch; major, minor = torch.cuda.get_device_capability(0); print(f"{major}.{minor}")')
echo "Building the bundled extension for allocated CUDA architecture $koor_cuda_arch"
"$koor_python" install_cuda_extension.py --arch "$koor_cuda_arch" --jobs 4
srun "$koor_python" -u verify_release.py --device cuda:0

"$koor_python" - "$koor_release" "$koor_cuda_arch" <<'PY'
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
source = Path.cwd()
release = Path(sys.argv[1])
report = json.loads((source / "release_verification.json").read_text())
manifest_hash = hashlib.sha256((release / "SHA256SUMS.json").read_bytes()).hexdigest()
if report["package_manifest_sha256"] != manifest_hash:
    raise RuntimeError("Source package changed during clean verification")
for name in ("file_hashes", "strict_weight_loading", "cuda_inference"):
    if report[name] != "passed":
        raise RuntimeError(f"Clean verification failed: {name}")
report.update(environment_isolated=True, extension_built_from_package=True,
              cuda_arch=sys.argv[2], verification_kind="clean-environment-package-smoke-test",
              runtime_packages={name: importlib.metadata.version(name) for name in
                                ("torch", "numpy", "scipy", "scikit-learn", "easydict", "safetensors", "pc_util")})
target = release / "release_verification_clean.json"
temporary = target.with_suffix(".tmp")
temporary.write_text(json.dumps(report, indent=2) + "\n")
temporary.replace(target)
print(json.dumps(report, indent=2))
print("PASS: clean-environment v2 package verification completed")
print("Report:", target)
PY

echo "Finished at $(date -Is). Temporary environment is on job-local storage."
