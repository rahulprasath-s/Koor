# Koor v2

V1 worked well on synthetic roofs, but real LiDAR exposed how much was missing. For v2, I trained on both synthetic roofs and Building3D, with a frozen copy of v1 helping preserve corner predictions on synthetic samples.

The result is more useful on the Building3D holdout. It also loses some synthetic accuracy. I'm keeping both sides of that result here: this is a Building3D-adapted experiment, rather than a replacement that improves every v1 metric.

Input is one roof's XYZ point cloud; output is a wireframe of 3D corners and edges. The model doesn't take an address or image, predict floors, or produce a complete textured building. V2 weights are being prepared for Hugging Face and are not included here.

## Held-out results

Both models were rerun on Alex using identical samples, preprocessing, seed, batch size and scoring. V1's original B200 reports remain in [v1/evaluation](../v1/evaluation/); small numerical differences are present in the Alex rerun.

| Dataset | Tolerance | Vertex F1, v1 | Vertex F1, v2 | Edge F1, v1 | Edge F1, v2 |
|---|---:|---:|---:|---:|---:|
| Synthetic test, 1,600 roofs | 1% | 0.5871 | 0.5279 | 0.4076 | 0.3334 |
| Synthetic test, 1,600 roofs | 2% | 0.8891 | 0.8650 | 0.8372 | 0.7985 |
| Synthetic test, 1,600 roofs | 5% | 0.9547 | 0.9451 | 0.9441 | 0.9276 |
| Building3D spatial holdout, 561 roofs | 1% | 0.0032 | 0.0711 | 0.0000 | 0.0133 |
| Building3D spatial holdout, 561 roofs | 2% | 0.0208 | 0.2650 | 0.0026 | 0.1457 |
| Building3D spatial holdout, 561 roofs | 5% | 0.1763 | 0.5275 | 0.0482 | 0.3979 |

All six Building3D F1 scores improved; all six synthetic scores declined. Building3D roofs with no accepted edges increased from **37 to 45**. Neither model had empty corners or edges on the synthetic test.

Scores are micro-F1 pooled across buildings. Tolerances are fractions of the full input cloud's largest XYZ bounding-box span, not fixed distances in metres. Corners use minimum-total-distance Hungarian assignment followed by the cutoff. An edge needs both endpoints to match an actual labelled edge; acceptance threshold is 0.5. These are Koor evaluations on a custom spatial holdout, not published Building3D benchmark reproductions.

The selected checkpoint is **epoch 14**, chosen on joint validation before test evaluation. All 15 epochs finished, but none passed the original all-metric synthetic retention gate; `best_retained.pt` was not created. Reported scores come from `best_joint.pt`, not the final checkpoint. See [reports](evaluation/), [history](training/history.json) and [selection](evaluation/selection.json).

## Code and experiment records

```text
v2/scripts/       Maintained preparation, training and evaluation code
v2/slurm/         Alex batch jobs, submitted from the repository root
v2/evaluation/    Aggregate reports and paired comparisons
v2/training/      Validation baseline and epoch history
v2/provenance/    Environment, split IDs, hashes and original Alex source
v1/Point2Roof/    Shared architecture and CUDA operations, unchanged
v1/scripts/      Shared dataset/loss/evaluation helpers, unchanged
```

`provenance/alex_source/` is an exact snapshot of the flat source layout used on Alex. It is an archive, not a second installation to run in place. Maintained scripts resolve the repository root from `v2/scripts`; Slurm jobs use their submission directory. The [source record](provenance/source.json) explains the layout changes.

The trainer checks source hashes and data identities on resume. **Keep the original Alex workspace to resume its archived checkpoints.** The maintained layout changes those hashes and is for new runs; it does not change the recorded experiment or scores. Recorded paths in reports belong to the original workspace.

## Environment

Run from the repository root. On Alex:

```bash
module load python/3.12-base cuda/12.8.1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r v2/requirements.txt -r v1/requirements-build.txt
python -m pip install -r v2/requirements-download.txt
TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=4 python -m pip install --no-build-isolation --no-deps ./v1/Point2Roof/pc_util
hf download Uruva/Koor-building-reconstruction-v1 --local-dir models/koor-v1
```

The experiment used Python 3.12.0, PyTorch 2.11.0+cu128 and A40. `8.6` targets A40; choose the matching architecture on other GPUs. Captured numerical/build versions are in [environment.json](provenance/environment.json). Downloader versions were not captured. Site modules apply to Alex; adapt them elsewhere.

## Data preparation

Download and extract datasets under `data/`; they are excluded from Git. Building3D requires provider access and lists CC BY-NC-SA 4.0. See [licensing](../LICENSING.md), [credits](../THIRD_PARTY.md) and original provider terms. The [v1 guide](../v1/README.md) links the synthetic data.

Expected extracted paths:

```text
data/building3d/raw/Entry-level/train/xyz/*.xyz
data/building3d/raw/Entry-level/train/wireframe/*.obj
data/building3d/raw/Entry-level/test/xyz/*.xyz
data/RoofReconstructionDataset/RoofReconstructionDataset/SyntheticDataset/train.txt
data/RoofReconstructionDataset/RoofReconstructionDataset/SyntheticDataset/test.txt
data/RoofReconstructionDataset/RoofReconstructionDataset/SyntheticDataset/simulateRoof/**/points.xyz
data/RoofReconstructionDataset/RoofReconstructionDataset/SyntheticDataset/simulateRoofTest/**/points.xyz
```

Synthetic sample directories also need `polygon.obj`. Override `--raw` or `--source` if your layout differs:

```bash
python v2/scripts/prepare_building3d.py
python v2/scripts/prepare_synthetic.py
```

Building3D conversion keeps XYZ, subtracts a shared float64 origin and preserves edges. Y bands separate complete building bounds; a 50-source-unit buffer is excluded on each side of a boundary. Matching normalized vertex fingerprints spanning splits are excluded. Source coordinate units/CRS are not inferred. Fingerprinting does not identify every rotated or near-duplicate shape.

| Dataset | Train | Validation | Held out | Excluded |
|---|---:|---:|---:|---:|
| Synthetic | 14,400 | 1,599 | 1,600 test | 1 training roof matching test geometry |
| Building3D | 4,538 | 533 | 561 spatial holdout | 66 in boundary buffers |

The 583 official unlabelled Building3D test clouds were not scored. Synthetic preparation verifies v1 split membership. [The split record](provenance/split_ids.json) contains actual experiment IDs and path-list hashes. Check new preparation memberships against it before claiming the same split. Prepared paths are absolute; moving data requires regenerating them and changes the strict resume fingerprint.

## Smoke test and training

Submit from the **repository root**, and check each job's log:

```bash
sbatch v2/slurm/verify_v1.sbatch
sbatch v2/slurm/smoke_mixed.sbatch
```

The smoke job performs two training batches in total: save after one batch, restart the process, resume at the second batch, and verify sample counters and the frozen teacher. Limited validation scores are not benchmark results.

After it passes, submit a fresh full run:

```bash
sbatch v2/slurm/train_mixed.sbatch
```

Maintained jobs write to `runs/koor-v2-mixed-distill`, avoiding the archived Alex run. Each batch has eight synthetic and eight real roofs, with four loader workers and FP32 training. All stages receive supervised losses. Synthetic-only point scores and offsets are distilled from the frozen v1 corner detector in eval mode. There is no direct refinement/edge distillation. Each full epoch processes 14,400 samples per domain, cycling the smaller real set.

The run has 15 epochs, Adam with corner/head learning rates 1e-5/1e-4 and weight decay 1e-3, and StepLR halving rates every five completed epochs. Both distillation weights are 1, temperature is 2. Score KD uses class-balanced Bernoulli KL; offsets use confidence-squared weights and SmoothL1. Inputs contain 1,024 normalized points. Per-draw deterministic augmentation allows mid-epoch resume.

The default allocation is four hours. Slurm requests a safe pause around two minutes before timeout. The trainer also checkpoints every 100 batches and around validation. Long GPU operations or filesystem delays may exceed the warning margin; the last complete atomic checkpoint remains. Resubmit the same batch script to resume; automatic requeue is not enabled. Laptop disconnection does not affect a submitted job.

`last.pt` holds optimizer, scheduler, RNG and sampler progress. `best_joint.pt` uses the harmonic mean of both domains' validation corner/edge harmonic F1 at 5%. `best_retained.pt` requires all six synthetic F1s to meet the threshold, all six real F1s not to regress, and real 5% harmonic F1 to improve. A safely paused job may have Slurm state `COMPLETED`; check the trainer's `PAUSED SAFELY` or `COMPLETED` message.

## Evaluation

```bash
sbatch v2/slurm/evaluate_heldout.sbatch
```

This freezes the selected checkpoint and scores both models on both held-out lists. Per-building predictions and aggregate results go to an ignored `evaluations/` directory. The committed `v2/evaluation/` reports belong to the archived Alex experiment. Avoid tuning thresholds or selecting checkpoints from test results.

## Preparing the recorded v2 weights for Hugging Face

The [release exporter](scripts/export_hf_release.py) defaults to the original Alex run and paired evaluation directory. Run it in that workspace, where the original checkpoint and hash-matched source still exist. It can be saved as `scripts/export_hf_v2.py` in the original flat workspace, or run under `v2/scripts` in a checkout containing the recorded files:

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/export_hf_v2.py
```

Use `--checkpoint`, `--evaluation`, or `--output` to override locations. It verifies the selected checkpoint against the held-out reports, checks exact safetensors weight conversion, bundles standalone inference and CUDA source, and writes a model card with both gains and regressions. No optimizer state or raw benchmark geometry is bundled, and nothing is uploaded. It refuses to overwrite a release directory.

The next step is a GPU package verification job running the generated `verify_release.py`. That writes `release_verification.json` with a hash of the package manifest. Verify the package in a clean environment before uploading it. Model-weight licensing metadata is not assigned by this exporter; see the generated licensing record and original data terms.

## What remains

Strict synthetic localization and missing Building3D edges remain weaknesses. A wireframe viewer is only the first part of a building reconstruction tool. Roof faces, walls with valid ground elevation, image-conditioned geometry and map-based data retrieval still need work.

This builds on [Point2Roof](https://github.com/Li-Li-Whu/Point2Roof) and [Building3D](https://szusic.github.io/Building3D/). The shared upstream software notice remains in [v1/Point2Roof/LICENSE](../v1/Point2Roof/LICENSE); see [credits](../THIRD_PARTY.md) and [licensing status](../LICENSING.md).
