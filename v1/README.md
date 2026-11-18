# Koor v1

Synthetic-trained Point2Roof roof-wireframe reconstruction. The best full-model checkpoint was selected at epoch 29 of a 30-epoch run, following a six-epoch corner stage. [Exact settings](training_metadata.json) and [evaluation reports](evaluation/).

## Environment

Run these commands from this directory (`cd v1` from the repository root), using an allocated GPU for GPU commands. CUDA 13.2 `nvcc` and a compatible C++ compiler must be available for the extension build.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r requirements-build.txt -r requirements.txt
python install_cuda_extension.py --arch 10.0 --jobs 4
```

Architecture 10.0 targets B200. This is the tested hardware configuration, not a claim of compatibility with every CUDA GPU. The source retains upstream tensor-constructor warnings; these did not prevent the reported training or clean-environment inference test.

## Run published weights

```bash
python -m pip install huggingface_hub
hf download Uruva/Koor-building-reconstruction-v1 model.safetensors --local-dir .
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python inference.py --input examples/demo_roof.xyz --output demo_prediction.json
```

The procedural demo is an I/O fixture, not an accuracy benchmark. Inference writes JSON and an OBJ containing vertices and lines. Input is one roof cloud; the CLI reads the first three XYZ columns. Coordinates and units are preserved through normalization and inverse transformation. Colors, NIR, and intensity are ignored; edge scores are uncalibrated. This is custom PyTorch inference, not a Transformers `AutoModel`.

## Prepare the training data

Obtain the synthetic dataset from the [Point2Roof authors](https://github.com/Li-Li-Whu/Point2Roof). After extracting the original archives, the preparation script expects:

```text
data/RoofReconstructionDataset/SyntheticDataset/
  train.txt
  test.txt
  simulateRoof/simulateRoof/<sample-id>/points.xyz
  simulateRoof/simulateRoof/<sample-id>/polygon.obj
  simulateRoofTest/simulateRoofTest/<sample-id>/points.xyz
  simulateRoofTest/simulateRoofTest/<sample-id>/polygon.obj
```

```bash
python scripts/prepare_dataset.py
```

This regenerates local absolute-path lists under `data/prepared/`; rerun it when moving data to a different machine. The supplied lists contain the original author's paths and are not directly usable. The reported split has 14,400 training, 1,599 validation, and 1,600 test samples. Matching normalized vertex layouts are isolated; one source training sample matching test geometry was excluded. This grouping does not detect every near-duplicate or rotated copy. [Recorded split identifiers](provenance/synthetic_split_ids.json).

## Train

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python scripts/train_corners.py --epochs 6 --batch-size 32 --workers 4 \
  --run-dir runs/koor-corners-v1

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python scripts/train_roofs.py --epochs 30 --batch-size 16 --workers 4 \
  --run-dir runs/koor-roofs-v1
```

The full trainer initializes its corner network from `runs/koor-corners-v1/best.pt`. It freezes corners for its first epoch, then jointly trains all stages. Validation selects `best.pt`; `last.pt` contains optimizer/scheduler/RNG state for resuming. Add `--resume` to the same training command and retain the original data, batch size, seed and hyperparameters. Do not resume smoke-test runs as full experiments.

The first-stage trainer currently requires all original CLI arguments, including `--epochs`, to match when resuming. Full-roof training permits increasing the requested total epoch count. These scripts are the archived implementations used in the reported experiment.

## Evaluate a training checkpoint

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python scripts/evaluate_roofs.py \
  --checkpoint runs/koor-roofs-v1/best.pt \
  --test-list data/prepared/test.txt --expected-samples 1600 \
  --output evaluations/koor-v1-synthetic-test
```

This evaluator uses the trusted, resumable training checkpoint format, not the inference-only Safetensors file. Targets are loaded only after point-cloud inference. Keep the checkpoint and thresholds fixed before scoring the test split. No public v1 training checkpoint with optimizer state is bundled here; the model Hub hosts inference-only weights.

Reports contain micro/macro F1, precision, recall, assignment error and empty-output counts at normalized tolerances 0.01, 0.02 and 0.05. Per-building predictions are generated locally and are excluded from GitHub. The position error ignores unmatched corners and must be interpreted alongside recall.

## Release export

`scripts/export_hf_release.py` bundles an inference release from the trusted full checkpoint, matching evaluation reports, local dataset metadata, and source. It exports Safetensors, verifies an exact tensor round trip, and produces a separate package verification script. Run it from a reconstructed training project with its generated data manifests and both evaluations present.

This GitHub tree has a source-oriented layout, not the release package's exact hash manifest. The [recorded verification report](provenance/release_verification.json) applies to the published Hub package. To repeat that package's hash/inference verification, download the complete Hub release into a separate directory and follow its README.
