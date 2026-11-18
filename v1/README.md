# Run and train Koor v1

If you want to try the released model, start with the environment and demo below. If you want to repeat the training, the data layout and training commands are further down.

This first model was trained on synthetic roofs using Point2Roof. I trained the corner detector for six epochs, then the full network for 30 epochs. Validation selected the full-model checkpoint from epoch 29. You can check the [saved settings](training_metadata.json) and [test results](evaluation/).

## Environment

Run these commands from the repository's `v1/` directory. You'll need CUDA 13.2 `nvcc` and a compatible C++ compiler to build the custom operations.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r requirements-build.txt -r requirements.txt
python install_cuda_extension.py --arch 10.0 --jobs 4
```

The `10.0` setting targets B200, the GPU I used for training and testing. Other GPU setups haven't been verified. You may see warnings about the upstream tensor constructors; the reported runs completed with those warnings present.

## Make a first prediction

Download the weights, then try the included demo. Set `CUDA_VISIBLE_DEVICES` to a GPU allocated to you; the example uses GPU 0.

```bash
python -m pip install huggingface_hub
hf download Uruva/Koor-building-reconstruction-v1 model.safetensors --local-dir .
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python inference.py --input examples/demo_roof.xyz --output demo_prediction.json
```

You should get `demo_prediction.json` and `demo_prediction.obj`. The OBJ contains corners and connecting lines. The demo cloud is a simple mathematical roof used to check that the code runs; it isn't one of the scored test buildings.

For your own roof, replace `examples/demo_roof.xyz` with its point-cloud file and choose a new output filename. The script reads the first three XYZ columns. It returns predictions in the input's coordinate system and units. Colors, NIR, and intensity aren't used. Edge scores haven't been calibrated as probabilities. The model loads through the included PyTorch code.

## Prepare the training data

Get the synthetic dataset from the [Point2Roof authors](https://github.com/Li-Li-Whu/Point2Roof) and extract the archives into this layout:

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

The downloaded lists point to the original author's machine, so the preparation step replaces those paths with paths on yours. It writes the manifest and split lists under `data/prepared/`. Run it again if you move the data to another machine.

The experiment used 14,400 training, 1,599 validation, and 1,600 test samples. Roofs with matching normalized vertex layouts were grouped together to reduce leakage. One source training sample matched test geometry and was left out. This check doesn't catch every near-duplicate or rotated copy; the [split identifiers](provenance/synthetic_split_ids.json) are included for reference.

## Train the two stages

First train the corner detector. Once that finishes, train the full network from its best checkpoint.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python scripts/train_corners.py --epochs 6 --batch-size 32 --workers 4 \
  --run-dir runs/koor-corners-v1

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python scripts/train_roofs.py --epochs 30 --batch-size 16 --workers 4 \
  --run-dir runs/koor-roofs-v1
```

The full trainer loads `runs/koor-corners-v1/best.pt`. It keeps that detector fixed for the first epoch while the new heads learn, then updates all three stages together. Each run writes `best.pt`, `last.pt`, and `history.json` into its run directory. Keep the test split out of checkpoint selection.

If a run is interrupted, repeat its command with `--resume`. Keep the data, batch size, seed, and hyperparameters the same. The corner trainer also requires the original `--epochs` value; the full trainer allows you to increase the total epoch count. Use a separate run directory for short checks or new experiments, so their checkpoints don't get mixed together.

## Evaluate a training checkpoint

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python scripts/evaluate_roofs.py \
  --checkpoint runs/koor-roofs-v1/best.pt \
  --test-list data/prepared/test.txt --expected-samples 1600 \
  --output evaluations/koor-v1-synthetic-test
```

This command evaluates the full checkpoint saved by training. The Safetensors download is for inference and doesn't include the training checkpoint's metadata or optimizer state. Only load trusted training checkpoints.

Choose the checkpoint and thresholds using validation, then leave them fixed for the test run. The evaluator predicts from the point cloud first and reads the target geometry afterward.

The output folder contains the overall metrics and per-building predictions. Reports include precision, recall, micro/macro F1, assignment error, and empty-output counts at tolerances of 1%, 2%, and 5% of the building span. The position error ignores unmatched corners, so read it alongside recall. Prediction dumps stay local.

## Package the trained model

`scripts/export_hf_release.py` collects the best full checkpoint, matching evaluation reports, data-preparation metadata, and source into an inference package. It exports Safetensors, checks that every exported tensor reloads unchanged, and creates a script for testing the package on a GPU. It needs the original training outputs and both evaluations to be present.

I also tested the published package in a clean environment on B200. The [verification report](provenance/release_verification.json) records that check. To repeat it, download the complete Hugging Face release into a separate directory and follow its README. Its file hashes describe the release package; this GitHub repository has a different layout for working on the source.
