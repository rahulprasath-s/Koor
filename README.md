# Koor

I'm building Koor to understand how a neural network can turn a building's roof point cloud into a 3D structure.

For this first release, I worked through the whole process: preparing the data, getting the CUDA code running on my university's B200 GPUs, training the model, testing it, and putting the weights on Hugging Face. This repository holds the code behind that work.

Koor uses the [Point2Roof architecture](https://github.com/Li-Li-Whu/Point2Roof). I've included the data preparation, training, evaluation, and inference scripts, along with the compatibility changes needed for the environment I used.

**[Download the model on Hugging Face](https://huggingface.co/Uruva/Koor-building-reconstruction-v1)** · **[Setup and training guide](v1/README.md)**

## What it does

Give the model a point cloud for one building roof. It predicts corners, refines their positions, and decides which corners should be connected. The result is a **roof wireframe**, saved as JSON and OBJ.

```text
Roof points → corner prediction → refinement → edge prediction → 3D wireframe
```

Walls, textures, addresses, and floor counts are outside what this model currently learns. The output can contain missing or extra connections, so it still needs inspection.

## What I learned from testing

The model performed well on unseen synthetic roofs. Testing it on real Building3D LiDAR was a different story: the scores fell sharply. I'm including both results because the synthetic score alone would give an incomplete picture of what the model can do.

| Test set | Buildings | Corner tolerance | Corner F1 | Edge F1 |
|---|---:|---:|---:|---:|
| Synthetic held-out test | 1,600 | 1% | 0.5874 | 0.4079 |
| Synthetic held-out test | 1,600 | 2% | 0.8891 | 0.8372 |
| Synthetic held-out test | 1,600 | 5% | 0.9549 | 0.9445 |
| Real Building3D holdout, without fine-tuning | 561 | 1% | 0.0032 | 0.0000 |
| Real Building3D holdout, without fine-tuning | 561 | 2% | 0.0212 | 0.0028 |
| Real Building3D holdout, without fine-tuning | 561 | 5% | 0.1770 | 0.0482 |

F1 balances missed predictions and extra predictions; higher is better. Here, a corner must fall within the stated fraction of the building's maximum bounding-box span to count as a match. An edge needs both endpoints to match a true roof connection. The table reports micro-F1, pooled across buildings.

The stricter scores matter. A 5% tolerance can hide positioning errors, and a high F1 doesn't mean the whole building was reconstructed correctly. On the real holdout, 37 of 561 buildings had no accepted edges.

The model was trained on synthetic data only. These tests use Koor's evaluation procedure and a custom Building3D spatial holdout, so they aren't directly comparable to published benchmark tables. I've written up [how the evaluation works](docs/results.md); the [full reports](v1/evaluation/) include precision, recall, macro-F1, and failure counts.

## Try it or train it yourself

Start with the [v1 guide](v1/README.md). It walks through the environment, a demo prediction, data preparation, training, and evaluation.

I trained and checked the release on NVIDIA B200 with Python 3.12, PyTorch 2.14.1, and CUDA 13.2. The custom CUDA extension needs to be built before inference. I haven't verified other GPU setups yet.

Weights are on Hugging Face. Datasets come from their original providers; neither is stored in this Git repository. The small demo cloud is generated mathematically and is included so you can check that inference runs.

## Finding your way around

```text
v1/scripts/          Data preparation, training, evaluation, and release export
v1/Point2Roof/      Patched model code and CUDA operations
v1/evaluation/      Saved results from the two test sets
v1/inference.py     Run a point cloud through the published model
tools/              Export source code from the GPU server
```

If you're working from a remote training machine, the [transfer guide](docs/server-transfer.md) explains how to bring the source across without copying datasets or environments.

## Credit where it's due

This project builds on the work of the Point2Roof authors and uses Building3D for the real-data evaluation. Their work made this experiment possible. References and the preserved upstream software license are in [THIRD_PARTY.md](THIRD_PARTY.md).

I haven't assigned a license to the original Koor code or weights yet. The training dataset's applicable terms still need to be confirmed; the upstream software's MIT license is recorded separately in [LICENSING.md](LICENSING.md).
