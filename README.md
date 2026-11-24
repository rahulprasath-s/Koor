# Koor

I'm building Koor to understand how a neural network can turn a roof point cloud into a 3D structure I can inspect and work with.

I started with synthetic roofs on my university's B200 GPUs. V1 did well on those examples, but testing it on real Building3D LiDAR exposed a large gap. For v2, I trained on both datasets on an A40 at NHR@FAU, keeping a frozen v1 model as a corner-prediction teacher.

V2 improved the real-data scores. It also gave up some synthetic accuracy, particularly when corners had to be placed precisely. This repository includes the code and both results, because that tradeoff is part of the experiment.

Koor uses the [Point2Roof architecture](https://github.com/Li-Li-Whu/Point2Roof), with compatibility fixes and preparation, training, evaluation and inference code around it.

**[V2 model on Hugging Face](https://huggingface.co/Uruva/Koor-building-reconstruction-v2)** · **[V2 training and results](v2/README.md)** · **[V1 model](https://huggingface.co/Uruva/Koor-building-reconstruction-v1)** · **[V1 guide](v1/README.md)**

## What it does

Give Koor a point cloud for one building roof. It predicts corners, refines their positions and decides which corners should be connected. The output is a **3D roof wireframe**.

```text
Roof points → corner prediction → refinement → edge prediction → wireframe
```

Predictions can have missing corners, misplaced points or disconnected edges. Koor doesn't yet take an address or image, recover floor counts, or produce walls and textures. An interactive viewer can display its wireframe, but a full building model still needs more geometry and data.

## What changed in v2

Both models were evaluated together on Alex for this comparison. These are micro-F1 scores at a tolerance of 5% of each input cloud's largest XYZ bounding-box span:

| Held-out dataset | Buildings | Vertex F1, v1 → v2 | Edge F1, v1 → v2 |
|---|---:|---:|---:|
| Synthetic test | 1,600 | 0.9547 → 0.9451 | 0.9441 → 0.9276 |
| Building3D spatial holdout | 561 | 0.1763 → 0.5275 | 0.0482 → 0.3979 |

The strict scores tell a less forgiving story. At 1% tolerance, synthetic edge F1 fell from 0.4076 to 0.3334. On Building3D, 45 of 561 roofs had no accepted edges in v2, compared with 37 in v1, despite the higher aggregate F1. I would use v2 as a Building3D-adapted experiment, not assume it improves every roof.

V2's reported checkpoint is epoch 14, selected on validation before test evaluation. No epoch met the original requirement to preserve all six synthetic F1 scores while improving the real-data score. All six Building3D held-out F1 scores increased, and all six synthetic scores declined.

The [v2 guide](v2/README.md#held-out-results) has every tolerance and the [full reports](v2/evaluation/) include precision, recall, macro-F1 and failure counts. The [evaluation notes](docs/results.md) explain the scoring and its limits. These are Koor evaluations on a custom Building3D holdout, not published benchmark reproductions. V1's original B200 reports are preserved separately.

## Try it or train it

Start with the [v1 guide](v1/README.md) to try the published weights and procedural demo. Use the [v2 guide](v2/README.md) for mixed training, deterministic replay, checkpoint/resume testing and Slurm jobs on Alex.

V1 was trained and checked on B200 with Python 3.12, PyTorch 2.14.1 and CUDA 13.2. The v2 experiment used A40 with Python 3.12.0, PyTorch 2.11.0 and CUDA 12.8. The custom CUDA extension must be built for the target GPU.

Weights are hosted separately on Hugging Face. Download datasets from their original providers. Environments, raw datasets, training checkpoints, logs and prediction dumps are excluded from this Git repository. The included v1 demo cloud is generated mathematically and is not a benchmark sample.

## Finding your way around

```text
v1/                 Published v1 source, inference and original evaluation
v1/Point2Roof/      Shared model architecture and patched CUDA operations
v2/scripts/         Dataset preparation, mixed distillation and paired evaluation
v2/slurm/           Batch jobs that run without an active laptop connection
v2/evaluation/      Controlled v1/v2 comparisons on Alex
v2/provenance/      Split IDs, source hashes, runtime and original Alex files
docs/               Evaluation and source-transfer notes
tools/              Source-transfer utilities
```

The server directory name can be `Koor-building-reconstruction-v2`; the Git repository contains both versioned source trees. Shared model code stays in `v1/Point2Roof`. The [v2 guide](v2/README.md#code-and-experiment-records) explains how the cleaned layout relates to the original Alex checkpoint paths.

## Credit where it's due

Koor builds on Point2Roof and uses Building3D for real-data training and evaluation in v2. Their research and data made this experiment possible. References and the preserved upstream software notice are in [THIRD_PARTY.md](THIRD_PARTY.md).

I haven't assigned a license to the original Koor code or weights. Building3D lists CC BY-NC-SA 4.0; applicable Point2Roof dataset terms still need to be confirmed. The upstream software's MIT license is recorded separately in [LICENSING.md](LICENSING.md).
