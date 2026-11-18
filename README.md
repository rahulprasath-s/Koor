# Koor — 3D Roof-Wireframe Reconstruction

Training, evaluation, data preparation, and inference code for reconstructing roof corners and edges from individual building point clouds, using the [Point2Roof architecture](https://github.com/Li-Li-Whu/Point2Roof).

[Model weights and inference release on Hugging Face](https://huggingface.co/Uruva/Koor-building-reconstruction-v1)

Koor currently produces a **roof wireframe**: 3D vertices and undirected edges. It does not recover an address, floor count, facades, textures, or a watertight building mesh. V1 is a synthetic-trained research baseline with weak transfer to real LiDAR.

## Versions

| Version | Data | Status |
|---|---|---|
| [V1](v1/README.md) | Point2Roof synthetic roofs | Trained; held-out synthetic and zero-shot real-data evaluations completed; weights published |
| [V2](v2/README.md) | Building3D Entry-level real LiDAR | Dataset prepared on the university server; preparation source transfer pending; no fine-tuning results yet |

The version directories preserve the server's original project layout. `Point2Roof/` contains the patched upstream model and CUDA extension; `scripts/` contains the Koor workflows. Each version uses its own local data, environment, and output directories.

## V1 results

| Evaluation | Buildings | Corner tolerance | Corner micro-F1 | Edge micro-F1 |
|---|---:|---:|---:|---:|
| Synthetic held-out test | 1,600 | 1% | 0.5874 | 0.4079 |
| Synthetic held-out test | 1,600 | 2% | 0.8891 | 0.8372 |
| Synthetic held-out test | 1,600 | 5% | 0.9549 | 0.9445 |
| Building3D custom spatial holdout, zero-shot | 561 | 1% | 0.0032 | 0.0000 |
| Building3D custom spatial holdout, zero-shot | 561 | 2% | 0.0212 | 0.0028 |
| Building3D custom spatial holdout, zero-shot | 561 | 5% | 0.1770 | 0.0482 |

Tolerance is a fraction of each input cloud's maximum XYZ bounding-box span. Report strict tolerances alongside 5%; a generous tolerance can conceal localization errors. Matching uses minimum-total-distance Hungarian assignment followed by a distance cutoff. An edge is correct only if both endpoints match a target edge. Edge threshold: 0.5. The reported F1 is aggregated over buildings; macro-F1 and other details are in [the evaluation reports](v1/evaluation/).

V1 produced no empty corner or edge predictions on the synthetic test set. On Building3D, 37/561 buildings produced no accepted edges. These are Koor-protocol scores, not exact reproductions of published benchmark protocols. Building3D was evaluation-only for V1.

## Start with V1

See [V1 setup, training, evaluation, and inference](v1/README.md). Training and clean-release inference were verified on NVIDIA B200 with Python 3.12, PyTorch 2.14.1+cu132, and a CUDA 13.2 compiler. Other hardware is unverified. The custom CUDA extension must be compiled in the environment that will run the model.

Raw datasets, trained weights, private credentials, environments, and generated build files are excluded from GitHub. Download weights from Hugging Face. Obtain datasets separately from their original providers.

## Moving code from the GPU server

The [source exporter](tools/export_gpu_sources.py) reads the two server project directories and creates a ZIP with selected source/configuration files and small summary reports. It does not connect to a remote service or change the training projects. See [the transfer workflow](docs/server-transfer.md).

## Attribution and licensing

Koor builds on Point2Roof. Its original MIT notice is preserved at [v1/Point2Roof/LICENSE](v1/Point2Roof/LICENSE). See [THIRD_PARTY.md](THIRD_PARTY.md) and [LICENSING.md](LICENSING.md) for scope, data references, and unresolved Koor licensing status. Do not infer a training dataset's license from the software repository's license.

The V1 source in this initial commit was recovered from the author's published Hugging Face release, with source hashes checked against its manifest. [Provenance](v1/provenance/source.json).
