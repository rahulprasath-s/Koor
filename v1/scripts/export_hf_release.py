"""Export an offline, reviewable Koor v1 Hugging Face release folder.

Save this file as scripts/export_hf_release.py in the v1 project. This script
does not upload, change training checkpoints, or assign a model-weights license.
The input checkpoint must be your own trusted training checkpoint.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PROJECT = Path(__file__).resolve().parents[1]
NORMALIZATION = "full-cloud bbox center; max XYZ span; +0.5"


INFERENCE_TEMPLATE = r'''"""Koor point-cloud-to-roof-wireframe inference; CUDA extension required."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from easydict import EasyDict
from safetensors.torch import load_file


__PREDICT_GRAPHS__


def load_model(model_dir=None, device="cuda:0"):
    model_dir = Path(model_dir or Path(__file__).resolve().parent).resolve()
    device = torch.device("cuda:0" if device == "cuda" else device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Koor requires CUDA and the compiled pc_util extension")
    torch.cuda.set_device(device)
    sys.path.insert(0, str(model_dir / "Point2Roof"))
    from model.roofnet import RoofNet

    config = json.loads((model_dir / "config.json").read_text())
    preprocessing = json.loads((model_dir / "preprocessing.json").read_text())
    if preprocessing["normalization"] != "full-cloud bbox center; max XYZ span; +0.5":
        raise ValueError("Unsupported normalization")
    architecture = EasyDict(copy.deepcopy(config["model_config"]))
    model = RoofNet(copy.deepcopy(architecture.MODEL))
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    model.to(device).eval()
    return model, preprocessing, config


def predict(model, preprocessing, points, seed=42, edge_threshold=0.5):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1:
        raise ValueError("Expected a nonempty N-by-3 XYZ array")
    if not np.isfinite(points).all() or not 0 <= edge_threshold <= 1:
        raise ValueError("Invalid points or edge threshold")
    lower, upper = points.min(0), points.max(0)
    center = (lower + upper) / 2
    scale = float((upper - lower).max())
    if scale <= 0:
        raise ValueError("Point cloud has zero spatial extent")
    normalized = (points - center) / scale + 0.5
    npoints = int(preprocessing["points"])
    rng = np.random.default_rng(seed)
    if len(points) >= npoints:
        selected = rng.choice(len(points), npoints, replace=False)
    else:
        selected = np.concatenate([
            np.arange(len(points)),
            rng.choice(len(points), npoints - len(points), replace=True),
        ])
        rng.shuffle(selected)
    device = next(model.parameters()).device
    with torch.cuda.device(device), torch.inference_mode():
        tensor = torch.from_numpy(normalized[selected].astype(np.float32))
        result = predict_graphs(model, tensor.unsqueeze(0).to(device))[0]
    accepted = result["scores"] >= edge_threshold
    vertices = (result["vertices"] - 0.5) * scale + center
    edges = result["pairs"][accepted]
    return {
        "vertices": vertices.tolist(), "edges": edges.tolist(),
        "edge_scores": result["scores"][accepted].tolist(),
        "candidate_edges": result["pairs"].tolist(),
        "candidate_edge_scores": result["scores"].tolist(),
        "status": "no_corner_proposals" if len(vertices) == 0
                  else "no_edges" if len(edges) == 0 else "ok",
        "metadata": {
            "output_type": "roof_wireframe", "coordinate_system": "same as input",
            "units": "same as input; not inferred", "source_center": center.tolist(),
            "source_scale": scale, "input_point_count": len(points),
            "sampled_point_count": npoints, "sampling_seed": seed,
            "edge_threshold": edge_threshold, "scores_calibrated": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    args = parser.parse_args()
    obj_path = args.output.with_suffix(".obj")
    if args.output.suffix.lower() != ".json":
        parser.error("--output must end in .json")
    if args.output.exists() or obj_path.exists():
        raise FileExistsError("Choose output names that do not already exist")
    model, preprocessing, _ = load_model(args.model_dir, args.device)
    points = np.loadtxt(args.input, dtype=np.float64, usecols=(0, 1, 2), ndmin=2)
    result = predict(model, preprocessing, points, args.seed, args.edge_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with obj_path.open("w") as file:
        file.write("# Koor predicted roof wireframe: vertices and undirected lines\n")
        for vertex in result["vertices"]:
            file.write("v " + " ".join(f"{value:.12f}" for value in vertex) + "\n")
        for a, b in result["edges"]:
            file.write(f"l {a + 1} {b + 1}\n")
    print("Status:", result["status"])
    print("Saved:", args.output, "and", obj_path)


if __name__ == "__main__":
    main()
'''


INSTALL_TEMPLATE = r'''"""Compile the bundled extension in the current Python environment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="10.0", help="B200 is 10.0")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    root = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment["TORCH_CUDA_ARCH_LIST"] = args.arch
    environment["MAX_JOBS"] = str(args.jobs)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--no-build-isolation",
        "--no-deps", "-v", str(root / "Point2Roof/pc_util"),
    ], check=True, env=environment)


if __name__ == "__main__":
    main()
'''


VERIFY_TEMPLATE = r'''"""Verify release hashes, strict model loading, and CUDA inference on a demo."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    hashes = json.loads((root / "SHA256SUMS.json").read_text())
    for name, expected in hashes.items():
        if digest(root / name) != expected:
            raise ValueError(f"Hash mismatch: {name}")

    import numpy as np
    import torch
    from inference import load_model, predict

    model, preprocessing, _ = load_model(root, args.device)
    points = np.loadtxt(root / "examples/demo_roof.xyz", ndmin=2)
    result = predict(model, preprocessing, points)
    vertices = np.asarray(result["vertices"], dtype=np.float64).reshape(-1, 3)
    edges = np.asarray(result["edges"], dtype=np.int64).reshape(-1, 2)
    if not np.isfinite(vertices).all():
        raise ValueError("Non-finite output")
    if len(edges) and (edges.min() < 0 or edges.max() >= len(vertices)):
        raise ValueError("Invalid edge indices")
    torch.cuda.synchronize()
    report = {
        "file_hashes": "passed", "strict_weight_loading": "passed",
        "cuda_inference": "passed", "gpu": torch.cuda.get_device_name(model.keypoint_det_net.cls_fc.weight.device),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "demo_status": result["status"], "predicted_vertices": len(vertices),
        "predicted_edges": len(edges),
        "scope": "Packaging and inference smoke test; not an accuracy benchmark",
    }
    (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
'''


def plain(value):
    return json.loads(json.dumps(value))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def extract_predict_function(path):
    source = path.read_text()
    tree = ast.parse(source)
    matches = [node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == "predict_graphs"]
    if len(matches) != 1:
        raise ValueError("Expected one predict_graphs function in the tested evaluator")
    function = ast.get_source_segment(source, matches[0])
    # Refuse to silently bundle a target-conditioned implementation.
    forbidden = {"truth", "truth_vertices", "truth_edges", "read_truth", "matches", "matcher"}
    names = {node.id for node in ast.walk(matches[0]) if isinstance(node, ast.Name)}
    strings = {node.value for node in ast.walk(matches[0])
               if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    if names.intersection(forbidden) or strings.intersection({"edges", "matches"}):
        raise ValueError("Unexpected target-related references in inference function")
    return function


def copy_model_source(source, destination):
    required = [source / "model/roofnet.py", source / "utils/loss_utils.py",
                source / "pc_util/setup.py", source / "pc_util/src", source / "LICENSE"]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    count = 0
    for directory in ("model", "utils", "pc_util"):
        for path in sorted((source / directory).rglob("*")):
            if not path.is_file() or any(part in {"__pycache__", "build", ".git"}
                                         or part.endswith(".egg-info") for part in path.parts):
                continue
            if path.suffix not in {".py", ".cpp", ".cu", ".h", ".cuh", ".hpp"}:
                continue
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            count += 1
    shutil.copy2(source / "LICENSE", destination / "LICENSE")
    if (source / "README.md").exists():
        shutil.copy2(source / "README.md", destination / "UPSTREAM_README.md")
    return count


def load_evaluation(path, expected_samples, checkpoint_hash):
    report = json.loads(path.read_text())
    if report["samples"] != expected_samples:
        raise ValueError(f"Unexpected sample count: {path}")
    if report["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError(f"Evaluation used a different checkpoint: {path}")
    if report["protocol"]["model_mode"] != "eval; target-free inference":
        raise ValueError(f"Unexpected evaluation mode: {path}")
    if report["protocol"]["edge_threshold"] != 0.5:
        raise ValueError(f"Unexpected edge threshold: {path}")
    for tolerance in ("0.01", "0.02", "0.05"):
        for kind in ("vertex", "edge"):
            score = report["metrics_by_tolerance"][tolerance][kind]["f1"]
            if not 0 <= score <= 1:
                raise ValueError(f"Invalid F1 score: {path}")
    # Keep reproducibility hashes and results, without publishing server home paths.
    for key in ("checkpoint", "test_list"):
        if key in report:
            report[key] = Path(report[key]).name
    return report


def package_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(f"Required package is missing in the export environment: {name}")
    return versions


def git_metadata(source):
    def run(*arguments):
        try:
            result = subprocess.run(["git", "-C", str(source), *arguments],
                                    capture_output=True, text=True, timeout=15)
            return result.stdout.strip() if result.returncode == 0 else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
    return run("rev-parse", "HEAD"), run("diff", "--no-ext-diff", "--", "model", "utils", "pc_util")


def model_card(config, synthetic, real, versions, torch_version, cuda_index, split_counts):
    table = []
    for label, report in (("Point2Roof synthetic test", synthetic),
                          ("Building3D spatial holdout, zero-shot", real)):
        for tolerance in ("0.01", "0.02", "0.05"):
            metrics = report["metrics_by_tolerance"][tolerance]
            table.append(f"| {label} | {report['samples']} | {float(tolerance):.0%} | "
                         f"{metrics['vertex']['f1']:.4f} | {metrics['edge']['f1']:.4f} |")
    vertex_empty = real["empty_corner_buildings"]
    edge_empty = real["empty_edge_buildings"]
    return f'''---
library_name: pytorch
tags:
  - point-cloud
  - building-reconstruction
  - roof-wireframe
  - point2roof
  - custom-code
  - safetensors
model_name: Koor-building-reconstruction-v1
---

# Koor v1: roof-wireframe reconstruction

Koor v1 maps a building roof point cloud to predicted 3D corners and undirected
edges. It uses the [Point2Roof architecture](https://github.com/Li-Li-Whu/Point2Roof)
with compatibility fixes and a custom training/evaluation workflow. It was trained
on Point2Roof's synthetic dataset. It is not a new foundation-model architecture.

**V1 is a synthetic-trained research baseline. Transfer to real Building3D LiDAR
is poor.** It does not infer an address, coordinates, floor count, walls, facades,
or textures. Output is a roof wireframe, not a watertight building mesh.

## Evaluation

The checkpoint was selected using validation, then evaluated with frozen
thresholds. Building3D was used for zero-shot evaluation, not v1 training.

| Benchmark | Buildings | Corner tolerance | Corner micro-F1 | Edge micro-F1 |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

Tolerance is a fraction of each complete input cloud's maximum XYZ bounding-box
span. The 5% tolerance is generous and should not be presented without the
stricter results. Corners are matched by minimum-total-distance Hungarian
assignment followed by a tolerance cutoff. An undirected edge is correct only
when both endpoints match a target edge. The edge-score threshold is 0.5.
Precision, recall, macro-F1, assignment error, failure counts, environment and
timing scope are in `evaluation/`. These are **Koor-protocol results**, not an
exact reproduction of published Point2Roof or Building3D benchmark scores.

On the real holdout, {vertex_empty}/{real['samples']} buildings had no corners and
{edge_empty}/{real['samples']} had no accepted edges. Nonempty output does not
imply correct reconstruction. The real holdout is a custom spatial split of
labelled Building3D Entry-level training data; the official unlabelled test set
was not scored. It is reserved for a frozen v1/v2 comparison.

## Input and preprocessing

One N-by-3 XYZ cloud for an individual roof, with consistent linear units.
Read coordinates as float64, subtract the full-cloud bounding-box center, divide
all axes by the same maximum bounding-box span, and add 0.5. Sample 1,024 points
without replacement when possible; otherwise retain original points and fill
with replacement. Inference has no augmentation. Colors, NIR and intensity are
not inputs. Output is transformed back into the input coordinate system and
units. No CRS or units are inferred. Edge scores are uncalibrated.

## Training

Synthetic split counts: `{json.dumps(split_counts, sort_keys=True)}`.
Matching normalized vertex layouts were grouped across splits; one source
training sample matching test geometry was excluded in the reported workflow.
This does not detect every rotated or near-duplicate roof.

A six-epoch corner stage preceded full-roof training. In full training the corner
stage was frozen initially, then jointly updated with refinement and edge heads.
The selected full-roof checkpoint is epoch **{config['checkpoint_epoch']}**.
Selection used the harmonic mean of validation corner and edge micro-F1 at
normalized tolerance 0.05. The exact saved settings and configuration are in
`training_metadata.json`. Neither the optimizer nor resumable training state is
included in `model.safetensors`.

## B200 setup

Training and evaluation ran on NVIDIA B200. The exported package must still pass
`verify_release.py` in the target environment. Other hardware is unverified.
CUDA inference is required; CPU inference is not implemented. Build from source
using a matching CUDA toolkit and supported C++ compiler. The recorded CUDA build
is {cuda_index}; the university setup used a CUDA 13.2 compiler.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch=={torch_version} --index-url https://download.pytorch.org/whl/{cuda_index}
python -m pip install -r requirements-build.txt -r requirements.txt
python install_cuda_extension.py --arch 10.0 --jobs 4
CUDA_VISIBLE_DEVICES=0 python verify_release.py
CUDA_VISIBLE_DEVICES=0 python inference.py --input examples/demo_roof.xyz --output demo_prediction.json
```

Replace GPU 0 with an allocated GPU. `inference.py` also writes an OBJ containing
vertices and lines. An XYZ text file may have extra columns; the CLI reads its
first three. Python API:

```python
from inference import load_model, predict
model, preprocessing, config = load_model(device="cuda:0")
result = predict(model, preprocessing, points_xyz, seed=42)
```

This is custom PyTorch code, not a Transformers `AutoModel` or built-in pipeline.
No Hub account or token is needed for local inference after downloading files.
The procedural demo cloud is an I/O fixture, not an accuracy benchmark. To
reproduce training, reconstruct the original project layout with archived scripts
under `scripts/` and the bundled `Point2Roof/` code, and obtain the dataset from
its authors. No raw benchmark clouds or wireframes are bundled here.

## Limitations and attribution

Real-LiDAR transfer is weak; fine corner positioning is weaker at strict
tolerances. Sparse or complex roofs can have missed/extra corners, incorrect
edges, disconnected graphs or empty predictions. Connectivity and closed roof
surfaces are not guaranteed. Input ground points or facades are not automatically
segmented away. This model is not validated for measurement-critical decisions.

Point2Roof: Li Li, Nan Song, Fei Sun, Xinyi Liu, Ruisheng Wang, Jian Yao and
Shapsheng Cao, *Point2Roof: End-to-end 3D building roof modeling from airborne
LiDAR point clouds*. The upstream software MIT notice is preserved in
`Point2Roof/LICENSE`. Source: https://github.com/Li-Li-Whu/Point2Roof

Building3D: Ruisheng Wang, Shangfeng Huang and Hongxin Yang,
*Building3D: An Urban-Scale Dataset and Benchmarks for Learning Roof Structures
from Point Clouds*, ICCV 2023. Dataset: https://huggingface.co/datasets/Building3D/Building3D
The dataset lists CC BY-NC-SA 4.0. No Building3D examples are redistributed here.

## Licensing status

This is a **release-preparation folder**. A license for Koor weights and original
Koor code has not been assigned by this exporter. Point2Roof's software MIT
license is not evidence of a separate license for its externally hosted training
dataset. Confirm the applicable training-data and model-release terms before
choosing public model-license metadata. See `LICENSING.md`.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--synthetic-report", type=Path)
    parser.add_argument("--real-report", type=Path)
    args = parser.parse_args()
    project = args.project.expanduser().resolve()
    checkpoint_path = (args.checkpoint or project / "runs/koor-roofs-v1/best.pt").resolve()
    output = (args.output or project / "release/Koor-building-reconstruction-v1").resolve()
    synthetic_path = args.synthetic_report or project / "evaluations/koor-v1-synthetic-test/summary.json"
    real_path = args.real_report or project / "evaluations/koor-v1-building3d-zero-shot/summary.json"
    evaluator = project / "scripts/evaluate_roofs.py"
    if output.exists():
        raise FileExistsError(f"Choose a new --output; already exists: {output}")
    for path in (checkpoint_path, synthetic_path, real_path, evaluator,
                 project / "data/prepared/manifest.json",
                 project / "scripts/train_corners.py", project / "scripts/train_roofs.py",
                 project / "scripts/prepare_dataset.py"):
        if not path.is_file():
            raise FileNotFoundError(path)

    import numpy as np
    import torch
    from safetensors.torch import save_file, load_file

    checkpoint_hash = digest(checkpoint_path)
    synthetic = load_evaluation(synthetic_path, 1600, checkpoint_hash)
    real = load_evaluation(real_path, 561, checkpoint_hash)
    if "synthetic" not in synthetic["benchmark"].lower() or "building3d" not in real["benchmark"].lower():
        raise ValueError("Evaluation benchmark labels are incorrect")
    print("PASS: both evaluation reports match the checkpoint", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "full_roof":
        raise ValueError("Expected the full-roof checkpoint")
    preprocessing = plain(checkpoint["preprocessing"])
    if preprocessing["normalization"] != NORMALIZATION or preprocessing["points"] != 1024:
        raise ValueError("Unexpected preprocessing")
    model_config = plain(checkpoint["model_config"])
    if checkpoint["epoch"] != synthetic["checkpoint_epoch"] or checkpoint["epoch"] != real["checkpoint_epoch"]:
        raise ValueError("Evaluation epoch does not match checkpoint epoch")

    runtime = package_versions([
        "numpy", "scipy", "scikit-learn", "easydict", "pyyaml", "safetensors",
    ])
    build = package_versions(["setuptools", "wheel", "ninja"])
    torch_version = str(torch.__version__).split("+")[0]
    if torch.version.cuda != "13.2":
        raise ValueError("This exporter generates setup instructions for the tested CUDA 13.2 environment")
    cuda_index = "cu132"
    function = extract_predict_function(evaluator)
    inference_source = INFERENCE_TEMPLATE.replace("__PREDICT_GRAPHS__", function)
    for name, source in (("inference.py", inference_source),
                         ("install_cuda_extension.py", INSTALL_TEMPLATE),
                         ("verify_release.py", VERIFY_TEMPLATE)):
        compile(source, name, "exec")

    state = {}
    for name, tensor in checkpoint["model"].items():
        if not torch.is_tensor(tensor) or tensor.layout != torch.strided:
            raise ValueError(f"Unsupported weight: {name}")
        tensor = tensor.detach().cpu().contiguous().clone()
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite weight: {name}")
        state[name] = tensor

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        save_file(state, str(staging / "model.safetensors"), metadata={
            "format": "pt", "model": "Koor-building-reconstruction-v1",
            "source_checkpoint_sha256": checkpoint_hash,
        })
        restored = load_file(str(staging / "model.safetensors"), device="cpu")
        if state.keys() != restored.keys() or any(
                state[key].dtype != restored[key].dtype
                or state[key].shape != restored[key].shape
                or not torch.equal(state[key], restored[key]) for key in state):
            raise RuntimeError("Safetensors round-trip mismatch")
        print(f"PASS: exact weight round trip ({len(state)} tensors)", flush=True)
        del restored

        copied = copy_model_source(project / "Point2Roof", staging / "Point2Roof")
        write_json(staging / "config.json", {
            "format_version": 1, "model_name": "Koor-building-reconstruction-v1",
            "model_type": "koor_point2roof", "model_config": model_config,
            "checkpoint_epoch": checkpoint["epoch"], "input_channels": 3,
            "output_type": "roof_wireframe", "inference_precision": "float32",
        })
        write_json(staging / "preprocessing.json", preprocessing)
        write_json(staging / "evaluation/synthetic_test.json", synthetic)
        write_json(staging / "evaluation/building3d_zero_shot.json", real)
        (staging / "inference.py").write_text(inference_source)
        (staging / "install_cuda_extension.py").write_text(INSTALL_TEMPLATE)
        (staging / "verify_release.py").write_text(VERIFY_TEMPLATE)
        (staging / "requirements.txt").write_text(
            "# Install the CUDA PyTorch build separately; see README.\n"
            + "".join(f"{name}=={version}\n" for name, version in runtime.items()))
        (staging / "requirements-build.txt").write_text(
            "".join(f"{name}=={version}\n" for name, version in build.items()))

        manifest_path = project / "data/prepared/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        split_ids = {}
        for split in ("train", "val", "test"):
            lines = (project / f"data/prepared/{split}.txt").read_text().splitlines()
            split_ids[split] = ["/".join(Path(line.strip()).parts[-2:])
                                for line in lines if line.strip()]
        if any(len(ids) != len(set(ids)) for ids in split_ids.values()):
            raise ValueError("Duplicate sample IDs in saved split lists")
        split_counts = {split: len(ids) for split, ids in split_ids.items()}
        if split_counts != {"train": 14400, "val": 1599, "test": 1600}:
            raise ValueError(f"Unexpected training split counts: {split_counts}")
        write_json(staging / "reproducibility/synthetic_split_ids.json", split_ids)
        archive = staging / "reproducibility/training_source"
        archive.mkdir(parents=True)
        for name in ("train_corners.py", "train_roofs.py", "prepare_dataset.py", "evaluate_roofs.py"):
            shutil.copy2(project / "scripts" / name, archive / name)
        shutil.copy2(Path(__file__).resolve(), archive / "export_hf_release.py")
        commit, patch = git_metadata(project / "Point2Roof")
        if patch:
            (staging / "reproducibility/point2roof_changes.patch").write_text(patch + "\n")

        setting_keys = ("epochs", "batch_size", "workers", "corner_lr", "head_lr",
                        "freeze_corner_epochs", "tolerance", "seed", "max_batches")
        settings = {key: checkpoint.get("settings", {}).get(key) for key in setting_keys}
        write_json(staging / "training_metadata.json", {
            "source_checkpoint_sha256": checkpoint_hash,
            "selected_full_roof_epoch": checkpoint["epoch"],
            "best_validation_selection_score": checkpoint["best_score"],
            "checkpoint_selection": "validation vertex/edge micro-F1 harmonic mean; normalized tolerance 0.05",
            "settings": settings, "training_implementation": checkpoint.get("implementation", {}),
            "split_counts": split_counts, "source_manifest_sha256": digest(manifest_path),
            "dataset_fingerprint": checkpoint.get("dataset_fingerprint"),
            "split_grouping": manifest.get("grouping"),
            "excluded_train_samples_matching_test_geometry": manifest.get("excluded_train_samples_matching_test_geometry"),
            "upstream": "https://github.com/Li-Li-Whu/Point2Roof",
            "upstream_commit": commit,
        })
        write_json(staging / "environment.json", {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch.__version__), "torch_cuda_build": torch.version.cuda,
            "runtime_packages": runtime, "build_packages": build,
            "export_device": "cpu", "gpu_release_smoke_test": "pending",
        })

        # A new procedural fixture, independent of the training/benchmark datasets.
        rng = np.random.default_rng(42)
        x, y = rng.uniform(-6, 6, 1024), rng.uniform(-4, 4, 1024)
        z = 5 + 2 * (1 - np.abs(y) / 4)
        (staging / "examples").mkdir()
        np.savetxt(staging / "examples/demo_roof.xyz", np.column_stack((x, y, z)), fmt="%.9f")
        (staging / "examples/README.md").write_text(
            "# Procedural demo\n\nA mathematical gable-roof point cloud generated by the exporter. "
            "It tests I/O and inference; it is not sampled from either benchmark and has no reported accuracy score.\n")
        (staging / "README.md").write_text(model_card(
            {"checkpoint_epoch": checkpoint["epoch"]}, synthetic, real,
            runtime, torch_version, cuda_index, split_counts))
        (staging / "LICENSING.md").write_text(
            "# Licensing status\n\n"
            "The bundled Point2Roof software retains its original MIT notice in Point2Roof/LICENSE.\n\n"
            "A license for Koor weights and original Koor code has not been selected. The separately hosted "
            "Point2Roof synthetic training dataset's applicable terms have not yet been established in this workflow. "
            "Do not infer its data license from the software repository's MIT license. Confirm the data and "
            "model-release terms before choosing public Hub license metadata.\n\n"
            "Building3D is evaluation-only for v1 and lists CC BY-NC-SA 4.0. Its raw point clouds and annotations "
            "are not included. Evaluation does not mean v1 was trained on Building3D.\n")
        (staging / ".gitattributes").write_text("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
        (staging / ".gitignore").write_text(
            ".venv/\n__pycache__/\n*.pyc\n*.so\nbuild/\n*.egg-info/\n*.pt\n*.pth\n"
            "demo_prediction.json\ndemo_prediction.obj\nverification.json\n")

        # Check generated Python files without executing model code or GPU work.
        python_files = list(staging.rglob("*.py"))
        for path in python_files:
            compile(path.read_text(), str(path), "exec")
        write_json(staging / "export_report.json", {
            "weights_round_trip": "passed_exact",
            "evaluation_checkpoint_identity": "passed_both_reports",
            "python_syntax": "passed", "python_files_checked": len(python_files),
            "weight_tensor_count": len(state),
            "state_dict_elements_including_buffers": sum(tensor.numel() for tensor in state.values()),
            "vendored_source_files": copied,
            "gpu_release_smoke_test": "pending: run verify_release.py on the server",
            "model_license_assignment": "not_selected", "uploaded": False,
        })
        hashes = {str(path.relative_to(staging)): digest(path)
                  for path in sorted(staging.rglob("*")) if path.is_file()}
        write_json(staging / "SHA256SUMS.json", hashes)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise

    print("PASS: generated Python syntax checked", flush=True)
    print("Release package:", output)
    print("GPU package verification remains to be run; nothing was uploaded.")
    print("Model/data licensing status is documented in LICENSING.md.")


if __name__ == "__main__":
    main()
