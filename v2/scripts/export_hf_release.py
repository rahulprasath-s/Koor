"""Export a reviewable Koor v2 Hugging Face package; uploads nothing.

Run on Alex with its venv. Can be saved under scripts/ in the original flat
workspace or v2/scripts/ in the maintained checkout. Load only your own .pt.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
if PROJECT.name == "v2":
    PROJECT = PROJECT.parent
MODEL_NAME = "Koor-building-reconstruction-v2"
CODE_REVISION = "4baa5ea68616a88e5acd3c69ec96ad43d493d3f5"


def load_helpers(root):
    path = root / "v1/scripts/export_hf_release.py"
    spec = importlib.util.spec_from_file_location("koor_v1_export_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load export helpers: {path}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def cleaned_report(report):
    result = json.loads(json.dumps(report))
    for key in ("checkpoint", "test_list"):
        if key in result:
            result[key] = Path(result[key]).name
    return result


def paired_reports(directory, checkpoint_hash, epoch):
    reports = {}
    for dataset, count in (("synthetic-test", 1600), ("building3d-holdout", 561)):
        v1, v2 = [json.loads((directory / f"{model}-{dataset}/summary.json").read_text()) for model in ("v1", "v2")]
        for key in ("samples", "test_list_sha256", "seed", "preprocessing", "protocol", "batch_size", "environment"):
            if v1[key] != v2[key]:
                raise ValueError(f"Different paired settings: {dataset}/{key}")
        if v2["samples"] != count or v2["checkpoint_sha256"] != checkpoint_hash or v2["checkpoint_epoch"] != epoch:
            raise ValueError(f"Evaluation checkpoint/count/epoch mismatch: {dataset}")
        if v2["protocol"]["model_mode"] != "eval; target-free inference" or v2["protocol"]["edge_threshold"] != .5:
            raise ValueError("Unexpected evaluation protocol")
        if set(v2["metrics_by_tolerance"]) != {"0.01", "0.02", "0.05"}:
            raise ValueError("Unexpected tolerances")
        for report in (v1, v2):
            for metric in report["metrics_by_tolerance"].values():
                for kind in ("vertex", "edge"):
                    if not 0 <= metric[kind]["f1"] <= 1:
                        raise ValueError("Invalid evaluation score")
        reports[f"v1-{dataset}"] = cleaned_report(v1)
        reports[f"v2-{dataset}"] = cleaned_report(v2)
    return reports


def model_card(reports, checkpoint, torch_version):
    rows = []
    for dataset, label in (("synthetic-test", "Synthetic test (1,600)"), ("building3d-holdout", "Building3D spatial holdout (561)")):
        a, b = reports[f"v1-{dataset}"], reports[f"v2-{dataset}"]
        for tolerance in ("0.01", "0.02", "0.05"):
            v1, v2 = a["metrics_by_tolerance"][tolerance], b["metrics_by_tolerance"][tolerance]
            rows.append(f"| {label} | {float(tolerance):.0%} | {v1['vertex']['f1']:.4f} | {v2['vertex']['f1']:.4f} | {v1['edge']['f1']:.4f} | {v2['edge']['f1']:.4f} |")
    real = reports["v2-building3d-holdout"]
    return f'''---
library_name: pytorch
base_model: Uruva/Koor-building-reconstruction-v1
datasets:
  - Building3D/Building3D
tags:
  - point-cloud
  - roof-wireframe
  - building-reconstruction
  - point2roof
  - custom-code
  - safetensors
model_name: {MODEL_NAME}
---

# Koor v2: Building3D-adapted roof reconstruction

I trained v2 to improve the real-LiDAR gap I saw in v1. It learns from synthetic
roofs and Building3D together, with a frozen v1 corner teacher on synthetic
examples. It improves Building3D holdout F1, but loses synthetic accuracy.
This is an experimental specialization, not an improvement on every v1 metric.

Input is one roof's N-by-3 XYZ point cloud; output is 3D corners and undirected
edges. It uses the [Point2Roof architecture](https://github.com/Li-Li-Whu/Point2Roof).
It does not infer geometry from photographs or map coordinates, predict floors,
generate JavaScript, or produce roof faces, walls, textures or watertight meshes.

## Held-out evaluation

The selected checkpoint is epoch {checkpoint['epoch']}, chosen on joint
validation before test evaluation. Both models were evaluated together on Alex
with identical samples and settings. The original all-metric synthetic retention
gate was not met by any epoch; `best_retained.pt` was never created. This package
uses the validation-selected `best_joint.pt`.

| Held-out set | Tolerance | Vertex F1 v1 | Vertex F1 v2 | Edge F1 v1 | Edge F1 v2 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Scores are micro-F1. Tolerances are fractions of the full input cloud's maximum
XYZ bounding-box span, not fixed metres. Corners use minimum-total-distance
Hungarian assignment then a tolerance cutoff. An accepted edge must connect two
matched corners with a labelled edge; threshold is 0.5. Scores are uncalibrated.
These are Koor evaluations on a custom Building3D spatial holdout, not published
benchmark reproductions. See the four aggregate reports in `evaluation/`.

All six real-data F1s improved and all six synthetic F1s declined. Real roofs with
no accepted edges increased from {reports['v1-building3d-holdout']['empty_edge_buildings']}
to {real['empty_edge_buildings']} of 561. Both models had zero empty corners or
edges on the synthetic test. Useful aggregate scores do not establish reliable
individual roofs or complete building reconstruction.

## Training and data

V1 initialization: selected synthetic-training epoch 29. The student trains all
three stages with supervised losses; synthetic-only point scores and offsets
are distilled from an eval-mode frozen v1 corner detector. There is no direct
teacher loss for refined graphs or edges. Training has 15 epochs, batch size 16
(eight synthetic and eight real), FP32, corner/head learning rates 1e-5/1e-4,
Adam weight decay 1e-3, and StepLR halving every five epochs. Both distillation
weights are 1, temperature 2. The recorded numerical environment is A40,
Python 3.12.0, PyTorch {torch_version}, CUDA 12.8.

Synthetic splits are 14,400/1,599/1,600 train/validation/test, with matching
normalized geometries isolated and one source training sample excluded.
Building3D uses 4,538/533/561 train/validation/custom-holdout roofs, with
66 excluded by spatial boundary buffers. The official 583 unlabelled test clouds
were not scored. Prepared Building3D inputs use XYZ only; RGB/NIR/intensity are
unused. Original dataset coordinates/units are not inferred by the model.

## Run inference

This package requires CUDA and its compiled `pc_util` extension. Run from the
downloaded model directory in an isolated Python environment:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt -r requirements-build.txt
python install_cuda_extension.py --arch 8.6
python verify_release.py
python inference.py --input examples/demo_roof.xyz --output demo_prediction.json
```

`8.6` is A40; use `8.0` for A100 or the appropriate architecture on other GPUs.
The CUDA compiler/toolkit must be available for extension compilation. A40 is
the recorded runtime; verification on other hardware is still needed. This is
custom PyTorch inference, not an AutoModel or hosted-inference integration.

Read XYZ as float64. Normalize using the complete cloud's bbox center and one
shared maximum-span scale, add 0.5, and sample 1,024 points. Small clouds retain
all points plus replacement samples. No inference augmentation is used. Returned
vertices are transformed back into the input coordinate frame and units. No
CRS/georeferencing, ground elevation or building identity is inferred.

The procedural demo is generated mathematically and is independent of both
benchmarks. `verify_release.py` checks packaging/loading/inference, not accuracy.
Run it on your target GPU and inspect its report. A successful smoke test does
not establish benchmark performance.

## Source and provenance

[Training, evaluation and data preparation](https://github.com/rahulprasath-s/Koor/tree/{CODE_REVISION}/v2)
are in GitHub, including exact archived Alex source, split IDs, environment and
the validation history. `training_metadata.json` identifies this checkpoint and
the recorded data/source fingerprints. Original shared Point2Roof model code is
bundled for inference; raw datasets and optimizer state are not.

## Credits and licensing status

Point2Roof software retains its MIT notice in `Point2Roof/LICENSE`.
Building3D: Ruisheng Wang, Shangfeng Huang and Hongxin Yang, *Building3D: An
Urban-Scale Dataset and Benchmarks for Learning Roof Structures from Point
Clouds*, ICCV 2023. [Dataset terms](https://szusic.github.io/Building3D/) list
CC BY-NC-SA 4.0. V2 uses Building3D for training as well as evaluation.

A license for Koor weights and original Koor code has not been assigned here.
The Point2Roof software license does not establish the separately hosted
synthetic dataset's terms. See `LICENSING.md`; do not assume a permissive or
commercial-use model license from repository visibility.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.project.expanduser().resolve()
    checkpoint_path = (args.checkpoint or root / "runs/koor-mixed-distill-alex-v2/best_joint.pt").resolve()
    evaluations = (args.evaluation or root / "evaluations/v1-v2-heldout-5064695").resolve()
    output = (args.output or root / "release" / MODEL_NAME).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output: {output}")
    import torch
    from safetensors.torch import load_file, save_file
    helper = load_helpers(root)
    source_hash = helper.digest(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("stage") != "full_roof" or checkpoint.get("run_mode") != "full"
            or checkpoint.get("model_version") != "koor-alex-mixed-distill-v1"
            or checkpoint["cursor"]["next_batch"] != 0 or checkpoint["epoch"] != checkpoint["best_joint_epoch"]):
        raise ValueError("Expected the epoch-complete selected Alex best-joint checkpoint")
    selection = json.loads((evaluations / "selection.json").read_text())
    if source_hash != selection["v2_checkpoint_sha256"] or checkpoint["epoch"] != selection["v2_epoch"]:
        raise ValueError("Checkpoint differs from the frozen evaluation selection")
    for relative, expected in checkpoint["implementation"].items():
        if helper.digest(root / relative) != expected:
            raise ValueError(f"Recorded training/evaluation source changed: {relative}")
    reports = paired_reports(evaluations, source_hash, checkpoint["epoch"])
    preprocessing = helper.plain(checkpoint["preprocessing"])
    if preprocessing["points"] != 1024 or preprocessing["normalization"] != helper.NORMALIZATION:
        raise ValueError("Unsupported preprocessing")
    if any(report["preprocessing"] != preprocessing for report in reports.values()):
        raise ValueError("Evaluation and checkpoint preprocessing differ")
    evaluator = root / "v1/scripts/evaluate_roofs.py"
    predict_function = helper.extract_predict_function(evaluator)
    inference = helper.INFERENCE_TEMPLATE.replace("__PREDICT_GRAPHS__", predict_function)
    installer = helper.INSTALL_TEMPLATE.replace('default="10.0", help="B200 is 10.0"',
                                               'default="8.6", help="A40 is 8.6; A100 is 8.0"')
    verify = helper.VERIFY_TEMPLATE.replace('root / "verification.json"', 'root / "release_verification.json"')
    verify = verify.replace('"file_hashes": "passed",', '"file_hashes": "passed", "package_manifest_sha256": digest(root / "SHA256SUMS.json"),')
    for name, content in (("inference.py", inference), ("install_cuda_extension.py", installer), ("verify_release.py", verify)):
        compile(content, name, "exec")
    runtime = helper.package_versions(["numpy", "scipy", "scikit-learn", "easydict", "pyyaml", "safetensors"])
    build = helper.package_versions(["setuptools", "wheel", "ninja"])
    if str(torch.__version__) != checkpoint["torch_version"] or torch.version.cuda != checkpoint["cuda_version"]:
        raise ValueError("Export using the recorded training environment")
    state = {}
    for name, tensor in checkpoint["model"].items():
        if not torch.is_tensor(tensor) or tensor.layout != torch.strided:
            raise ValueError(f"Unsupported tensor: {name}")
        tensor = tensor.detach().cpu().contiguous().clone()
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite weight: {name}")
        state[name] = tensor
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        save_file(state, str(staging / "model.safetensors"), metadata={"format": "pt", "model": MODEL_NAME, "source_checkpoint_sha256": source_hash})
        restored = load_file(str(staging / "model.safetensors"), device="cpu")
        if set(state) != set(restored) or any(state[key].dtype != restored[key].dtype or state[key].shape != restored[key].shape or not torch.equal(state[key], restored[key]) for key in state):
            raise RuntimeError("Safetensors exact round trip failed")
        print(f"PASS: exact weight round trip ({len(state)} tensors)", flush=True)
        helper.copy_model_source(root / "v1/Point2Roof", staging / "Point2Roof")
        upstream = root / "v1/Point2Roof/UPSTREAM_README.md"
        if upstream.is_file():
            shutil.copy2(upstream, staging / "Point2Roof/UPSTREAM_README.md")
        config = {"format_version": 1, "model_name": MODEL_NAME, "model_type": "koor_point2roof",
                  "model_config": helper.plain(checkpoint["model_config"]), "checkpoint_epoch": checkpoint["epoch"],
                  "input_channels": 3, "output_type": "roof_wireframe", "inference_precision": "float32",
                  "source_checkpoint_sha256": source_hash, "base_model": "Uruva/Koor-building-reconstruction-v1"}
        helper.write_json(staging / "config.json", config)
        helper.write_json(staging / "preprocessing.json", preprocessing)
        for name, report in reports.items():
            helper.write_json(staging / "evaluation" / (name + ".json"), report)
        helper.write_json(staging / "training_metadata.json", {
            "source_checkpoint_sha256": source_hash, "selected_full_roof_epoch": checkpoint["epoch"],
            "checkpoint_selection": "best joint validation score; frozen before held-out scoring",
            "best_joint_validation_score": checkpoint["best_joint_score"], "best_joint_epoch": checkpoint["best_joint_epoch"],
            "retention_qualified_epoch": checkpoint["best_retained_epoch"],
            "split_counts": checkpoint["split_counts"], "dataset_fingerprint": checkpoint["dataset_fingerprint"],
            "initialization_release_sha256": checkpoint["initialization_release_sha256"],
            "initialization_v1_epoch": checkpoint["initialization_v1_epoch"],
            "teacher_state_sha256": checkpoint["teacher_state_sha256"],
            "training_protocol": checkpoint["training_protocol"], "original_source_hashes": checkpoint["implementation"],
            "settings": {key: value for key, value in checkpoint["settings"].items() if key not in {"initial_model", "synthetic_dir", "real_dir", "run_dir"}},
            "source_repository": "https://github.com/rahulprasath-s/Koor", "source_revision": CODE_REVISION,
        })
        for name, content in (("inference.py", inference), ("install_cuda_extension.py", installer), ("verify_release.py", verify)):
            (staging / name).write_text(content)
        (staging / "requirements.txt").write_text("# Install CUDA PyTorch separately; see README.\n" + "".join(f"{name}=={version}\n" for name, version in runtime.items()))
        (staging / "requirements-build.txt").write_text("".join(f"{name}=={version}\n" for name, version in build.items()))
        helper.write_json(staging / "environment.json", {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                          "runtime_packages": runtime, "build_packages": build, "export_device": "cpu", "gpu_package_verification": "pending"})
        (staging / "examples").mkdir()
        shutil.copy2(root / "v1/examples/demo_roof.xyz", staging / "examples/demo_roof.xyz")
        (staging / "examples/README.md").write_text("# Procedural demo\n\nMathematically generated gable roof; independent of both datasets. Used for packaging/inference checks, not accuracy evaluation.\n")
        (staging / "README.md").write_text(model_card(reports, checkpoint, str(torch.__version__)))
        (staging / "LICENSING.md").write_text("# Licensing status\n\nBundled Point2Roof software retains its MIT notice in Point2Roof/LICENSE.\n\nBuilding3D data lists CC BY-NC-SA 4.0 (https://szusic.github.io/Building3D/). V2 used it for training and evaluation. No raw benchmark geometry is bundled.\n\nThe separately hosted Point2Roof synthetic dataset's applicable terms have not been established in this workflow. Its terms must not be inferred from the upstream software's MIT license.\n\nA license for Koor weights and original Koor code has not been assigned here. Repository visibility does not establish a permissive or commercial-use license.\n")
        (staging / ".gitattributes").write_text("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
        (staging / ".gitignore").write_text(".venv*/\n__pycache__/\n*.pyc\n*.so\nbuild/\n*.egg-info/\n*.pt\n*.pth\ndemo_prediction.json\ndemo_prediction.obj\n")
        for path in staging.rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))
        helper.write_json(staging / "export_report.json", {"source_checkpoint_sha256": source_hash, "evaluation_identity": "passed_both_v2_reports",
                          "baseline_comparability": "passed_both_datasets", "weights_round_trip": "passed_exact", "weight_tensor_count": len(state),
                          "exporter_sha256": helper.digest(Path(__file__)), "template_source_sha256": helper.digest(root / "v1/scripts/export_hf_release.py"),
                          "python_syntax": "passed", "gpu_package_verification": "pending", "uploaded": False})
        helper.write_json(staging / "SHA256SUMS.json", {str(path.relative_to(staging)): helper.digest(path) for path in sorted(staging.rglob("*")) if path.is_file()})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print("PASS: both paired evaluations match the selected checkpoint; generated source syntax checked", flush=True)
    print("Release package:", output)
    print("Next: GPU package verification. Nothing was uploaded.")


if __name__ == "__main__":
    main()
