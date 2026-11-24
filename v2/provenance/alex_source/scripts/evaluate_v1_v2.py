"""Evaluate frozen v1/v2 checkpoints on identical synthetic/Building3D splits.

Save in the v2 project's scripts directory. Runs the published v1 evaluator
unchanged in separate processes. v1 safetensors are wrapped into an evaluation-
only checkpoint; v2 best_joint.pt is snapshotted before any held-out scoring.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from mixed_utils import PROJECT, digest, inspect_data
from train_mixed_distill import initial_release


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compare(v1, v2):
    for key in ("samples", "test_list_sha256", "seed", "preprocessing", "protocol", "batch_size", "environment"):
        if v1[key] != v2[key]:
            raise ValueError(f"Incomparable evaluation reports: {key}")
    if set(v1["metrics_by_tolerance"]) != set(v2["metrics_by_tolerance"]):
        raise ValueError("Tolerance sets differ")
    changes = {}
    for tolerance in v1["metrics_by_tolerance"]:
        changes[tolerance] = {}
        for kind in ("vertex", "edge"):
            before = v1["metrics_by_tolerance"][tolerance][kind]["f1"]
            after = v2["metrics_by_tolerance"][tolerance][kind]["f1"]
            changes[tolerance][kind] = {"v1_f1": before, "v2_f1": after, "absolute_change": after - before}
    return {
        "samples": v1["samples"], "test_list_sha256": v1["test_list_sha256"],
        "metrics_by_tolerance": changes,
        "empty_corner_buildings": {"v1": v1["empty_corner_buildings"], "v2": v2["empty_corner_buildings"]},
        "empty_edge_buildings": {"v1": v1["empty_edge_buildings"], "v2": v2["empty_edge_buildings"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v1-release", type=Path, default=PROJECT / "models/koor-v1")
    parser.add_argument("--v2-checkpoint", type=Path,
                        default=PROJECT / "runs/koor-mixed-distill-alex-v2/best_joint.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid batch size or workers")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new evaluation directory: {output}")
    config, preprocessing, training_metadata, weights, release_hash = initial_release(args.v1_release)
    v2_source = args.v2_checkpoint.expanduser().resolve()
    v2_hash = digest(v2_source)
    # Only load your own trusted training checkpoints.
    selected = torch.load(v2_source, map_location="cpu", weights_only=False)
    if selected.get("stage") != "full_roof" or selected.get("run_mode") != "full":
        raise ValueError("Expected a full-training roof checkpoint, not a smoke checkpoint")
    if selected.get("model_version") != "koor-alex-mixed-distill-v1":
        raise ValueError("Expected the Alex mixed-distillation experiment")
    if selected["model_config"] != config or selected["preprocessing"] != preprocessing:
        raise ValueError("Model configuration or preprocessing differs from v1")
    if selected["initialization_release_sha256"] != release_hash:
        raise ValueError("v1 release differs from v2's actual initialization")
    if selected["cursor"]["next_batch"] != 0 or selected["epoch"] <= 0:
        raise ValueError("Select an epoch-complete trained checkpoint")
    if selected["best_joint_epoch"] != selected["epoch"]:
        raise ValueError("Checkpoint does not identify itself as the selected best joint epoch")
    for relative_path, expected in selected["implementation"].items():
        if digest(PROJECT / relative_path) != expected:
            raise ValueError(f"Training/evaluation source changed: {relative_path}")
    counts, fingerprint = inspect_data(PROJECT / "data/prepared", PROJECT / "data/building3d/koor-v2")
    if selected["dataset_fingerprint"] != fingerprint:
        raise ValueError("Prepared manifests or path lists changed since training")
    if counts["synthetic"]["test"] != 1600 or counts["real"]["holdout"] != 561:
        raise ValueError("Unexpected held-out split counts")
    output.mkdir(parents=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    v1_adapter = checkpoints / "v1-evaluation-only.pt"
    torch.save({"stage": "full_roof", "model_version": "koor-v1",
                "epoch": training_metadata["selected_full_roof_epoch"],
                "model": weights, "model_config": config, "preprocessing": preprocessing,
                "purpose": "Evaluation-only wrapper of unchanged published safetensors",
                "source_release_sha256": release_hash}, v1_adapter)
    roundtrip = torch.load(v1_adapter, map_location="cpu", weights_only=False)["model"]
    if set(roundtrip) != set(weights) or not all(torch.equal(value, roundtrip[key]) for key, value in weights.items()):
        raise RuntimeError("v1 adapter changed weights")
    v2_snapshot = checkpoints / "v2-selected.pt"
    shutil.copy2(v2_source, v2_snapshot)
    if digest(v2_source) != v2_hash or digest(v2_snapshot) != v2_hash:
        raise RuntimeError("Selected v2 checkpoint changed while snapshotting")
    write_json(output / "selection.json", {
        "v1_epoch": training_metadata["selected_full_roof_epoch"],
        "v1_source_release_sha256": release_hash, "v1_adapter_sha256": digest(v1_adapter),
        "v1_source_weights_sha256": digest(args.v1_release / "model.safetensors"),
        "v2_epoch": selected["epoch"], "v2_source_checkpoint": str(v2_source),
        "v2_checkpoint_sha256": v2_hash, "dataset_fingerprint": fingerprint,
        "checkpoint_selection": "Existing best_joint.pt selected on validation only; frozen before held-out scoring",
        "evaluations": ["synthetic test: 1600", "custom Building3D holdout: 561"],
        "published_benchmark_protocol_reproduction": False,
    })
    print("PASS: unchanged v1 weight round trip; v2 checkpoint snapshotted", flush=True)
    print("v1 epoch:", training_metadata["selected_full_roof_epoch"], "| v2 selected epoch:", selected["epoch"], flush=True)
    del roundtrip, weights, selected
    evaluator = PROJECT / "v1/scripts/evaluate_roofs.py"
    for name, path_list, count, label in (
        ("synthetic-test", PROJECT / "data/prepared/test.txt", 1600, "Point2Roof synthetic held-out test; Koor evaluation protocol"),
        ("building3d-holdout", PROJECT / "data/building3d/koor-v2/holdout.txt", 561, "Building3D Entry-level custom spatial holdout; Koor evaluation protocol"),
    ):
        reports = {}
        for model_name, checkpoint in (("v1", v1_adapter), ("v2", v2_snapshot)):
            destination = output / f"{model_name}-{name}"
            print(f"\n=== {model_name}: {name} ({count} samples) ===", flush=True)
            subprocess.run([
                sys.executable, "-u", str(evaluator), "--checkpoint", str(checkpoint),
                "--test-list", str(path_list), "--expected-samples", str(count),
                "--benchmark", label, "--batch-size", str(args.batch_size),
                "--workers", str(args.workers), "--seed", "42", "--edge-threshold", "0.5",
                "--tolerances", "0.01", "0.02", "0.05", "--output", str(destination),
            ], check=True, cwd=PROJECT)
            reports[model_name] = json.loads((destination / "summary.json").read_text())
        comparison = compare(reports["v1"], reports["v2"])
        write_json(output / f"comparison-{name}.json", comparison)
        print(f"\nPASS: comparable {name} reports", flush=True)
        for tolerance, metrics in comparison["metrics_by_tolerance"].items():
            for kind, values in metrics.items():
                print(f"  {tolerance} {kind} F1: {values['v1_f1']:.4f} -> {values['v2_f1']:.4f} ({values['absolute_change']:+.4f})", flush=True)
    print("\nPASS: all four held-out evaluations completed", flush=True)
    print("Results:", output, flush=True)


if __name__ == "__main__":
    main()
