"""Evaluate Koor wireframes with target-free inference on a frozen test split.

Save in the project's scripts/ directory. Only load checkpoints you generated
or trust: the training checkpoint format requires weights_only=False.
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment


PROJECT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_paths(path):
    paths = [Path(line.strip()).expanduser().resolve()
             for line in path.read_text().splitlines() if line.strip()]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError(f"Empty list or duplicate sample paths: {path}")
    return paths


class PointCloudDataset:
    """Read point clouds only. Targets are loaded separately after inference."""

    def __init__(self, paths, npoints, seed):
        self.paths, self.npoints, self.seed = paths, npoints, seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        folder = self.paths[index]
        points = np.loadtxt(folder / "points.xyz", dtype=np.float64,
                            usecols=(0, 1, 2), ndmin=2)
        if len(points) < 1 or not np.isfinite(points).all():
            raise ValueError(f"Invalid point cloud: {folder}")
        lower, upper = points.min(0), points.max(0)
        center = (lower + upper) / 2
        scale = float((upper - lower).max())
        if scale <= 0:
            raise ValueError(f"Degenerate point cloud: {folder}")
        normalized = (points - center) / scale + 0.5
        rng = np.random.default_rng(self.seed + index)
        if len(points) >= self.npoints:
            selected = rng.choice(len(points), self.npoints, replace=False)
        else:
            selected = np.concatenate([
                np.arange(len(points)),
                rng.choice(len(points), self.npoints - len(points), replace=True),
            ])
            rng.shuffle(selected)
        return {
            "points": normalized[selected].astype(np.float32),
            "folder": str(folder), "center": center, "scale": scale,
            "sample_id": f"{folder.parent.name}/{folder.name}",
            "source_point_count": len(points),
        }


def collate_clouds(records):
    import torch
    points = torch.from_numpy(np.stack([record["points"] for record in records]))
    metadata = [{key: value for key, value in record.items() if key != "points"}
                for record in records]
    return points, metadata


def read_truth(path):
    vertices, edges, records = [], set(), []
    for line in path.read_text().splitlines():
        fields = line.split("#")[0].split()
        if fields:
            records.append(fields)
            if fields[0] == "v":
                vertices.append([float(value) for value in fields[1:4]])
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"Invalid target vertices: {path}")
    seen = 0
    for fields in records:
        if fields[0] == "v":
            seen += 1
        if fields[0] not in {"f", "l"}:
            continue
        indices = []
        for token in fields[1:]:
            value = int(token.split("/")[0])
            if value == 0:
                raise ValueError(f"Invalid OBJ index zero: {path}")
            index = value - 1 if value > 0 else seen + value
            if not 0 <= index < len(vertices):
                raise ValueError(f"Invalid target edge: {path}")
            indices.append(index)
        pairs = list(zip(indices, indices[1:]))
        if fields[0] == "f" and len(indices) > 2:
            pairs.append((indices[-1], indices[0]))
        edges.update(tuple(sorted((a, b))) for a, b in pairs if a != b)
    if not edges:
        raise ValueError(f"Target has no edges: {path}")
    return vertices, edges


def counts_to_metrics(tp, predicted, truth):
    return {
        "tp": int(tp), "predicted": int(predicted), "truth": int(truth),
        "precision": float(tp / predicted) if predicted else 0.0,
        "recall": float(tp / truth) if truth else 0.0,
        "f1": float(2 * tp / (predicted + truth)) if predicted + truth else 0.0,
    }


def score_graph(vertices, predicted_edges, truth_vertices, truth_edges, tolerance):
    mapping = {}
    distance_sum, assignments = 0.0, 0
    if len(vertices):
        distances = np.linalg.norm(vertices[:, None] - truth_vertices[None], axis=-1)
        rows, columns = linear_sum_assignment(distances)
        assigned_distances = distances[rows, columns]
        distance_sum = float(assigned_distances.sum())
        assignments = len(rows)
        mapping = {int(a): int(b) for a, b in zip(rows, columns)
                   if distances[a, b] <= tolerance}
    edges = {tuple(sorted(map(int, edge))) for edge in predicted_edges}
    edge_tp = sum(
        a in mapping and b in mapping
        and tuple(sorted((mapping[a], mapping[b]))) in truth_edges
        for a, b in edges
    )
    return {
        "vertex": counts_to_metrics(len(mapping), len(vertices), len(truth_vertices)),
        "edge": counts_to_metrics(edge_tp, len(edges), len(truth_edges)),
        "assignment_distance_sum_normalized": distance_sum,
        "assignment_count": assignments,
    }


def summarize(scores):
    result = {}
    for kind in ("vertex", "edge"):
        counts = [sum(item[kind][field] for item in scores)
                  for field in ("tp", "predicted", "truth")]
        result[kind] = counts_to_metrics(*counts)
        result[kind]["macro_f1"] = float(np.mean([item[kind]["f1"] for item in scores]))
    assigned = sum(item["assignment_count"] for item in scores)
    result["mean_assigned_corner_error_normalized"] = (
        sum(item["assignment_distance_sum_normalized"] for item in scores) / assigned
        if assigned else None
    )
    result["assigned_corner_count"] = assigned
    return result


def predict_graphs(model, points):
    """Inference has no access to ground-truth vertices, edges, or matching."""
    import torch
    batch_size = points.shape[0]
    # The legacy PointNet2 forward reads this key even in eval mode;
    # None supplies no label information. Target assignment is disabled by eval().
    batch = {"points": points, "batch_size": batch_size, "vectors": None}
    batch = model.keypoint_det_net(batch)
    try:
        batch = model.cluster_refine_net(batch)
    except SystemExit as error:
        if error.code not in (None, 0):
            raise
        return [{"vertices": np.empty((0, 3)),
                 "pairs": np.empty((0, 2), dtype=np.int64),
                 "scores": np.empty(0), "reason": "no_corner_proposals"}
                for _ in range(batch_size)]

    features = batch["keypoint_features"]
    ids = batch["keypoint"][:, 0].long()
    pairs = []
    for building in range(batch_size):
        rows = torch.where(ids == building)[0]
        if len(rows) >= 2:
            pairs.append(torch.combinations(rows, r=2))
    if pairs:
        pairs = torch.cat(pairs)
        head = model.edge_att_net
        fused = head.att_layer(features[pairs[:, 0]], features[pairs[:, 1]])
        logits = head.cls_fc(head.drop(head.shared_fc(fused))).flatten()
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite edge logits")
        scores = logits.sigmoid().cpu().numpy()
        pair_array = pairs.cpu().numpy()
    else:
        scores = np.empty(0)
        pair_array = np.empty((0, 2), dtype=np.int64)

    vertices = batch["refined_keypoint"].cpu().numpy().astype(np.float64)
    ids = ids.cpu().numpy()
    if not np.isfinite(vertices).all():
        raise RuntimeError("Non-finite predicted corners")
    results = []
    for building in range(batch_size):
        rows = np.flatnonzero(ids == building)
        selected = ids[pair_array[:, 0]] == building
        to_local = np.full(len(vertices), -1, dtype=np.int64)
        to_local[rows] = np.arange(len(rows))
        results.append({
            "vertices": vertices[rows],
            "pairs": to_local[pair_array[selected]],
            "scores": scores[selected],
            "reason": "no_corner_proposals" if len(rows) == 0 else None,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT / "runs/koor-roofs-v1/best.pt")
    parser.add_argument("--test-list", type=Path,
                        default=PROJECT / "data/prepared/test.txt")
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "evaluations/koor-v1-synthetic-test")
    parser.add_argument("--expected-samples", type=int, default=1600)
    parser.add_argument("--benchmark", default="Point2Roof synthetic held-out test; Koor evaluation protocol",
                        help="Descriptive benchmark label recorded in summary.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--tolerances", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.expected_samples < 1:
        parser.error("Invalid sample count, batch size, or worker count")
    if not 0 <= args.edge_threshold <= 1 or any(t <= 0 for t in args.tolerances):
        parser.error("Invalid edge threshold or corner tolerance")
    args.tolerances = sorted(set(args.tolerances))

    paths = load_paths(args.test_list)
    if len(paths) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} samples, found {len(paths)}")
    for name in ("train.txt", "val.txt"):
        other = args.test_list.parent / name
        if other.exists() and set(paths).intersection(load_paths(other)):
            raise ValueError(f"Evaluation samples overlap {other}")
    for folder in paths:
        for name in ("points.xyz", "polygon.obj"):
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)

    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("summary.json", "predictions.jsonl")):
        raise FileExistsError("Evaluation results already exist; choose a new --output")

    import torch
    import scipy
    from torch.utils.data import DataLoader
    from easydict import EasyDict
    sys.path.insert(0, str(PROJECT / "Point2Roof"))
    from model.roofnet import RoofNet

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    checkpoint_hash = sha256_file(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "full_roof":
        raise ValueError("Expected a Koor full_roof checkpoint")
    preprocessing = checkpoint["preprocessing"]
    if preprocessing["normalization"] != "full-cloud bbox center; max XYZ span; +0.5":
        raise ValueError("Unsupported checkpoint normalization")
    config = EasyDict(copy.deepcopy(checkpoint["model_config"]))
    model = RoofNet(copy.deepcopy(config.MODEL)).cuda()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    dataset = PointCloudDataset(paths, int(preprocessing["points"]), args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        collate_fn=collate_clouds,
                        generator=torch.Generator().manual_seed(args.seed))
    all_scores = {str(t): [] for t in args.tolerances}
    empty_corners, empty_edges, processed = 0, 0, 0
    inference_seconds = 0.0
    partial = args.output / "predictions.jsonl.partial"
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Checkpoint epoch:", checkpoint["epoch"], flush=True)
    print("Test samples:", len(dataset), "| Target-free inference", flush=True)

    with partial.open("w") as output, torch.inference_mode():
        for step, (points, records) in enumerate(loader):
            points = points.cuda(non_blocking=True)
            if step == 0:
                predict_graphs(model, points)  # Unscored timing warm-up; no labels.
            torch.cuda.synchronize()
            start = time.perf_counter()
            predictions = predict_graphs(model, points)
            torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - start

            # Only now read target geometry; it was never passed to the model.
            for record, prediction in zip(records, predictions):
                vertices = prediction["vertices"]
                edges = prediction["pairs"][prediction["scores"] >= args.edge_threshold]
                truth_vertices, truth_edges = read_truth(Path(record["folder"]) / "polygon.obj")
                truth_normalized = (truth_vertices - record["center"]) / record["scale"] + 0.5
                metrics = {}
                for tolerance in args.tolerances:
                    score = score_graph(vertices, edges, truth_normalized, truth_edges, tolerance)
                    all_scores[str(tolerance)].append(score)
                    metrics[str(tolerance)] = score
                empty_corners += len(vertices) == 0
                empty_edges += len(edges) == 0
                processed += 1
                item = {
                    "sample_id": record["sample_id"],
                    "source_point_count": record["source_point_count"],
                    "source_center": record["center"].tolist(),
                    "source_scale": record["scale"],
                    "vertices_normalized": vertices.tolist(),
                    "vertices_source_coordinates": (
                        (vertices - 0.5) * record["scale"] + record["center"]
                    ).tolist(),
                    "candidate_edges": prediction["pairs"].tolist(),
                    "uncalibrated_edge_scores": prediction["scores"].tolist(),
                    "accepted_edges": edges.tolist(),
                    "failure_reason": prediction["reason"], "metrics": metrics,
                }
                output.write(json.dumps(item, allow_nan=False) + "\n")
            if (step + 1) % 10 == 0 or processed == len(dataset):
                print(f"Evaluated {processed}/{len(dataset)}", flush=True)

    assert processed == len(dataset)
    report = {
        "benchmark": args.benchmark,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint["epoch"],
        "test_list": str(args.test_list.resolve()),
        "test_list_sha256": sha256_file(args.test_list),
        "samples": processed, "seed": args.seed, "preprocessing": preprocessing,
        "protocol": {
            "model_mode": "eval; target-free inference",
            "vertex_matching": "minimum-total-distance Hungarian assignment, then tolerance cutoff",
            "edge_matching": "both endpoints must match a ground-truth edge; undirected",
            "tolerance_units": "fraction of full-cloud maximum XYZ bounding-box span",
            "edge_threshold": args.edge_threshold,
            "corner_score_threshold": float(config.MODEL.ClusterRefineNet.ScoreThresh),
            "assignment_error": "mean over all assigned corners, including those outside tolerance; unmatched corners omitted",
            "published_protocol_reproduction": False,
        },
        "metrics_by_tolerance": {key: summarize(value) for key, value in all_scores.items()},
        "empty_corner_buildings": empty_corners,
        "empty_edge_buildings": empty_edges,
        "empty_corner_rate": empty_corners / processed,
        "empty_edge_rate": empty_edges / processed,
        "inference_seconds_excluding_warmup": inference_seconds,
        "inference_buildings_per_second": processed / inference_seconds,
        "timing_scope": "batched GPU inference plus graph postprocessing; excludes input loading and target scoring",
        "batch_size": args.batch_size,
        "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "gpu": torch.cuda.get_device_name(0)},
    }
    partial.replace(args.output / "predictions.jsonl")
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for tolerance, metrics in report["metrics_by_tolerance"].items():
        print(f"Tolerance {tolerance}: vertex F1={metrics['vertex']['f1']:.4f}, "
              f"edge F1={metrics['edge']['f1']:.4f}")
    print(f"Empty corners: {empty_corners}/{processed}; empty edges: {empty_edges}/{processed}")
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
