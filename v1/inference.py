"""Koor point-cloud-to-roof-wireframe inference; CUDA extension required."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from easydict import EasyDict
from safetensors.torch import load_file


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
