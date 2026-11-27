#!/usr/bin/env python3
"""Encode explicit triangle coordinates and turn them into a Three.js module."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from export_3dbag_building import threejs_module


DEFAULT_BUILDING = "NL.IMBAG.Pand.1655100000492238"
MM_PER_M = 1000
MAX_COORD_MM = 1_000_000
MAX_TRIANGLES = 4096


def encode_geometry(vertices, faces):
    millimeters = np.rint(np.asarray(vertices, dtype=np.float64)[faces] * MM_PER_M)
    if not np.isfinite(millimeters).all() or np.any(np.abs(millimeters) > MAX_COORD_MM):
        raise ValueError("Reference coordinates are outside the supported millimetre range")
    return json.dumps({"triangles_mm": millimeters.astype(np.int64).reshape(-1, 9).tolist()},
                      separators=(",", ":"))


def decode_geometry(source):
    payload = json.loads(source)
    if not isinstance(payload, dict) or set(payload) != {"triangles_mm"}:
        raise ValueError("Expected a JSON object with a triangles_mm field")
    rows = payload["triangles_mm"]
    if not isinstance(rows, list) or not 4 <= len(rows) <= MAX_TRIANGLES:
        raise ValueError(f"Expected 4 to {MAX_TRIANGLES} triangles")

    vertex_ids = {}
    vertices = []
    faces = []
    for row in rows:
        if (not isinstance(row, list) or len(row) != 9
                or any(type(value) is not int or abs(value) > MAX_COORD_MM for value in row)):
            raise ValueError("Each triangle must contain nine bounded integer millimetre coordinates")
        corners = [tuple(row[start:start + 3]) for start in (0, 3, 6)]
        a, b, c = np.asarray(corners, dtype=np.int64)
        if not np.any(np.cross(b - a, c - a)):
            raise ValueError("Triangle has zero area")
        face = []
        for point in corners:
            if point not in vertex_ids:
                vertex_ids[point] = len(vertices)
                vertices.append(point)
            face.append(vertex_ids[point])
        faces.append(face)
    vertices_m = np.asarray(vertices, dtype=np.float64) / MM_PER_M
    if np.ptp(vertices_m[:, 1]) == 0 or np.linalg.matrix_rank(vertices_m - vertices_m[0], tol=1e-7) < 3:
        raise ValueError("Geometry has no three-dimensional extent")
    return vertices_m, np.asarray(faces, dtype=np.int64)


def validation_roundtrip(shard, building_id, output):
    if output.exists() or output.is_symlink():
        raise ValueError("Choose a new output directory")
    split = shard / "splits-geographic"
    summary = json.loads((split / "summary.json").read_text(encoding="utf-8"))
    asset_root = (split / summary["asset_paths_relative_to"]).resolve()
    with (split / "validation.jsonl").open(encoding="utf-8") as manifest:
        row = next((item for item in map(json.loads, filter(str.strip, manifest))
                    if item["building_id"] == building_id), None)
    if row is None:
        raise ValueError(f"Building is absent from the validation split: {building_id}")
    reference = trimesh.load(asset_root / row["mesh_path"], force="mesh", process=False)
    target = encode_geometry(reference.vertices, reference.faces)
    vertices, faces = decode_geometry(target)
    expected = reference.vertices[reference.faces]
    compiled = vertices[faces]
    max_error = float(np.max(np.abs(expected - compiled)))
    if max_error > 0.000501:
        raise ValueError(f"Compiled geometry differs from the reference by {max_error:.6f} m per axis")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    report = {"building_id": building_id, "target_units": "millimeters", "source": "validation reference GLB",
              "triangles": len(faces), "vertices": len(vertices), "max_axis_error_m": max_error,
              "compiled_mesh_watertight": bool(mesh.is_watertight)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".geometry-roundtrip-") as temp:
        staged = Path(temp) / "geometry"
        staged.mkdir()
        (staged / "geometry.json").write_text(target + "\n", encoding="utf-8")
        (staged / "building.mjs").write_text(threejs_module(vertices, faces), encoding="utf-8")
        (staged / "building.glb").write_bytes(mesh.export(file_type="glb"))
        (staged / "roundtrip.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staged.rename(output)
    print(f"PASS: {building_id} | {len(faces)} triangles | max per-axis difference {max_error * 1000:.3f} mm")
    print(f"Wrote geometry.json, building.mjs and building.glb to {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--building-id", default=DEFAULT_BUILDING)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        validation_roundtrip(args.shard, args.building_id, args.output)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Round-trip failed: {error}\n")


if __name__ == "__main__":
    main()
