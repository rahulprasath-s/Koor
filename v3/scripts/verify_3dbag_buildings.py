#!/usr/bin/env python3
"""Verify the geometry and metadata of an exported 3DBAG collection."""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import trimesh

from export_3dbag_building import ATTRIBUTION, LICENSE_URL, LOD, threejs_module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_building(root, row):
    building_id = row["building_id"]
    require(re.fullmatch(r"NL\.IMBAG\.Pand\.[0-9]{16}", building_id), "Invalid building ID")
    directory = Path("buildings") / building_id
    expected_paths = {
        "mesh_path": (directory / "building.glb").as_posix(),
        "threejs_path": (directory / "building.mjs").as_posix(),
        "metadata_path": (directory / "metadata.json").as_posix(),
    }
    require(all(row.get(key) == path for key, path in expected_paths.items()), "Incorrect manifest paths")

    metadata = json.loads((root / row["metadata_path"]).read_text(encoding="utf-8"))
    require(metadata.get("mesh_path") == "building.glb" and metadata.get("threejs_path") == "building.mjs",
            "Incorrect file paths in metadata.json")
    require(row == {**metadata, **expected_paths}, "Metadata does not match the manifest record")
    require(metadata["lod"] == LOD, "Unexpected mesh detail level")
    require(metadata["attribution"] == ATTRIBUTION and metadata["license_url"] == LICENSE_URL,
            "Missing or incorrect source attribution")
    location = metadata["location"]
    require(-90 <= location["latitude"] <= 90 and -180 <= location["longitude"] <= 180,
            "Invalid latitude or longitude")

    mesh = trimesh.load(root / row["mesh_path"], force="mesh", process=False)
    code = (root / row["threejs_path"]).read_text(encoding="utf-8")
    positions_match = re.search(r"Float32BufferAttribute\((\[[^\]]+\]), 3\)", code)
    indices_match = re.search(r"setIndex\((\[[^\]]+\])\)", code)
    require(positions_match is not None and indices_match is not None, "Missing geometry arrays in Three.js module")
    positions = np.asarray(json.loads(positions_match[1]), dtype=np.float64).reshape(-1, 3)
    indices = np.asarray(json.loads(indices_match[1])).reshape(-1, 3)
    require(np.issubdtype(indices.dtype, np.integer), "Triangle indices must be integers")
    require(len(positions) and len(indices) and np.all(np.isfinite(positions)), "Empty or invalid mesh")
    require(np.all(0 <= indices) and np.all(indices < len(positions)), "Triangle index out of bounds")
    require(code == threejs_module(positions, indices), "Three.js module differs from the exporter template")
    require(np.array_equal(positions.astype(np.float32), mesh.vertices), "GLB and Three.js vertex positions differ")
    require(np.array_equal(indices, mesh.faces), "GLB and Three.js triangle indices differ")
    require(metadata["triangle_count"] == len(mesh.faces), "Triangle count differs from metadata")
    require(math.isclose(metadata["height_m"], float(np.ptp(mesh.vertices[:, 1])), abs_tol=1e-4),
            "GLB height differs from metadata")
    require(metadata["mesh_watertight"] is bool(mesh.is_watertight), "GLB watertight status differs from metadata")
    return bool(mesh.is_watertight)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Collection directory containing buildings.jsonl")
    parser.add_argument("--count", type=int, default=100, help="Expected number of unique buildings")
    args = parser.parse_args()
    try:
        with (args.directory / "buildings.jsonl").open(encoding="utf-8") as index:
            rows = [json.loads(line) for line in index if line.strip()]
        require(len(rows) == args.count, f"Expected {args.count} rows, found {len(rows)}")
        require(len({row["building_id"] for row in rows}) == len(rows), "Duplicate building IDs in manifest")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"FAIL: {error}\n")

    failures = 0
    watertight = 0
    for row in rows:
        try:
            watertight += verify_building(args.directory, row)
        except (OSError, KeyError, TypeError, ValueError) as error:
            print(f"FAIL {row['building_id']}: {error}")
            failures += 1
    if failures:
        parser.exit(1, f"{failures}/{len(rows)} buildings failed verification.\n")
    print(f"PASS: {len(rows)}/{len(rows)} buildings have matching GLB, Three.js geometry, and metadata.")
    print(f"Watertight: {watertight}/{len(rows)} | Unknown floor count: {sum(row['floor_count'] is None for row in rows)}")


if __name__ == "__main__":
    main()
