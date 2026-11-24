"""Prepare Building3D Entry-level XYZ/wireframe pairs for Koor on Alex.

Save as ~/Koor/scripts/prepare_building3d.py. Uses NumPy and the standard library.
Splits use complete building bounds, geographic bands and exclusion buffers.
This creates a NEW split manifest; do not assume it reproduces SECE splits.
Official unlabelled test clouds are inventoried but never used for training.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_wireframe(path):
    records = [line.split("#")[0].split() for line in path.read_text().splitlines()]
    vertices = np.asarray([[float(v) for v in fields[1:4]]
                           for fields in records if fields and fields[0] == "v"], dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"Invalid OBJ vertices: {path}")
    seen, edges = 0, set()
    for fields in records:
        if not fields:
            continue
        if fields[0] == "v":
            seen += 1
        elif fields[0] in {"l", "f"}:
            indices = []
            for token in fields[1:]:
                index = int(token.split("/")[0])
                if index == 0:
                    raise ValueError(f"Zero OBJ index: {path}")
                index = index - 1 if index > 0 else seen + index
                if not 0 <= index < len(vertices):
                    raise ValueError(f"OBJ index out of range: {path}")
                indices.append(index)
            if len(indices) < 2:
                raise ValueError(f"Incomplete OBJ edge: {path}")
            pairs = list(zip(indices, indices[1:]))
            if fields[0] == "f" and len(indices) > 2:
                pairs.append((indices[-1], indices[0]))
            edges.update(tuple(sorted((a, b))) for a, b in pairs if a != b)
    if not edges:
        raise ValueError(f"No nondegenerate edges: {path}")
    return vertices, sorted(edges)


def geometry_group(vertices):
    """Translation/scale invariant, rounded labelled-vertex fingerprint.

    Conservative grouping ignores edges, and does not identify rotated roofs
    or every near duplicate. It is not a complete duplicate-detection method.
    """
    lower, upper = vertices.min(0), vertices.max(0)
    scale = float((upper - lower).max())
    if scale <= 0:
        raise ValueError("Degenerate labelled geometry")
    normalized = np.round((vertices - (lower + upper) / 2) / scale, 6)
    normalized[normalized == 0] = 0  # Canonicalize negative zero.
    ordered = normalized[np.lexsort((normalized[:, 2], normalized[:, 1], normalized[:, 0]))]
    return hashlib.sha256(ordered.astype("<f8").tobytes()).hexdigest()


def assign_spatial_splits(records, axis, cuts, buffer):
    for record in records:
        lower, upper = record["bounds_lower"][axis], record["bounds_upper"][axis]
        if upper < cuts[0] - buffer:
            split = "train"
        elif lower > cuts[0] + buffer and upper < cuts[1] - buffer:
            split = "val"
        elif lower > cuts[1] + buffer:
            split = "holdout"
        else:
            split = "excluded_buffer"
        record["split"] = split
    groups = defaultdict(list)
    for record in records:
        if record["split"] in {"train", "val", "holdout"}:
            groups[record["geometry_group"]].append(record)
    for members in groups.values():
        if len({record["split"] for record in members}) > 1:
            for record in members:
                record["spatial_candidate"] = record["split"]
                record["split"] = "excluded_cross_split_geometry"


def verify_splits(records, axis, cuts, buffer):
    groups = {}
    for name in ("train", "val", "holdout"):
        members = [record for record in records if record["split"] == name]
        if not members:
            raise ValueError(f"Empty {name} split after exclusions; inspect bounds/buffer before changing the protocol")
        for record in members:
            lower, upper = record["bounds_lower"][axis], record["bounds_upper"][axis]
            valid = {
                "train": upper < cuts[0] - buffer,
                "val": lower > cuts[0] + buffer and upper < cuts[1] - buffer,
                "holdout": lower > cuts[1] + buffer,
            }[name]
            if not valid:
                raise AssertionError("A building crosses its spatial split boundary")
        groups[name] = {record["geometry_group"] for record in members}
    for a, b in (("train", "val"), ("train", "holdout"), ("val", "holdout")):
        if groups[a] & groups[b]:
            raise AssertionError("Labelled geometry fingerprint crosses splits")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=PROJECT / "data/building3d/raw/Entry-level")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/building3d/koor-v2")
    parser.add_argument("--axis", choices=("X", "Y"), default="Y")
    parser.add_argument("--buffer", type=float, default=50.0,
                        help="Excluded distance on EACH side of a boundary, in source coordinate units")
    parser.add_argument("--expected-pairs", type=int, default=5698)
    parser.add_argument("--expected-official-test", type=int, default=583)
    args = parser.parse_args()
    if not np.isfinite(args.buffer) or args.buffer < 0:
        parser.error("--buffer must be finite and nonnegative")
    raw, output = args.raw.expanduser().resolve(), args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    points = {path.stem: path for path in sorted((raw / "train/xyz").glob("*.xyz"))}
    labels = {path.stem: path for path in sorted((raw / "train/wireframe").glob("*.obj"))}
    if len(points) != args.expected_pairs or set(points) != set(labels):
        raise ValueError(f"Expected {args.expected_pairs} labelled pairs; found {len(points)} clouds, {len(labels)} labels; missing labels={sorted(set(points)-set(labels))[:5]}, missing clouds={sorted(set(labels)-set(points))[:5]}")
    official_test = sorted((raw / "test/xyz").glob("*.xyz"))
    if len(official_test) != args.expected_official_test:
        raise ValueError(f"Expected {args.expected_official_test} official test clouds; found {len(official_test)}")
    if any((raw / "test").rglob("*.obj")):
        raise ValueError("Unexpected official test labels; inspect this dataset version")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    records = []
    try:
        for index, sample_id in enumerate(sorted(points), 1):
            cloud = np.loadtxt(points[sample_id], dtype=np.float64, usecols=(0, 1, 2), ndmin=2)
            if not len(cloud) or not np.isfinite(cloud).all():
                raise ValueError(f"Invalid points: {points[sample_id]}")
            vertices, edges = read_wireframe(labels[sample_id])
            cloud_lower, cloud_upper = cloud.min(0), cloud.max(0)
            scale = float((cloud_upper - cloud_lower).max())
            if scale <= 0:
                raise ValueError(f"Degenerate cloud: {sample_id}")
            origin = (cloud_lower + cloud_upper) / 2
            destination = staging / "samples" / sample_id
            destination.mkdir(parents=True)
            # Center float64 coordinates before storage/float32 model loading.
            # Do NOT rescale: the Koor dataset loader performs normalization.
            np.savetxt(destination / "points.xyz", cloud - origin, fmt="%.12f")
            with (destination / "polygon.obj").open("w") as file:
                for vertex in vertices - origin:
                    file.write("v " + " ".join(f"{value:.12f}" for value in vertex) + "\n")
                for a, b in edges:
                    file.write(f"l {a + 1} {b + 1}\n")
            record = {
                "id": sample_id, "sample_dir": f"samples/{sample_id}",
                "source_points": str(points[sample_id].relative_to(raw)),
                "source_wireframe": str(labels[sample_id].relative_to(raw)),
                "source_origin": origin.tolist(), "point_count": len(cloud),
                "vertex_count": len(vertices), "edge_count": len(edges),
                "bounds_lower": np.minimum(cloud_lower, vertices.min(0)).tolist(),
                "bounds_upper": np.maximum(cloud_upper, vertices.max(0)).tolist(),
                "geometry_group": geometry_group(vertices),
            }
            records.append(record)
            if index % 500 == 0 or index == len(points):
                print(f"Converted {index}/{len(points)}", flush=True)

        axis = 0 if args.axis == "X" else 1
        centers = [(record["bounds_lower"][axis] + record["bounds_upper"][axis]) / 2 for record in records]
        cuts = np.quantile(centers, [0.8, 0.9]).tolist()
        assign_spatial_splits(records, axis, cuts, args.buffer)
        verify_splits(records, axis, cuts, args.buffer)
        counts = dict(Counter(record["split"] for record in records))
        for name in ("train", "val", "holdout"):
            paths = [str(output / record["sample_dir"]) for record in records if record["split"] == name]
            if len(paths) != len(set(paths)):
                raise AssertionError("Duplicate sample paths")
            (staging / f"{name}.txt").write_text("\n".join(paths) + "\n")
        (staging / "excluded.txt").write_text("".join(
            f"{record['id']}\t{record['split']}\n" for record in records if record["split"].startswith("excluded")))
        archive = raw.parent.parent / "Entry_level.zip"
        manifest = {
            "schema": "koor-building3d-spatial-v1", "dataset": "Building3D Entry-level",
            "license": "CC-BY-NC-SA-4.0", "source_crs": None,
            "source_units": "Unverified; buffer distances use source coordinate units",
            "source_archive_sha256": sha256(archive) if archive.is_file() else None,
            "preparation_script_sha256": sha256(Path(__file__)),
            "conversion": "XYZ only; float64 shared bbox-center subtraction; 12-decimal storage; no rescaling; OBJ edges canonicalized",
            "partition": {
                "axis": args.axis, "center_quantiles": [0.8, 0.9], "cuts": cuts,
                "buffer_each_side": args.buffer,
                "bounds": "Union of full point-cloud and labelled-vertex bounds",
                "geometry_group": "SHA256 of sorted float64 labelled vertices, centered/scaled by labelled bbox, rounded to 6 decimals; no rotation invariance",
                "cross_split_group_policy": "Exclude every assigned member of a fingerprint group spanning multiple splits",
                "note": "New Alex manifest; no claim of identical splits to earlier SECE experiments",
            },
            "counts": counts, "labelled_pairs": len(records),
            "official_test": {"count": len(official_test), "labelled": False,
                              "used_for_training_or_selection": False,
                              "source_paths": [str(path.relative_to(raw)) for path in official_test]},
            "samples": records,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print("\nSplit axis:", args.axis, "| buffer each side:", args.buffer)
    print("Counts:", counts)
    print("Point-count range:", min(r["point_count"] for r in records), max(r["point_count"] for r in records))
    print("PASS: paired samples converted; full bounds separated; geometry fingerprints isolated")
    print("Official unlabelled test clouds kept separate:", len(official_test))
    print("Saved:", output)


if __name__ == "__main__":
    main()
