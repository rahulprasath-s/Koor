"""Validate synthetic samples and prepare reproducible dataset splits."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "data/RoofReconstructionDataset/SyntheticDataset"
OUTPUT = PROJECT / "data/prepared"
SEED = 42


def inspect_sample(folder):
    points = np.loadtxt(folder / "points.xyz", ndmin=2)
    if points.shape[1] != 3 or len(points) < 3:
        raise ValueError(f"Expected XYZ points: {folder}")
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite points: {folder}")

    vertices = []
    for line in (folder / "polygon.obj").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "v":
            vertices.append([float(value) for value in fields[1:4]])

    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError(f"Invalid mesh vertices: {folder}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"Non-finite mesh vertices: {folder}")

    # Group matching vertex layouts, allowing translation and uniform scaling.
    # This is conservative: connectivity is deliberately ignored.
    scale = np.ptp(vertices, axis=0).max()
    if scale <= 0:
        raise ValueError(f"Degenerate geometry: {folder}")

    normalized = (vertices - vertices.mean(axis=0)) / scale
    normalized = np.round(normalized, decimals=6)
    normalized[normalized == 0] = 0  # Normalize negative zero.
    order = np.lexsort(normalized.T[::-1])
    canonical = normalized[order].astype("<f8").tobytes()
    group = hashlib.sha256(canonical).hexdigest()

    return {
        "path": str(folder.relative_to(PROJECT)),
        "geometry_group": group,
        "point_count": len(points),
        "vertex_count": len(vertices),
    }


def load_subset(list_name, extracted_name):
    extracted = SOURCE / extracted_name
    index = {}

    for points_file in sorted(extracted.rglob("points.xyz")):
        folder = points_file.parent
        if folder.name in index:
            raise ValueError(f"Ambiguous sample ID: {folder.name}")
        index[folder.name] = folder

    entries = [
        line.strip()
        for line in (SOURCE / list_name).read_text().splitlines()
        if line.strip()
    ]
    identifiers = [Path(entry).name for entry in entries]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Duplicate entries in {list_name}")

    rows = []
    for number, identifier in enumerate(identifiers, 1):
        if identifier not in index:
            raise FileNotFoundError(f"Missing {extracted_name}/{identifier}")
        row = inspect_sample(index[identifier])
        row["sample_id"] = f"{extracted_name}/{identifier}"
        rows.append(row)
        if number % 2000 == 0:
            print(f"Validated {number}/{len(identifiers)}: {extracted_name}",
                  flush=True)
    return rows


def main():
    source_train = load_subset("train.txt", "simulateRoof")
    test = load_subset("test.txt", "simulateRoofTest")

    # Keep official test samples; exclude matching geometry from development.
    test_groups = {row["geometry_group"] for row in test}
    development = [
        row for row in source_train
        if row["geometry_group"] not in test_groups
    ]

    groups = sorted({row["geometry_group"] for row in development})
    if len(groups) < 2:
        raise ValueError("Insufficient independent geometry groups")

    random.Random(SEED).shuffle(groups)
    validation_count = min(len(groups) - 1, max(1, round(len(groups) * 0.1)))
    validation_groups = set(groups[:validation_count])

    for row in development:
        row["split"] = (
            "val" if row["geometry_group"] in validation_groups else "train"
        )
    for row in test:
        row["split"] = "test"

    rows = development + test
    split_groups = {
        split: {
            row["geometry_group"] for row in rows if row["split"] == split
        }
        for split in ("train", "val", "test")
    }
    assert split_groups["train"].isdisjoint(split_groups["val"])
    assert split_groups["train"].isdisjoint(split_groups["test"])
    assert split_groups["val"].isdisjoint(split_groups["test"])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": SEED,
        "counts": dict(Counter(row["split"] for row in rows)),
        "excluded_train_samples_matching_test_geometry": (
            len(source_train) - len(development)
        ),
        "grouping": "Translation/scale-normalized vertices rounded to 6 decimals",
        "samples": rows,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    # Absolute-path lists work with the upstream loader.
    # Regenerate these lists if you move the project.
    for split in ("train", "val", "test"):
        paths = [
            str(PROJECT / row["path"])
            for row in rows if row["split"] == split
        ]
        (OUTPUT / f"{split}.txt").write_text("\n".join(paths) + "\n")

    print("\nSplit counts:", manifest["counts"])
    print("Excluded training samples matching test geometry:",
          manifest["excluded_train_samples_matching_test_geometry"])
    print("PASS: geometry groups are isolated across splits")
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()