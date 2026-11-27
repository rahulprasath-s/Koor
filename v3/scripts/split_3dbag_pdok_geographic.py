#!/usr/bin/env python3
"""Create west-to-east train, validation, and test manifests with geographic gaps."""

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


SPLITS = ("train", "validation", "test")
FRACTIONS = (0.8, 0.9)


def easting_interval(row):
    aerial = row["aerial"]
    west, south, east, north = aerial["bbox_wsen_m"]
    if (aerial["crs"] != "EPSG:28992" or not all(map(math.isfinite, (west, south, east, north)))
            or not (west < east and south < north)):
        raise ValueError(f"Invalid Dutch RD crop bounds for {row['building_id']}")
    return west, east


def geographic_split(rows, gap):
    intervals = [(easting_interval(row), row) for row in rows]
    centers = sorted((west + east) / 2 for (west, east), _ in intervals)
    cuts = [centers[min(len(centers) - 1, int(fraction * len(centers)))] for fraction in FRACTIONS]
    left, right = cuts
    half_gap = gap / 2
    groups = {name: [] for name in (*SPLITS, "excluded_buffer")}
    for (west, east), row in intervals:
        if east <= left - half_gap:
            name = "train"
        elif west >= left + half_gap and east <= right - half_gap:
            name = "validation"
        elif west >= right + half_gap:
            name = "test"
        else:
            name = "excluded_buffer"
        groups[name].append(row)
    if any(not groups[name] for name in SPLITS):
        raise ValueError("At least one split is empty. Choose a smaller gap or a wider study area.")

    bounds = {name: [min(easting_interval(row)[0] for row in groups[name]),
                     max(easting_interval(row)[1] for row in groups[name])] for name in SPLITS}
    measured_gaps = {
        f"{a}_to_{b}": bounds[b][0] - bounds[a][1]
        for a, b in zip(SPLITS, SPLITS[1:])
    }
    if any(value < gap - 1e-7 for value in measured_gaps.values()):
        raise ValueError("Aerial crops do not meet the requested geographic separation")
    return groups, cuts, bounds, measured_gaps


def split_shard(shard, output, gap):
    if output.exists() or output.is_symlink():
        raise ValueError("Choose a new split output directory")
    source = shard / "examples.jsonl"
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows or len({row["building_id"] for row in rows}) != len(rows):
        raise ValueError("The shard manifest must contain unique building records")
    groups, cuts, bounds, measured_gaps = geographic_split(rows, gap)
    counts = {name: len(group) for name, group in groups.items()}
    summary = {
        "source_manifest": os.path.relpath(source, output),
        "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "asset_paths_relative_to": os.path.relpath(shard, output),
        "crs": "EPSG:28992",
        "axis": "easting",
        "nominal_fractions": {"train": 0.80, "validation": 0.10, "test": 0.10},
        "cut_eastings_m": cuts,
        "required_gap_m": gap,
        "crop_easting_bounds_m": bounds,
        "measured_crop_gaps_m": measured_gaps,
        "counts": counts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".geo-splits-") as temp:
        staged = Path(temp) / "splits"
        staged.mkdir()
        for name, group in groups.items():
            with (staged / f"{name}.jsonl").open("w", encoding="utf-8") as file:
                for row in group:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
        (staged / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        staged.rename(output)
    for name, count in counts.items():
        print(f"{name}: {count}")
    print("Minimum gaps between crop extents:", ", ".join(f"{key}={value:.1f} m" for key, value in measured_gaps.items()))
    print(f"Saved geographic split manifests in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--output", type=Path, help="Defaults to SHARD/splits-geographic")
    parser.add_argument("--gap-m", type=float, default=50.0, help="Minimum distance between split crop extents")
    args = parser.parse_args()
    if not math.isfinite(args.gap_m) or args.gap_m <= 0:
        parser.error("--gap-m must be positive and finite")
    shard = args.shard.resolve()
    output = args.output.resolve() if args.output else shard / "splits-geographic"
    try:
        split_shard(shard, output, args.gap_m)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"Split failed: {error}\n")


if __name__ == "__main__":
    main()
