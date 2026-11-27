#!/usr/bin/env python3
"""Verify every aerial image, footprint mask, and 3D export in a shard."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from collect_3dbag_pdok import BUILDING_ID, MATRIX_SET, ORIGIN_X, ORIGIN_Y, RESOLUTION, TILE, WMTS, read_tile
from verify_3dbag_buildings import require, verify_building


def verify_mask(mask, row, bounds):
    west, south, east, north = bounds
    width, height = mask.size
    expected = np.zeros((height, width), dtype=bool)
    for polygon in row["footprint_polygons_xy"]:
        part = Image.new("L", mask.size)
        draw = ImageDraw.Draw(part)
        for index, ring in enumerate(polygon):
            require(len(ring) >= 3, "Footprint ring has fewer than three points")
            points = [((x - west) * width / (east - west), (north - y) * height / (north - south))
                      for x, y in ring]
            require(all(0 < x < width and 0 < y < height for x, y in points), "Footprint crosses the crop boundary")
            draw.polygon(points, fill=255 if index == 0 else 0)
        expected |= np.asarray(part) != 0

    pixels = np.asarray(mask)
    require(np.all((pixels == 0) | (pixels == 255)), "Footprint mask contains non-binary values")
    actual = pixels == 255
    require(actual.any() and not actual.all() and expected.any(), "Footprint mask is empty or fills the image")
    reference = Image.fromarray(expected.astype(np.uint8) * 255)
    # Tolerate one pixel at the polygon boundary for coordinate rounding.
    outer = np.asarray(reference.filter(ImageFilter.MaxFilter(3))) > 0
    inner = np.asarray(reference.filter(ImageFilter.MinFilter(3))) > 0
    require(not np.any(actual & ~outer) and not np.any(inner & ~actual),
            "Footprint mask differs from the mapped footprint by more than one pixel")


def verify_pair(root, row, cache):
    building_id = row["building_id"]
    require(BUILDING_ID.fullmatch(building_id), "Invalid building ID")
    relative = Path("buildings") / building_id
    require(row["aerial_path"] == (relative / "aerial.jpg").as_posix() and
            row["footprint_mask_path"] == (relative / "footprint.png").as_posix(), "Incorrect image or mask path")
    saved = json.loads((root / relative / "example.json").read_text(encoding="utf-8"))
    require(saved == row, "Building example.json differs from examples.jsonl")
    pair_fields = {"aerial_path", "footprint_mask_path", "pairing_method", "quality_flags", "aerial"}
    require(verify_building(root, {key: value for key, value in row.items() if key not in pair_fields}),
            "Mesh is not watertight")

    aerial = row["aerial"]
    width, height = aerial["image_size_px"]
    require(all(type(size) is int and 0 < size <= 2048 for size in (width, height)), "Invalid image dimensions")
    west, south, east, north = bounds = aerial["bbox_wsen_m"]
    require(aerial["crs"] == MATRIX_SET and aerial["meters_per_pixel"] == RESOLUTION and
            math.isclose(east - west, width * RESOLUTION, abs_tol=1e-7) and
            math.isclose(north - south, height * RESOLUTION, abs_tol=1e-7), "Invalid aerial coordinate metadata")
    left = round((west - ORIGIN_X) / RESOLUTION)
    top = round((ORIGIN_Y - north) / RESOLUTION)
    require(math.isclose(west, ORIGIN_X + left * RESOLUTION, abs_tol=1e-7) and
            math.isclose(north, ORIGIN_Y - top * RESOLUTION, abs_tol=1e-7), "Crop is off the PDOK pixel grid")
    tiles = [(col, line)
             for line in range(top // TILE, (top + height - 1) // TILE + 1)
             for col in range(left // TILE, (left + width - 1) // TILE + 1)]
    require(aerial["tile_urls"] == [f"{WMTS}/{col}/{line}.jpeg" for col, line in tiles],
            "Tile provenance does not cover the recorded crop")

    with Image.open(root / row["aerial_path"]) as photo, Image.open(root / row["footprint_mask_path"]) as mask:
        require(photo.format == "JPEG" and photo.mode == "RGB" and photo.size == (width, height),
                "Unexpected aerial photo format or dimensions")
        require(mask.format == "PNG" and mask.mode == "L" and mask.size == photo.size,
                "Unexpected footprint mask format or dimensions")
        require(any(low != high for low, high in photo.getextrema()), "Aerial photo is blank")
        verify_mask(mask, row, bounds)
        if cache is not None:
            source = Image.new("RGB", photo.size)
            for col, line in tiles:
                tile = read_tile((cache / f"{col}-{line}.jpg").read_bytes())
                source.paste(tile, (col * TILE - left, line * TILE - top))
            error = np.abs(np.asarray(source, dtype=np.int16) - np.asarray(photo, dtype=np.int16)).mean()
            require(error <= 6, f"Photo differs from cached PDOK tiles (mean channel error {error:.2f}/255)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--count", type=int, default=1000, help="Expected number of unique pairs")
    args = parser.parse_args()
    try:
        with (args.shard / "examples.jsonl").open(encoding="utf-8") as file:
            rows = [json.loads(line) for line in file if line.strip()]
        require(len(rows) == args.count and args.count > 0, f"Expected {args.count} pairs, found {len(rows)}")
        require(len({row["building_id"] for row in rows}) == len(rows), "Duplicate building ID")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"FAIL: {error}\n")

    cache = args.shard / ".pdok-tile-cache"
    cache = cache if cache.is_dir() else None
    failed = 0
    for index, row in enumerate(rows, start=1):
        try:
            verify_pair(args.shard, row, cache)
        except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
            failed += 1
            print(f"FAIL {row['building_id']}: {error}", flush=True)
        if index % 100 == 0:
            print(f"Checked {index}/{len(rows)} pairs", flush=True)
    if failed:
        parser.exit(1, f"{failed}/{len(rows)} pairs failed verification.\n")
    print(f"PASS: {len(rows)}/{len(rows)} aerial photos, footprint masks, and GLB/Three.js pairs verified.")
    print("Aerial pixels were checked against the cached PDOK tiles." if cache is not None else
          "Aerial files passed format, dimensions and tile coverage checks. Source tile cache unavailable.")


if __name__ == "__main__":
    main()
