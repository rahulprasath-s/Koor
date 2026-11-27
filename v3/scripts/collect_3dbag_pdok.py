#!/usr/bin/env python3
"""Collect complete 3DBAG geometry and PDOK aerial pairs without manual review."""

import argparse
import io
import json
import math
import re
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from PIL import Image, ImageChops, ImageDraw, UnidentifiedImageError

from collect_3dbag_buildings import COLLECTION, RETRYABLE, fetch_page, write_jsonl
from export_3dbag_building import export_building


DEFAULT_BBOX = (88000, 395000, 94000, 401000)
LAYER = "2024_orthoHR"
MATRIX_SET = "EPSG:28992"
MATRIX = 15
TILE = 256
RESOLUTION = 0.105  # 375 × 0.00028 metres, per PDOK WMTS capabilities.
ORIGIN_X, ORIGIN_Y = -285401.92, 903401.92
MATRIX_WIDTH = 32768
WMTS = f"https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/{LAYER}/{MATRIX_SET}/{MATRIX}"
BUILDING_ID = re.compile(r"NL\.IMBAG\.Pand\.[0-9]{16}")
PAIR_FILES = {"building.glb", "building.mjs", "metadata.json", "aerial.jpg", "footprint.png", "example.json"}


def read_tile(data):
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG" or image.size != (TILE, TILE):
            raise ValueError("Unexpected PDOK tile format or size")
        return image.convert("RGB")


def fetch_tile(cache, col, row):
    if not (0 <= col < MATRIX_WIDTH and 0 <= row < MATRIX_WIDTH):
        raise ValueError("Crop falls outside the PDOK tile grid")
    url = f"{WMTS}/{col}/{row}.jpeg"
    path = cache / f"{col}-{row}.jpg"
    if path.is_file():
        try:
            return read_tile(path.read_bytes()), url
        except (OSError, ValueError):
            pass  # Replace a damaged cache entry after a successful refetch.
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "3DBAG-PDOK-PairCollector/1.0"})
            with urlopen(request, timeout=30) as response:
                data = response.read(2_000_001)
            if len(data) > 2_000_000:
                raise ValueError("PDOK tile exceeds 2 MB")
            image = read_tile(data)
        except (URLError, TimeoutError) as error:
            if attempt == 2 or (isinstance(error, HTTPError) and error.code not in RETRYABLE):
                raise
            wait = error.headers.get("Retry-After", "") if isinstance(error, HTTPError) else ""
            time.sleep(max(2 ** attempt, int(wait)) if wait.isdecimal() else 2 ** attempt)
        else:
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as file:
                file.write(data)
            Path(file.name).replace(path)
            time.sleep(0.10)  # One request at a time, including a pause on cache misses.
            return image, url


def save_aerial(metadata, cache, entry):
    if metadata["source_crs"].rstrip("/").rsplit("/", 1)[-1] != "7415":
        raise ValueError("Expected 3DBAG's Dutch RD horizontal coordinates")
    polygons = [
        [[((x - ORIGIN_X) / RESOLUTION, (ORIGIN_Y - y) / RESOLUTION) for x, y in ring]
         for ring in polygon] for polygon in metadata["footprint_polygons_xy"]
    ]
    points = [point for polygon in polygons for ring in polygon for point in ring]
    if not points or not all(math.isfinite(value) for point in points for value in point):
        raise ValueError("Missing or invalid building footprint")
    x, y = zip(*points)
    span = max(max(x) - min(x), max(y) - min(y))
    size = max(TILE, math.ceil(span + 12 / RESOLUTION) + 2)
    if size > 2048:
        raise ValueError("Footprint exceeds the 2048-pixel crop limit")
    left = math.floor((min(x) + max(x) - size) / 2)
    top = math.floor((min(y) + max(y) - size) / 2)
    right, bottom = left + size, top + size
    photo = Image.new("RGB", (size, size))
    urls = []
    for row in range(top // TILE, (bottom - 1) // TILE + 1):
        for col in range(left // TILE, (right - 1) // TILE + 1):
            image, url = fetch_tile(cache, col, row)
            photo.paste(image, (col * TILE - left, row * TILE - top))
            urls.append(url)
    if all(low == high for low, high in photo.getextrema()):
        raise ValueError("PDOK returned a blank crop")

    mask = Image.new("1", photo.size)
    for polygon in polygons:
        part = Image.new("1", photo.size)
        draw = ImageDraw.Draw(part)
        for index, ring in enumerate(polygon):
            draw.polygon([(px - left, py - top) for px, py in ring], fill=1 if index == 0 else 0)
        mask = ImageChops.logical_or(mask, part)
    if mask.getbbox() is None:
        raise ValueError("Empty footprint mask")
    photo.save(entry / "aerial.jpg", quality=94)
    mask.convert("L").save(entry / "footprint.png")
    return {
        "year": 2024,
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Samenwerkingsverband Beeldmateriaal via PDOK",
        "license_source": "https://www.pdok.nl/-/nu-hoge-resolutie-luchtfoto-2024-bij-pdok",
        "tile_urls": urls,
        "changes": "Cropped and recompressed as JPEG from PDOK aerial imagery tiles.",
        "crs": MATRIX_SET,
        "meters_per_pixel": RESOLUTION,
        "image_size_px": [size, size],
        "bbox_wsen_m": [ORIGIN_X + left * RESOLUTION, ORIGIN_Y - bottom * RESOLUTION,
                       ORIGIN_X + right * RESOLUTION, ORIGIN_Y - top * RESOLUTION],
    }


def build_example(root, cache, building_id, feature, header):
    relative = Path("buildings") / building_id
    with tempfile.TemporaryDirectory(dir=root, prefix=".building-") as tmp:
        entry = Path(tmp) / "entry"
        geometry = export_building({"feature": feature, "metadata": header}, building_id,
                                   f"{COLLECTION}/{building_id}", entry)
        if not geometry["mesh_watertight"]:
            raise ValueError("Exported mesh is not watertight")
        aerial = save_aerial(geometry, cache, entry)
        example = {
            **geometry,
            "mesh_path": (relative / "building.glb").as_posix(),
            "threejs_path": (relative / "building.mjs").as_posix(),
            "aerial_path": (relative / "aerial.jpg").as_posix(),
            "footprint_mask_path": (relative / "footprint.png").as_posix(),
            "metadata_path": (relative / "metadata.json").as_posix(),
            "pairing_method": "crop centered on the georeferenced 3DBAG footprint",
            "quality_flags": [flag for flag, applies in (
                ("multiple_footprint_polygons", len(geometry["footprint_polygons_xy"]) > 1),
                ("unknown_floor_count", geometry["floor_count"] is None),
            ) if applies],
            "aerial": aerial,
        }
        (entry / "example.json").write_text(json.dumps(example, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        entry.rename(root / relative)
    return example


def collect(root, bbox, count):
    settings = {"bbox_rd_m": list(bbox), "image_layer": LAYER, "wmts_matrix": MATRIX, "geometry_lod": "2.2"}
    root.mkdir(parents=True, exist_ok=True)
    settings_file = root / "collection.json"
    if settings_file.exists():
        if json.loads(settings_file.read_text(encoding="utf-8")) != settings:
            raise ValueError("Existing output uses different collection settings")
    else:
        if any(root.iterdir()):
            raise ValueError("Choose a new output directory or restart an existing shard")
        settings_file.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    buildings = root / "buildings"
    cache = root / ".pdok-tile-cache"
    buildings.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    rows = []
    for entry in sorted(buildings.iterdir()):
        if not entry.is_dir() or not BUILDING_ID.fullmatch(entry.name):
            continue
        if not all((entry / name).is_file() for name in PAIR_FILES):
            raise ValueError(f"Incomplete existing pair at {entry}")
        example = json.loads((entry / "example.json").read_text(encoding="utf-8"))
        if example["building_id"] != entry.name:
            raise ValueError(f"Incorrect building ID at {entry}")
        rows.append(example)
    if len(rows) > count:
        raise ValueError(f"Output already contains {len(rows)} pairs. Increase --count or use a new directory.")

    seen = {row["building_id"] for row in rows}
    failures = []
    visited_pages = set()
    url = f"{COLLECTION}?" + urlencode({"bbox": ",".join(map(str, bbox)), "limit": 10})
    if rows:
        print(f"Reusing {len(rows)} complete pairs", flush=True)
    try:
        while url and len(rows) < count:
            if url in visited_pages:
                raise ValueError("3DBAG returned a repeated pagination link")
            visited_pages.add(url)
            page = fetch_page(url)
            for feature in page["features"]:
                building_id = feature["id"]
                if not BUILDING_ID.fullmatch(building_id) or building_id in seen:
                    continue
                seen.add(building_id)
                try:
                    example = build_example(root, cache, building_id, feature, page["metadata"])
                except HTTPError as error:
                    if error.code in RETRYABLE or error.code in {401, 403}:
                        raise  # Stop and preserve completed data when the service is unavailable.
                    if error.code != 404:
                        raise
                    failures.append({"building_id": building_id, "reason": "Missing PDOK tile (HTTP 404)"})
                    continue
                except UnidentifiedImageError as error:
                    failures.append({"building_id": building_id, "reason": f"Invalid aerial image: {error}"})
                    continue
                except (ValueError, IndexError, KeyError) as error:
                    failures.append({"building_id": building_id, "reason": str(error)})
                    continue
                rows.append(example)
                if count <= 10 or len(rows) % 25 == 0 or len(rows) == count:
                    print(f"{len(rows)}/{count} complete pairs", flush=True)
                if len(rows) == count:
                    break
            next_url = next((link["href"] for link in page["links"] if link["rel"] == "next"), None)
            url = urljoin(url, next_url) if next_url else None
            if url and len(rows) < count:
                time.sleep(0.5)
    finally:
        write_jsonl(root / "examples.jsonl", rows)
        if failures or not (root / "skipped.jsonl").exists():
            write_jsonl(root / "skipped.jsonl", failures)
    print(f"Saved {len(rows)} complete pairs to {root / 'examples.jsonl'}; skipped {len(failures)} this run.")
    if len(rows) != count:
        raise ValueError(f"This bounding box yielded {len(rows)} of {count} requested complete pairs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX,
                        metavar=("WEST", "SOUTH", "EAST", "NORTH"), help="Dutch RD metre coordinates")
    args = parser.parse_args()
    w, s, e, n = args.bbox
    if args.count < 1 or not all(map(math.isfinite, args.bbox)) or not (w < e and s < n):
        parser.error("Use a positive count and an ordered, finite bounding box")
    try:
        collect(args.output.resolve(), args.bbox, args.count)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Collection stopped: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Collection interrupted. Rerun the same command to continue.\n")


if __name__ == "__main__":
    main()
