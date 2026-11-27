#!/usr/bin/env python3
"""Export a batch of 3DBAG buildings from a projected bounding box."""

import argparse
import json
import math
import re
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import Request, urlopen

from export_3dbag_building import LOD, export_building


COLLECTION = "https://api.3dbag.nl/collections/pand/items"
RETRYABLE = {429, 500, 502, 503, 504}


def fetch_page(url):
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != "api.3dbag.nl":
        raise ValueError("Unexpected 3DBAG pagination URL")
    request = Request(url, headers={"Accept": "application/json"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=45) as response:
                return json.load(response)
        except (URLError, TimeoutError) as error:
            if attempt == 2 or (isinstance(error, HTTPError) and error.code not in RETRYABLE):
                raise
            wait = error.headers.get("Retry-After", "") if isinstance(error, HTTPError) else ""
            time.sleep(max(2 ** (attempt + 1), int(wait)) if wait.isdecimal() else 2 ** (attempt + 1))


def write_jsonl(path, rows):
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".index-") as temp:
        staged = Path(temp) / path.name
        with staged.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        staged.replace(path)


def manifest_row(metadata, relative_dir):
    return {
        **metadata,
        "mesh_path": (relative_dir / "building.glb").as_posix(),
        "threejs_path": (relative_dir / "building.mjs").as_posix(),
        "metadata_path": (relative_dir / "metadata.json").as_posix(),
    }


def existing_exports(root):
    buildings = root / "buildings"
    if not buildings.exists():
        return []
    rows = []
    for directory in sorted(buildings.iterdir()):
        if not directory.is_dir() or not re.fullmatch(r"NL\.IMBAG\.Pand\.[0-9]{16}", directory.name):
            continue
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("building_id") != directory.name or any(
            not (directory / filename).is_file() for filename in ("building.glb", "building.mjs")
        ):
            raise ValueError(f"Incomplete existing export: {directory}")
        rows.append(manifest_row(metadata, directory.relative_to(root)))
    return rows


def collect(root, bbox, count, page_size):
    root.mkdir(parents=True, exist_ok=True)
    settings = {"bbox": bbox, "lod": LOD, "api": COLLECTION}
    settings_path = root / "collection.json"
    if settings_path.exists():
        previous = json.loads(settings_path.read_text(encoding="utf-8"))
        if previous.get("lod") != LOD or previous.get("api") != COLLECTION:
            raise ValueError("Existing collection uses another source or detail level")
        west, south, east, north = previous["bbox"]
        if not (bbox[0] <= west and bbox[1] <= south and bbox[2] >= east and bbox[3] >= north):
            raise ValueError("To extend this collection, use a bounding box that contains the previous one")
    else:
        if any(root.iterdir()):
            raise ValueError("Choose an empty directory or an existing collection created by this script.")
        previous = None

    rows = existing_exports(root)
    if len(rows) > count:
        raise ValueError(f"This collection already has {len(rows)} exports. Set --count to at least that number.")
    if rows:
        print(f"Reusing {len(rows)} existing exports", flush=True)
    if previous != settings:
        with tempfile.TemporaryDirectory(dir=root, prefix=".settings-") as temp:
            staged = Path(temp) / settings_path.name
            staged.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
            staged.replace(settings_path)

    url = f"{COLLECTION}?" + urlencode({"bbox": ",".join(map(str, bbox)), "limit": page_size})
    seen_pages = set()
    seen_ids = {row["building_id"] for row in rows}
    try:
        while url and len(rows) < count:
            if url in seen_pages:
                raise ValueError("The API returned a repeated pagination link")
            seen_pages.add(url)
            page = fetch_page(url)
            for feature in page["features"]:
                building_id = feature["id"]
                if not re.fullmatch(r"NL\.IMBAG\.Pand\.[0-9]{16}", building_id):
                    continue
                if building_id in seen_ids:
                    continue
                seen_ids.add(building_id)
                relative_dir = Path("buildings") / building_id
                destination = root / relative_dir
                try:
                    metadata = export_building(
                        {"feature": feature, "metadata": page["metadata"]},
                        building_id,
                        f"{COLLECTION}/{building_id}",
                        destination,
                    )
                except (KeyError, ValueError, IndexError) as error:
                    print(f"Skipped {building_id}: {error}", flush=True)
                    continue
                rows.append(manifest_row(metadata, relative_dir))
                print(f"{len(rows)}/{count}  {building_id}", flush=True)
                if len(rows) == count:
                    break
            next_url = next((link["href"] for link in page["links"] if link["rel"] == "next"), None)
            url = urljoin(url, next_url) if next_url else None
            if url and len(rows) < count:
                time.sleep(0.5)
    except BaseException:
        # Keep a previous index on failed reruns. Finished exports can be reused.
        if rows and not (root / "buildings.jsonl").exists():
            write_jsonl(root / "buildings.jsonl", rows)
        raise

    write_jsonl(root / "buildings.jsonl", rows)
    print(f"Saved {len(rows)} records to {root / 'buildings.jsonl'}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bbox", nargs=4, type=float, default=[91000, 398000, 92000, 399000],
                        metavar=("MIN_X", "MIN_Y", "MAX_X", "MAX_Y"),
                        help="Dutch RD coordinates in metres, EPSG:28992 horizontal axes")
    parser.add_argument("--count", default=100, type=int)
    parser.add_argument("--page-size", default=10, type=int)
    args = parser.parse_args()
    bbox = args.bbox
    if not all(map(math.isfinite, bbox)) or not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        parser.error("--bbox must be an ordered, finite bounding box")
    if args.count < 1 or not 2 <= args.page_size <= 10:
        parser.error("--count must be positive and --page-size must be between 2 and 10")
    try:
        exported = collect(args.output, bbox, args.count, args.page_size)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Collection failed: {error}\n")
    if exported < args.count:
        parser.exit(1, f"Only {exported} buildings could be exported from this bounding box.\n")


if __name__ == "__main__":
    main()
