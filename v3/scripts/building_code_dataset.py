#!/usr/bin/env python3
"""Load geographic 3DBAG/PDOK splits for aerial photo and mask to Three.js training."""

import argparse
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset


SPLITS = ("train", "validation", "test")
PROMPT = (
    "The first image is an aerial photo. The second image is a binary footprint mask: "
    "white pixels identify the target building. Reconstruct that building as a simplified 3D exterior. "
    "Return a JavaScript ES module exporting buildBuilding(THREE). The function must construct "
    "the roof and walls as geometry in local Y-up coordinates and return a THREE.Mesh. "
    "Return code only."
)


class BuildingCodeDataset(Dataset):
    """One row per building. Image preprocessing stays with the downstream model."""

    def __init__(self, shard, split="train"):
        if split not in SPLITS:
            raise ValueError(f"Unknown geographic split: {split}")
        split_dir = Path(shard).resolve() / "splits-geographic"
        summary = json.loads((split_dir / "summary.json").read_text(encoding="utf-8"))
        self.asset_root = (split_dir / summary["asset_paths_relative_to"]).resolve()
        self.split = split
        self.manifest = split_dir / f"{split}.jsonl"
        with self.manifest.open(encoding="utf-8") as file:
            self.rows = [json.loads(line) for line in file if line.strip()]
        if len(self.rows) != summary["counts"][split] or not self.rows:
            raise ValueError(f"{split} manifest count disagrees with the geographic split summary")
        if len({row["building_id"] for row in self.rows}) != len(self.rows):
            raise ValueError(f"Duplicate building IDs in {self.manifest}")

    def __len__(self):
        return len(self.rows)

    def asset_path(self, value):
        path = (self.asset_root / value).resolve()
        if not path.is_relative_to(self.asset_root):
            raise ValueError(f"Asset path is outside the shard: {value}")
        return path

    def __getitem__(self, index):
        row = self.rows[index]
        building_id = row["building_id"]
        with Image.open(self.asset_path(row["aerial_path"])) as source:
            aerial = source.convert("RGB")
        with Image.open(self.asset_path(row["footprint_mask_path"])) as source:
            footprint = source.convert("L")
        if aerial.size != footprint.size or aerial.size != tuple(row["aerial"]["image_size_px"]):
            raise ValueError(f"Image and mask dimensions disagree for {building_id}")
        colors = footprint.getcolors(maxcolors=2)
        if colors is None or {value for _, value in colors} != {0, 255}:
            raise ValueError(f"Invalid binary footprint mask for {building_id}")
        target = self.asset_path(row["threejs_path"]).read_text(encoding="utf-8")
        if "export function buildBuilding(THREE)" not in target:
            raise ValueError(f"Missing buildBuilding export for {building_id}")
        return {"building_id": building_id, "aerial": aerial, "footprint_mask": footprint,
                "prompt": PROMPT, "target": target}


def collate_buildings(samples):
    """Keep variable-size images and untruncated code for model-specific processing."""
    return {key: [sample[key] for sample in samples] for key in samples[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        dataset = BuildingCodeDataset(args.shard, args.split)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_buildings)
        batch = next(iter(loader))
        print(f"Geographic split: {dataset.split} | {len(dataset)} buildings")
        for building_id, aerial, mask, code in zip(
            batch["building_id"], batch["aerial"], batch["footprint_mask"], batch["target"], strict=True
        ):
            print(f"{building_id}: aerial {aerial.size} {aerial.mode} + mask {mask.size} {mask.mode} -> "
                  f"{len(code.encode('utf-8')):,} bytes of Three.js code")
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Loader test failed: {error}\n")
    print("PASS: one batch loaded with aligned inputs and complete code targets.")


if __name__ == "__main__":
    main()
