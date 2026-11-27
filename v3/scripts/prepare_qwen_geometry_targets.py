#!/usr/bin/env python3
"""Encode and audit Qwen geometry targets for the geographic training split."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor

from building_code_dataset import BuildingCodeDataset
from check_building_code_tokens import MODEL_ID
from qwen_building_collator import BuildingQwenCollator, IGNORE_INDEX, MAX_TOKENS, PROCESSOR_REVISION, percentile


GEOMETRY_PROMPT = (
    "The first image is an aerial photo. The second is a binary footprint mask, with white marking the target building. "
    "Predict its simplified exterior geometry in local Y-up coordinates. Return one JSON object with the key "
    "triangles_mm. Its value must be a list of triangles in millimetres. Each triangle is nine integers in the order "
    "x1, y1, z1, x2, y2, z2, x3, y3, z3. Return JSON only."
)


class PreparedGeometryDataset(Dataset):
    """Pair fitted JSON targets with their original geographic training inputs."""

    def __init__(self, prepared):
        self.root = Path(prepared).resolve()
        self.summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        self.base = BuildingCodeDataset(self.root / self.summary["shard_relative_to_output"], "train")
        digest = hashlib.sha256(self.base.manifest.read_bytes()).hexdigest()
        if digest != self.summary["train_manifest_sha256"]:
            raise ValueError("Geographic training manifest changed after target preparation")
        self.manifest = self.root / "fit-8192.jsonl"
        self.rows = [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(self.rows) != self.summary["fit"] or not self.rows:
            raise ValueError("Prepared target count differs from the audit summary")
        if len({row["building_id"] for row in self.rows}) != len(self.rows):
            raise ValueError("Duplicate building IDs in the fitted target manifest")

    def __len__(self):
        return len(self.rows)

    def target_path(self, row):
        source = (self.root / row["geometry_target_path"]).resolve()
        if not source.is_relative_to(self.root / "targets"):
            raise ValueError("Prepared target path escapes the targets directory")
        return source

    def target_fingerprint(self):
        digest = hashlib.sha256(self.manifest.read_bytes())
        for row in self.rows:
            digest.update(hashlib.sha256(self.target_path(row).read_bytes()).digest())
        return digest.hexdigest()

    def __getitem__(self, index):
        row = self.rows[index]
        sample = self.base[row["train_row_index"]]
        if sample["building_id"] != row["building_id"] or row["full_tokens"] > MAX_TOKENS:
            raise ValueError("Prepared target does not belong to this training example or exceeds the limit")
        return {**sample, "prompt": GEOMETRY_PROMPT, "target": self.target_path(row).read_text(encoding="utf-8")}


def save_manifest(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_collinear_face(points_mm, faces):
    """Replace an isolated collinear triangle by splitting its neighbor's long edge."""
    import numpy as np

    points = points_mm[faces]
    degenerate = np.flatnonzero(np.all(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]) == 0, axis=1))
    if not len(degenerate):
        return faces, 0
    if len(degenerate) != 1:
        raise ValueError("Expected at most one isolated collinear triangle")
    bad = int(degenerate[0])
    a, b, c = map(int, faces[bad])
    pairs = [(a, b, c), (a, c, b), (b, c, a)]
    u, v, middle = max(pairs, key=lambda pair: int(np.square(points_mm[pair[0]] - points_mm[pair[1]]).sum()))
    if len({tuple(points_mm[i]) for i in (u, v, middle)}) != 3:
        raise ValueError("Coincident vertices in collinear triangle")

    neighbors = [i for i, face in enumerate(faces) if i != bad and u in face and v in face]
    if len(neighbors) != 1:
        raise ValueError("Collinear triangle does not have one neighbor along its long edge")
    neighbor = neighbors[0]
    face = faces[neighbor]
    edge_start = next(i for i in range(3) if {int(face[i]), int(face[(i + 1) % 3])} == {u, v})
    start, end, other = [int(face[(edge_start + i) % 3]) for i in range(3)]
    replacements = [[start, middle, other], [middle, end, other]]
    return np.concatenate((np.delete(faces, [bad, neighbor], axis=0), replacements)), 1


def prepare(shard, output, batch_size):
    # These mesh dependencies are only needed when preparing targets.
    import numpy as np
    import trimesh
    from building_geometry_target import decode_geometry, encode_geometry

    if output.exists() or output.is_symlink():
        raise ValueError("Choose a new output directory")
    dataset = BuildingCodeDataset(shard, "train")
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=PROCESSOR_REVISION, trust_remote_code=False)
    collator = BuildingQwenCollator(processor, max_tokens=None, supervise_end_token=True)
    eos_id = processor.tokenizer.eos_token_id
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".json-target-audit-") as temp:
        stage = Path(temp) / "prepared"
        (stage / "targets").mkdir(parents=True)
        for start in range(0, len(dataset), batch_size):
            samples, batch_rows = [], []
            for index in range(start, min(start + batch_size, len(dataset))):
                row = dataset.rows[index]
                building_id = row["building_id"]
                mesh = trimesh.load(dataset.asset_path(row["mesh_path"]), force="mesh", process=False)
                points_mm = np.rint(mesh.vertices * 1000).astype(np.int64)
                try:
                    source_faces, repaired = split_collinear_face(points_mm, mesh.faces)
                    target = encode_geometry(mesh.vertices, source_faces)
                    vertices, faces = decode_geometry(target)
                except ValueError as error:
                    raise ValueError(f"{building_id}: {error}") from error
                error = float(np.max(np.abs(mesh.vertices[source_faces] - vertices[faces])))
                if error > 0.000501:
                    raise ValueError(f"Geometry encoding exceeds millimetre rounding tolerance for {building_id}")
                if repaired and not trimesh.Trimesh(vertices=vertices, faces=faces, process=False).is_watertight:
                    raise ValueError(f"Collinear-face repair broke mesh closure for {building_id}")
                target_path = Path("targets") / f"{building_id}.json"
                if target_path.parent != Path("targets"):
                    raise ValueError(f"Invalid building ID: {building_id}")
                (stage / target_path).write_text(target, encoding="utf-8")
                samples.append({**dataset[index], "prompt": GEOMETRY_PROMPT, "target": target})
                batch_rows.append({"building_id": building_id, "train_row_index": index,
                                   "geometry_target_path": target_path.as_posix(), "collinear_faces_repaired": repaired})
            encoded = collator(samples)
            lengths = encoded["attention_mask"].sum(dim=1).tolist()
            for row, length, labels in zip(batch_rows, lengths, encoded["labels"], strict=True):
                supervised = labels[labels != IGNORE_INDEX]
                if int(supervised[-1]) != eos_id or int((supervised == eos_id).sum()) != 1:
                    raise ValueError(f"Incorrect end-token supervision for {row['building_id']}")
                records.append({**row, "full_tokens": length, "geometry_tokens": len(supervised) - 1,
                                "end_tokens_supervised": 1, "fits_8192": length <= MAX_TOKENS})
            if len(records) // 100 > start // 100 or len(records) == len(dataset):
                print(f"Audited {len(records)}/{len(dataset)} training targets", flush=True)

        fit = [row for row in records if row["fits_8192"]]
        over = [row for row in records if not row["fits_8192"]]
        save_manifest(stage / "examples.jsonl", records)
        save_manifest(stage / "fit-8192.jsonl", fit)
        save_manifest(stage / "over-8192.jsonl", over)
        lengths = [row["full_tokens"] for row in records]
        summary = {"split": "train", "shard_relative_to_output": os.path.relpath(shard.resolve(), output.resolve()),
                   "train_manifest_sha256": hashlib.sha256(dataset.manifest.read_bytes()).hexdigest(),
                   "model": MODEL_ID, "processor_revision": PROCESSOR_REVISION,
                   "max_tokens": MAX_TOKENS, "eos_token": processor.tokenizer.eos_token, "eos_token_id": eos_id,
                   "total": len(records), "fit": len(fit), "over_limit": len(over),
                   "buildings_with_collinear_face_repair": sum(bool(row["collinear_faces_repaired"]) for row in records),
                   "full_tokens": {"p50": percentile(lengths, 50), "p95": percentile(lengths, 95), "max": max(lengths)}}
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    print(f"{len(fit)}/{len(records)} fit; {len(over)} over {MAX_TOKENS:,}. No examples were truncated.")
    print(f"Longest full sequence: {max(lengths):,} tokens. End-token supervision verified for every target.")
    print(f"Saved targets and manifests in {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    try:
        prepare(args.shard, args.output, args.batch_size)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")


if __name__ == "__main__":
    main()
