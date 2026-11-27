#!/usr/bin/env python3
"""Measure full Three.js target lengths in the geographic training split."""

import argparse
import math
from pathlib import Path

from transformers import AutoTokenizer

from building_code_dataset import BuildingCodeDataset


MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


def measure_targets(dataset, tokenizer, batch_size=32):
    lengths = []
    for start in range(0, len(dataset), batch_size):
        rows = dataset.rows[start:start + batch_size]
        targets = [dataset.asset_path(row["threejs_path"]).read_text(encoding="utf-8") for row in rows]
        encoded = tokenizer(targets, add_special_tokens=False, truncation=False, padding=False)
        for row, ids in zip(rows, encoded["input_ids"], strict=True):
            if not ids:
                raise ValueError(f"Empty code target for {row['building_id']}")
            lengths.append((len(ids), row["building_id"]))
    return sorted(lengths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path, help="Directory containing splits-geographic")
    args = parser.parse_args()
    dataset = BuildingCodeDataset(args.shard, "train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=False)
    lengths = measure_targets(dataset, tokenizer)
    count = len(lengths)
    print(f"Tokenizer: {MODEL_ID} ({type(tokenizer).__name__})")
    print(f"Geographic train split: {count} targets")
    for percentile in (50, 90, 95, 99, 100):
        print(f"p{percentile:>3}: {lengths[math.ceil(percentile / 100 * count) - 1][0]:,} tokens")
    for limit in (512, 1024, 2048):
        exceeds = sum(length > limit for length, _ in lengths)
        print(f"Above {limit:,} tokens: {exceeds}/{count}")
    print(f"Longest building: {lengths[-1][1]}")
    print("Code tokens only. Image tokens, instructions, and chat markers are additional.")


if __name__ == "__main__":
    main()
