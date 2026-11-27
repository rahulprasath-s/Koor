#!/usr/bin/env python3
"""Format a Qwen3-VL batch and audit full sequences on the geographic train split."""

import argparse
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from building_code_dataset import BuildingCodeDataset
from check_building_code_tokens import MODEL_ID


PROCESSOR_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
MAX_TOKENS = 8192
MIN_IMAGE_PIXELS = 256 * 256
MAX_IMAGE_PIXELS = 512 * 512
IGNORE_INDEX = -100


class BuildingQwenCollator:
    """Return Qwen3-VL tensors with labels on the target and optional end token."""

    def __init__(self, processor, max_tokens=MAX_TOKENS, supervise_end_token=False):
        if not processor.tokenizer.is_fast:
            raise ValueError("This collator requires the fast tokenizer for character offsets")
        self.processor = processor
        self.max_tokens = max_tokens
        self.supervise_end_token = supervise_end_token
        processor.tokenizer.padding_side = "right"
        image_processor = processor.image_processor
        self.patch_factor = image_processor.patch_size * image_processor.merge_size

    def __call__(self, samples):
        texts, prefixes, images = [], [], []
        image_token = self.processor.image_token
        for sample in samples:
            user = {"role": "user", "content": [
                {"type": "image"}, {"type": "image"}, {"type": "text", "text": sample["prompt"]},
            ]}
            target = sample["target"]
            assistant = {"role": "assistant", "content": [{"type": "text", "text": target}]}
            prefix = self.processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
            text = self.processor.apply_chat_template([user, assistant], tokenize=False, add_generation_prompt=False)
            if prefix.count(image_token) != 2 or text.count(image_token) != 2 or not text.startswith(prefix + target):
                raise ValueError(f"Unexpected Qwen template or target boundary for {sample['building_id']}")
            texts.append(text)
            prefixes.append(prefix)

            aerial, footprint = sample["aerial"], sample["footprint_mask"]
            width, height = aerial.size
            if footprint.size != aerial.size:
                raise ValueError(f"Image and footprint dimensions differ for {sample['building_id']}")
            new_height, new_width = smart_resize(height, width, factor=self.patch_factor,
                                                 min_pixels=MIN_IMAGE_PIXELS, max_pixels=MAX_IMAGE_PIXELS)
            size = (new_width, new_height)
            images.extend((aerial.resize(size, Image.Resampling.BICUBIC),
                           footprint.resize(size, Image.Resampling.NEAREST).convert("RGB")))

        encoded = self.processor(
            text=texts,
            images=images,
            text_kwargs={"return_offsets_mapping": True, "padding": True, "truncation": False},
            images_kwargs={"do_resize": False},
            return_tensors="pt",
        )
        if len(encoded["image_grid_thw"]) != 2 * len(samples):
            raise ValueError("Expected exactly two image grids per sample")
        offsets = encoded.pop("offset_mapping")
        ids = encoded["input_ids"]
        labels = torch.full_like(ids, IGNORE_INDEX)
        for index, (sample, prefix) in enumerate(zip(samples, prefixes, strict=True)):
            length = int(encoded["attention_mask"][index].sum())
            if self.max_tokens is not None and length > self.max_tokens:
                raise ValueError(f"{sample['building_id']}: {length:,} tokens exceeds {self.max_tokens:,}; no truncation applied")

            # Qwen expands the two image placeholders before tokenizing; offsets
            # are in the expanded string, so account for the extra placeholders.
            image_count = int((ids[index, :length] == self.processor.image_token_id).sum())
            start = len(prefix) + (image_count - 2) * len(image_token)
            end = start + len(sample["target"])
            starts, ends = offsets[index].unbind(dim=1)
            supervised = (starts >= start) & (ends <= end) & (ends > starts) & encoded["attention_mask"][index].bool()
            code_ids = ids[index, supervised]
            recovered = self.processor.tokenizer.decode(code_ids.tolist(), clean_up_tokenization_spaces=False)
            if not len(code_ids) or recovered != sample["target"]:
                raise ValueError(f"Labels do not reconstruct the full target for {sample['building_id']}")
            labels[index, supervised] = code_ids
            if self.supervise_end_token:
                eos = self.processor.tokenizer.eos_token
                eos_id = self.processor.tokenizer.eos_token_id
                if not eos or eos_id is None:
                    raise ValueError("Qwen tokenizer is missing its end token")
                end_mask = ((starts == end) & (ends == end + len(eos)) & (ids[index] == eos_id)
                            & encoded["attention_mask"][index].bool())
                if int(end_mask.sum()) != 1:
                    raise ValueError(f"Expected exactly one end token after the target for {sample['building_id']}")
                labels[index, end_mask] = eos_id
        encoded["labels"] = labels
        return encoded


def percentile(values, percent):
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * percent / 100) - 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    try:
        dataset = BuildingCodeDataset(args.shard, "train")
        print(f"Auditing {len(dataset)} geographic training examples, with an {MAX_TOKENS:,}-token limit…", flush=True)
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=PROCESSOR_REVISION, trust_remote_code=False)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=BuildingQwenCollator(processor))
        lengths, supervised_counts = [], []
        for batch in loader:
            previous = len(lengths)
            lengths.extend(batch["attention_mask"].sum(dim=1).tolist())
            supervised_counts.extend((batch["labels"] != IGNORE_INDEX).sum(dim=1).tolist())
            if len(lengths) // 100 > previous // 100 or len(lengths) == len(dataset):
                print(f"Checked {len(lengths)}/{len(dataset)}", flush=True)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Audit failed: {error}\n")
    longest = max(range(len(lengths)), key=lengths.__getitem__)
    print(f"PASS: {len(lengths)}/{len(dataset)} fit without truncation and decode to the complete source from labels alone.")
    print(f"Full tokens: p50={percentile(lengths, 50):,}, p95={percentile(lengths, 95):,}, max={lengths[longest]:,}")
    print(f"Longest sequence: {dataset.rows[longest]['building_id']}; {supervised_counts[longest]:,} code tokens")


if __name__ == "__main__":
    main()
