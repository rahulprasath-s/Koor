#!/usr/bin/env python3
"""Run a CUDA LoRA smoke test for aerial photo and footprint to Three.js."""

import argparse
import math
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from building_code_dataset import BuildingCodeDataset
from check_building_code_tokens import MODEL_ID
from qwen_building_collator import BuildingQwenCollator, IGNORE_INDEX, PROCESSOR_REVISION


TARGET_SUFFIXES = (".self_attn.q_proj", ".self_attn.v_proj")


def add_language_lora(base):
    targets = [name for name, layer in base.named_modules()
               if name.startswith("model.language_model.layers.") and name.endswith(TARGET_SUFFIXES)
               and isinstance(layer, torch.nn.Linear)]
    expected = 2 * base.config.text_config.num_hidden_layers
    if len(targets) != expected:
        raise RuntimeError(f"Expected {expected} text attention projections, found {len(targets)}")
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                          bias="none", target_modules=targets))
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable or any(".lora_" not in name or ".language_model." not in name for name in trainable):
        raise RuntimeError("Unexpected trainable parameters outside the text LoRA adapters")
    return model, trainable


def optimizer_step(model, batch, optimizer, trainable, device, visual_dtype):
    token_count = int((batch["labels"] != IGNORE_INDEX).sum())
    if token_count == 0:
        raise RuntimeError("Batch has no supervised code tokens")
    optimizer.zero_grad(set_to_none=True)
    batch = {name: value.to(device) for name, value in batch.items()}
    batch["pixel_values"] = batch["pixel_values"].to(visual_dtype)

    outputs = model(**batch, use_cache=False)
    loss = outputs.loss
    if loss is None or not torch.isfinite(loss).item():
        raise RuntimeError("Model loss is missing or non-finite")
    loss.backward()

    gradients = [(name, parameter) for name, parameter in trainable.items() if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("LoRA adapters received no gradients")
    norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), max_norm=1.0, error_if_nonfinite=True)
    if not norm.item() > 0:
        raise RuntimeError("LoRA gradient norm is zero")
    name, tracked = max(gradients, key=lambda item: item[1].grad.detach().norm().item())
    before = tracked.detach().clone()
    optimizer.step()
    if not torch.isfinite(tracked).all().item() or torch.equal(before, tracked.detach()):
        raise RuntimeError(f"Optimizer did not produce a finite adapter change: {name}")
    return float(loss.detach()), token_count, float(norm), name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path, help="The shard with its saved splits-geographic folder")
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be positive and finite")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.exit(1, "This smoke test needs a CUDA GPU with bfloat16 support.\n")

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    dataset = BuildingCodeDataset(args.shard, "train")
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=PROCESSOR_REVISION, trust_remote_code=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=BuildingQwenCollator(processor))
    batch = next(iter(loader))
    print(f"Geographic train split: {len(dataset)} rows | smoke building: {dataset.rows[0]['building_id']}", flush=True)
    print(f"Full sequence: {int(batch['attention_mask'].sum()):,} tokens. Loading {MODEL_ID}…", flush=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=PROCESSOR_REVISION, dtype=dtype, use_safetensors=True,
        attn_implementation="sdpa", trust_remote_code=False,
    ).to(device)
    model, trainable = add_language_lora(base)
    model.train()
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=args.lr, weight_decay=0)
    loss, code_tokens, norm, updated = optimizer_step(model, batch, optimizer, trainable, device, dtype)
    print(f"Step 1/1: loss={loss:.6f} | code tokens={code_tokens:,} | LoRA grad norm={norm:.6f}")
    print(f"PASS: one optimizer step updated {updated}")
    print(f"Peak CUDA memory allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
