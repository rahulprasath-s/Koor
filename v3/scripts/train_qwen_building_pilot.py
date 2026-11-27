#!/usr/bin/env python3
"""Fine-tune Qwen3-VL on geographic training rows and checkpoint LoRA progress."""

import argparse
import hashlib
import json
import math
import random
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.checkpoint import checkpoint
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from building_code_dataset import BuildingCodeDataset
from check_building_code_tokens import MODEL_ID
from qwen_building_collator import BuildingQwenCollator, IGNORE_INDEX, MAX_TOKENS, PROCESSOR_REVISION
from train_qwen_building_lora import add_language_lora


SEED = 42
LOGIT_CHUNK = 128


def code_loss_sum(model, batch):
    """Evaluate the next-token loss for only the positions that predict code."""
    base = model.get_base_model()
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    hidden = base.model(**inputs, use_cache=False).last_hidden_state
    shifted = batch["labels"][:, 1:]
    include = shifted != IGNORE_INDEX
    targets = shifted[include]
    if not targets.numel():
        raise ValueError("Training example has no code labels after the causal shift")
    states = hidden[:, :-1][include]

    def chunk_loss(chunk_states, chunk_targets):
        logits = base.lm_head(chunk_states).float()
        return F.cross_entropy(logits, chunk_targets, reduction="sum")

    # Recompute a chunk during backward so full-vocabulary logits from every
    # code position never have to stay resident at once.
    loss = torch.stack([
        checkpoint(chunk_loss, states[start:start + LOGIT_CHUNK], targets[start:start + LOGIT_CHUNK],
                   use_reentrant=False)
        for start in range(0, targets.numel(), LOGIT_CHUNK)
    ]).sum()
    return loss, targets.numel()


def save_checkpoint(run, model, optimizer, progress, settings):
    destination = run / f"step-{progress['step']:06d}"
    with tempfile.TemporaryDirectory(dir=run, prefix=".checkpoint-") as temp:
        staged = Path(temp) / "checkpoint"
        staged.mkdir()
        model.save_pretrained(staged / "adapter", safe_serialization=True)
        torch.save({"optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None},
                   staged / "optimizer.pt")
        (staged / "progress.json").write_text(
            json.dumps({"progress": progress, "settings": settings}, indent=2) + "\n", encoding="utf-8"
        )
        staged.rename(destination)
    print(f"Checkpoint: {destination}", flush=True)


def load_training_model(device, resume):
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=PROCESSOR_REVISION, dtype=torch.bfloat16, use_safetensors=True,
        attn_implementation="sdpa", trust_remote_code=False,
    ).to(device)
    if resume is None:
        model, _ = add_language_lora(base)
    else:
        model = PeftModel.from_pretrained(base, resume / "adapter", is_trainable=True)
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable or any(".lora_" not in name or ".language_model." not in name for name in trainable):
        raise RuntimeError("Trainable parameters must belong to the language LoRA adapters")
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    # The vision weights remain frozen and its features need no training-mode dropout.
    base.model.visual.eval()
    return model, list(trainable.values())


def training_data(shard, geometry_targets=None):
    if geometry_targets is None:
        return BuildingCodeDataset(shard, "train"), {}

    from prepare_qwen_geometry_targets import PreparedGeometryDataset

    dataset = PreparedGeometryDataset(geometry_targets)
    summary = dataset.summary
    if dataset.base.asset_root != shard.resolve():
        raise ValueError("Prepared targets point to a different shard")
    if (summary["split"] != "train" or summary["model"] != MODEL_ID
            or summary["processor_revision"] != PROCESSOR_REVISION or summary["max_tokens"] != MAX_TOKENS):
        raise ValueError("Prepared targets were audited with a different split, processor or token budget")
    return dataset, {"target_format": "triangles_mm_json", "supervise_end_token": True,
                     "prepared_targets_sha256": dataset.target_fingerprint(),
                     "source_train_manifest_sha256": summary["train_manifest_sha256"]}


def train(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Pilot training needs a CUDA GPU with bfloat16 support")
    torch.manual_seed(SEED)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device("cuda:0")
    dataset, target_settings = training_data(args.shard, args.geometry_targets)
    run = args.run_dir.resolve()
    settings = {"base_model": MODEL_ID, "revision": PROCESSOR_REVISION, "seed": SEED,
                "train_manifest_sha256": hashlib.sha256(dataset.manifest.read_bytes()).hexdigest(),
                "accumulate": args.accumulate, "lr": args.lr, "examples_per_epoch": len(dataset),
                "logit_chunk": LOGIT_CHUNK, **target_settings}
    previous = sorted(path for path in run.glob("step-*")
                      if path.is_dir() and path.name.removeprefix("step-").isdigit())
    resume = None
    progress = {"step": 0, "epoch": 0, "next_example": 0}
    if args.resume:
        if not previous:
            raise ValueError("No completed checkpoint in this run directory")
        resume = max(previous, key=lambda path: int(path.name.removeprefix("step-")))
        state = json.loads((resume / "progress.json").read_text(encoding="utf-8"))
        if state["settings"] != settings:
            raise ValueError("Resume requires the same training manifest, learning rate and accumulation settings")
        progress = state["progress"]
    elif run.exists() and any(run.iterdir()):
        raise ValueError("Run directory is not empty. Use --resume or a new --run-dir.")
    run.mkdir(parents=True, exist_ok=True)
    if progress["epoch"] >= args.epochs:
        print(f"Already completed {progress['epoch']} epoch(s). Latest checkpoint: {resume}")
        return

    print(f"GPU: {torch.cuda.get_device_name(device)} | training examples: {len(dataset)} | epochs: {args.epochs}",
          flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=PROCESSOR_REVISION, trust_remote_code=False)
    supervise_end = bool(target_settings)
    if supervise_end and processor.tokenizer.eos_token_id != dataset.summary["eos_token_id"]:
        raise ValueError("Loaded tokenizer end token differs from the audited target format")
    if supervise_end:
        print(f"Target: geometry JSON + {processor.tokenizer.eos_token}", flush=True)
    collator = BuildingQwenCollator(processor, supervise_end_token=supervise_end)
    model, parameters = load_training_model(device, resume)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0)
    if resume is not None:
        state = torch.load(resume / "optimizer.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        print(f"Resumed at step {progress['step']}, next example {progress['next_example']}", flush=True)

    for epoch in range(progress["epoch"], args.epochs):
        order = list(range(len(dataset)))
        random.Random(SEED + epoch).shuffle(order)
        start = progress["next_example"]
        for offset in range(start, len(order), args.accumulate):
            end = min(offset + args.accumulate, len(order))
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), device=device)
            total_tokens = 0
            for index in order[offset:end]:
                sample = dataset[index]
                batch = {key: value.to(device) for key, value in collator([sample]).items()}
                batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
                try:
                    loss_sum, token_count = code_loss_sum(model, batch)
                    loss_sum.backward()
                except torch.OutOfMemoryError as error:
                    raise RuntimeError(f"CUDA ran out of memory for {sample['building_id']}. "
                                       "The latest complete checkpoint is preserved.") from error
                total_loss += loss_sum.detach()
                total_tokens += token_count

            mean_loss = float(total_loss / total_tokens)
            if not math.isfinite(mean_loss):
                raise RuntimeError("Non-finite target loss; optimizer step skipped")
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(total_tokens)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            if float(norm) <= 0:
                raise RuntimeError("LoRA gradient norm is zero")
            optimizer.step()

            finished = end == len(order)
            step = progress["step"] + 1
            progress = {"step": step, "epoch": epoch + int(finished), "next_example": 0 if finished else end}
            print(f"epoch {epoch + 1}/{args.epochs} | step {step} | examples {end}/{len(order)} | "
                  f"target loss {mean_loss:.5f} | supervised tokens {total_tokens:,}", flush=True)
            if step == 1 or step % args.save_every == 0 or finished:
                save_checkpoint(run, model, optimizer, progress, settings)
    print(f"Completed {args.epochs} epoch(s); peak CUDA allocated {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--geometry-targets", type=Path, help="Prepared, audited JSON targets; supervise JSON and EOS")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=10, help="Checkpoint every N optimizer steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.accumulate, args.save_every) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Epochs, accumulation, checkpoint interval and learning rate must be positive")
    try:
        train(args)
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Rerun with --resume to use the latest complete checkpoint.\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Training stopped: {error}\n")


if __name__ == "__main__":
    main()
