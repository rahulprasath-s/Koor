#!/usr/bin/env python3
"""Retest the first validation building with bounded, geometry-checked retries."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from evaluate_qwen_building import (
    BuildingCodeDataset,
    BuildingQwenCollator,
    MAX_TOKENS,
    MODEL_ID,
    PROCESSOR_REVISION,
    StopAtModule,
    compare_mesh,
    generation_inputs,
    match_module,
    parse_generated,
    read_reference_glb,
    sample_surface,
)


CORRECTION = (
    "Build a new mesh from the two images. Use indexed THREE.BufferGeometry with literal position and index arrays. "
    "Count the XYZ triplets in the position array first. With N vertices, each face index must be an integer from "
    "0 through N-1, and each triangle must use three distinct, non-collinear vertices. "
    "Use Y for height and put the roof above ground. Return the complete buildBuilding module only."
)


def diagnose(code):
    match = match_module(code)
    if match is None:
        return ["Output did not contain a complete module in the expected format."]
    try:
        positions = json.loads(re.sub(r",\s*\]$", "]", match["positions"]))
        indices = json.loads(re.sub(r",\s*\]$", "]", match["indices"]))
    except ValueError:
        return ["Position and index arrays must contain literal numeric values."]
    if (not positions or len(positions) % 3 or
            any(type(value) not in (int, float) for value in positions)):
        return ["Positions must be a nonempty array of complete XYZ triplets."]
    if not indices or len(indices) % 3 or any(type(value) is not int for value in indices):
        return ["Indices must be a nonempty array of complete integer triplets."]
    count = len(positions) // 3
    failures = []
    invalid = sorted({value for value in indices if not 0 <= value < count})
    if invalid:
        failures.append(f"Position array has {count} vertices, so valid indices are 0 to {count - 1}. "
                        f"Out-of-range indices included {', '.join(map(str, invalid[:8]))}.")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            vertices = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    except (OverflowError, ValueError):
        failures.append("Vertex positions exceed the numeric range of Three.js geometry.")
        return failures
    if not np.isfinite(vertices).all():
        failures.append("Vertex positions must be finite.")
    elif float(np.ptp(vertices[:, 1].astype(np.float64))) <= 0.001:
        failures.append("All vertex heights are effectively the same; the mesh has no height.")
    return failures


def validate_candidate(code):
    failures = diagnose(code)
    if failures:
        raise ValueError(" ".join(failures))
    vertices, faces = parse_generated(code)
    used = vertices[np.unique(faces)].astype(np.float64)
    if float(np.ptp(used[:, 1])) <= 0.001 or np.linalg.matrix_rank(used - used[0], tol=0.001) < 3:
        raise ValueError("Vertices referenced by triangles do not span a three-dimensional building.")
    return vertices, faces


def generate_candidate(model, processor, inputs, seed):
    prompt_length = inputs["input_ids"].shape[1]
    budget = min(4096, MAX_TOKENS - prompt_length)
    torch.manual_seed(seed)
    with torch.inference_mode():
        output = model.generate(
            **inputs, do_sample=True, temperature=0.45, top_p=0.9, top_k=0,
            max_new_tokens=budget, use_cache=True,
            stopping_criteria=[StopAtModule(processor.tokenizer, prompt_length)],
        )
    new_ids = output[0, prompt_length:].tolist()
    code = processor.tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return code, len(new_ids), budget


def retry_validation(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("This test requires a CUDA GPU with bfloat16 support")
    old_summary = json.loads((args.prior_eval / "summary.json").read_text(encoding="utf-8"))
    building_id = old_summary["selected_buildings"][0]
    dataset = BuildingCodeDataset(args.shard, "validation")
    if (old_summary["revision"] != PROCESSOR_REVISION or Path(old_summary["adapter"]).resolve() != args.adapter.resolve()
            or old_summary["validation_manifest_sha256"] != hashlib.sha256(dataset.manifest.read_bytes()).hexdigest()):
        raise ValueError("Use the same checkpoint and validation manifest as the prior evaluation")
    index = next((i for i, row in enumerate(dataset.rows) if row["building_id"] == building_id), None)
    if index is None:
        raise ValueError("Prior evaluation building is missing from the geographic validation manifest")
    old_code = (args.prior_eval / f"{building_id}.lora.txt").read_text(encoding="utf-8")
    try:
        validate_candidate(old_code)
        previous_failure = ""
    except (ValueError, OverflowError) as error:
        previous_failure = str(error)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Choose a new output directory")

    device, dtype = torch.device("cuda:0"), torch.bfloat16
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=PROCESSOR_REVISION, trust_remote_code=False)
    collator = BuildingQwenCollator(processor)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=PROCESSOR_REVISION, dtype=dtype, attn_implementation="sdpa",
        use_safetensors=True, trust_remote_code=False,
    ).to(device)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    sample = dataset[index]
    args.output.mkdir(parents=True)
    report = {"building_id": building_id, "previous_failure": previous_failure, "attempts": [], "accepted": None}
    feedback = previous_failure
    print(f"Building: {building_id} | previous output: {previous_failure or 'passed structural checks'}", flush=True)

    for attempt in range(1, args.attempts + 1):
        correction = f"\n\n{CORRECTION}"
        if feedback:
            correction += f" The previous attempt failed this check: {feedback}"
        inputs = generation_inputs(collator, {**sample, "prompt": sample["prompt"] + correction}, device, dtype)
        seed = 4200 + attempt
        code, tokens, budget = generate_candidate(model, processor, inputs, seed)
        file = args.output / f"attempt-{attempt:02d}.txt"
        file.write_text(code, encoding="utf-8")
        result = {"attempt": attempt, "seed": seed, "new_tokens": tokens, "reached_token_budget": tokens == budget,
                  "generated_text_path": file.name}
        report["attempts"].append(result)
        try:
            predicted = validate_candidate(code)
        except (ValueError, OverflowError) as error:
            feedback = str(error)
            result.update({"status": "rejected", "reason": feedback})
            print(f"Attempt {attempt}: rejected | {feedback}", flush=True)
        else:
            result["status"] = "accepted"
            # The reference is read only after accepting a candidate on structure.
            reference = read_reference_glb(dataset.asset_path(dataset.rows[index]["mesh_path"]))
            metrics = compare_mesh(predicted, reference, sample_surface(reference))
            report["accepted"] = {"attempt": attempt, **metrics}
            print(f"Attempt {attempt}: passed structural checks | surface distance "
                  f"{metrics['mean_surface_distance_m']:.3f} m | surface F1 {metrics['surface_f1_at_2pct']:.3f}", flush=True)
            break
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if report["accepted"] is None:
        print("No attempt passed the structural checks.")
    print(f"Saved retry report to {args.output / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--prior-eval", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempts", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.attempts <= 8:
        parser.error("--attempts must be between 1 and 8")
    try:
        retry_validation(args)
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Retry test stopped: {error}\n")


if __name__ == "__main__":
    main()
