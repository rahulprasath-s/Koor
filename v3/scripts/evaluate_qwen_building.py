#!/usr/bin/env python3
"""Compare base and LoRA Qwen3-VL on the same geographic validation buildings."""

import argparse
import hashlib
import json
import random
import re
import struct
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from scipy.spatial import cKDTree
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, StoppingCriteria

from building_code_dataset import BuildingCodeDataset
from check_building_code_tokens import MODEL_ID
from qwen_building_collator import BuildingQwenCollator, IGNORE_INDEX, MAX_TOKENS, PROCESSOR_REVISION


SURFACE_SAMPLES = 4096
MODES = ("base", "lora")
MODULE = re.compile(r"""
    export\s+function\s+buildBuilding\s*\(\s*THREE\s*\)\s*\{\s*
    (?:const|let)\s+(?P<indexed>[A-Za-z_$][\w$]*)\s*=\s*new\s+THREE\.BufferGeometry\s*\(\s*\)\s*;?\s*
    (?P=indexed)\.setAttribute\s*\(\s*['"]position['"]\s*,\s*
        new\s+THREE\.Float32BufferAttribute\s*\(\s*(?P<positions>\[[^\[\]]*\])\s*,\s*3\s*\)\s*\)\s*;?\s*
    (?P=indexed)\.setIndex\s*\(\s*(?P<indices>\[[^\[\]]*\])\s*\)\s*;?\s*
    (?:(?:const|let)\s+(?P<unindexed>[A-Za-z_$][\w$]*)\s*=\s*
        (?P=indexed)\.toNonIndexed\s*\(\s*\)\s*;?\s*)?
    (?:(?P<normals>[A-Za-z_$][\w$]*)\.computeVertexNormals\s*\(\s*\)\s*;?\s*)?
    return\s+new\s+THREE\.Mesh\s*\(\s*(?P<returned>[A-Za-z_$][\w$]*)\s*,\s*
        new\s+THREE\.Mesh(?:Standard|Basic|Phong|Lambert|Normal)Material\s*\(\s*
        (?:\{\s*(?:color\s*:\s*(?:0[xX][\da-fA-F]+|\d+|['"]\#[\da-fA-F]{3,8}['"])(?:\s*,)?)?\s*\})?
        \s*\)\s*\)\s*;?\s*\}\s*;?\s*
""", re.VERBOSE)


def match_module(text):
    text = re.sub(r"^\s*```(?:javascript|js|mjs)?[ \t]*\r?\n", "", text, count=1, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text, count=1)
    # The supported grammar has no strings containing comment delimiters.
    text = re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", "", text)
    return MODULE.fullmatch(text.strip())


def check_mesh(vertices, faces):
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Invalid vertex array")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Invalid triangle indices")
    triangles = vertices[faces]
    area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if np.any(~np.isfinite(area2)) or np.any(area2 <= 1e-12):
        raise ValueError("Non-finite or degenerate triangle")
    return triangles.astype(np.float64), area2.astype(np.float64)


def parse_generated(text):
    match = match_module(text)
    if match is None:
        raise ValueError("Incomplete code or unsupported Three.js module structure")
    indexed, unindexed = match["indexed"], match["unindexed"]
    expected = unindexed or indexed
    if indexed == unindexed or match["returned"] != expected or match["normals"] not in (None, expected):
        raise ValueError("Geometry variables do not refer to the exported mesh")
    if {indexed, expected} & {"THREE", "new", "const", "let", "return", "export", "function"}:
        raise ValueError("Geometry variable uses a reserved name")
    coords = json.loads(re.sub(r",\s*\]$", "]", match["positions"]))
    indices = json.loads(re.sub(r",\s*\]$", "]", match["indices"]))
    if any(type(value) not in (int, float) for value in coords) or any(type(value) is not int for value in indices):
        raise ValueError("Vertex and index arrays must contain numeric literals and integer indices respectively")
    with np.errstate(over="ignore", invalid="ignore"):
        vertices = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    check_mesh(vertices, faces)
    return vertices, faces


def read_reference_glb(path):
    """Read the one indexed triangle mesh emitted by our 3DBAG exporter."""
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or length != len(data):
        raise ValueError(f"Invalid GLB header: {path}")
    size, kind = struct.unpack_from("<I4s", data, 12)
    if kind != b"JSON":
        raise ValueError(f"Missing JSON chunk: {path}")
    spec = json.loads(data[20:20 + size])
    bin_size, bin_kind = struct.unpack_from("<I4s", data, 20 + size)
    binary = memoryview(data)[28 + size:]
    if bin_kind != b"BIN\0" or len(binary) != bin_size:
        raise ValueError(f"Invalid binary chunk: {path}")
    [mesh] = spec["meshes"]
    [primitive] = mesh["primitives"]
    if primitive.get("mode", 4) != 4 or any(
        {"translation", "rotation", "scale", "matrix"}.intersection(node) for node in spec["nodes"]
    ):
        raise ValueError(f"Expected untransformed triangle geometry: {path}")

    def read_accessor(index, kind, components, dtypes):
        accessor = spec["accessors"][index]
        view = spec["bufferViews"][accessor["bufferView"]]
        dtype = np.dtype(dtypes[accessor["componentType"]])
        if (accessor["type"] != kind or "sparse" in accessor or view["buffer"] != 0
                or view.get("byteStride", components * dtype.itemsize) != components * dtype.itemsize):
            raise ValueError(f"Unsupported GLB accessor: {path}")
        offset = accessor.get("byteOffset", 0)
        count = accessor["count"] * components
        start = view.get("byteOffset", 0) + offset
        if offset < 0 or start < 0 or count <= 0 or offset + count * dtype.itemsize > view["byteLength"]:
            raise ValueError(f"GLB accessor exceeds its buffer view: {path}")
        return np.frombuffer(binary, dtype=dtype, count=count, offset=start).reshape(-1, 3)

    verts = read_accessor(primitive["attributes"]["POSITION"], "VEC3", 3, {5126: "<f4"})
    faces = read_accessor(primitive["indices"], "SCALAR", 1, {5121: "u1", 5123: "<u2", 5125: "<u4"})
    check_mesh(verts, faces)
    return verts, faces


def sample_surface(mesh):
    triangles, areas = check_mesh(*mesh)
    rng = np.random.default_rng(42)
    selected = triangles[rng.choice(len(triangles), size=SURFACE_SAMPLES, p=areas / areas.sum())]
    uv = rng.random((SURFACE_SAMPLES, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    return selected[:, 0] + uv[:, :1] * (selected[:, 1] - selected[:, 0]) + uv[:, 1:] * (selected[:, 2] - selected[:, 0])


def compare_mesh(predicted, reference, reference_points):
    diagonal = float(np.linalg.norm(np.ptp(reference[0].astype(np.float64), axis=0)))
    if diagonal <= 0:
        raise ValueError("Reference has zero bounding-box diagonal")
    samples = sample_surface(predicted)
    forward = cKDTree(reference_points).query(samples)[0]
    backward = cKDTree(samples).query(reference_points)[0]
    distance = float((forward.mean() + backward.mean()) / 2)
    precision, recall = (float((values <= 0.02 * diagonal).mean()) for values in (forward, backward))
    f_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    faces = predicted[1]
    edges = np.sort(np.concatenate((faces[:, :2], faces[:, 1:], faces[:, [2, 0]])), axis=1)
    counts = np.unique(edges, axis=0, return_counts=True)[1]
    return {"mean_surface_distance_m": distance, "distance_fraction_of_reference_diagonal": distance / diagonal,
            "surface_f1_at_2pct": f_score, "reference_diagonal_m": diagonal,
            "triangles": len(faces), "all_indexed_edges_have_two_faces": bool(np.all(counts == 2))}


def generation_inputs(collator, sample, device, dtype):
    # Use a fixed dummy target to locate the assistant boundary with the audited
    # formatter. The validation reference never enters the generation prefix.
    batch = collator([{**sample, "target": "// generation boundary\n"}])
    labels = batch.pop("labels")
    start = int(torch.where(labels[0] != IGNORE_INDEX)[0][0])
    for key in ("input_ids", "attention_mask"):
        batch[key] = batch[key][:, :start]
    inputs = {key: value.to(device) for key, value in batch.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    return inputs


class StopAtModule(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        recent = input_ids[0, max(self.prompt_length, input_ids.shape[1] - 5):].tolist()
        tail = self.tokenizer.decode(recent, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if "}" not in tail:
            return False
        code = self.tokenizer.decode(input_ids[0, self.prompt_length:].tolist(), skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
        return match_module(code) is not None


def generate(model, processor, inputs, max_new_tokens):
    prompt_length = inputs["input_ids"].shape[1]
    budget = min(max_new_tokens, MAX_TOKENS - prompt_length)
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=budget, use_cache=True,
                                stopping_criteria=[StopAtModule(processor.tokenizer, prompt_length)])
    new_ids = output[0, prompt_length:].tolist()
    code = processor.tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return code, {"new_tokens": len(new_ids), "generation_budget": budget, "reached_token_budget": len(new_ids) == budget}


def evaluate(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("This evaluation requires a CUDA GPU with bfloat16 support")
    dataset = BuildingCodeDataset(args.shard, "validation")
    if not 1 <= args.count <= len(dataset):
        raise ValueError(f"--count must be between 1 and {len(dataset)}")
    ordered = sorted(range(len(dataset)), key=lambda index: dataset.rows[index]["building_id"])
    selected = random.Random(42).sample(ordered, args.count)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Choose a new output directory")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    print(f"GPU: {torch.cuda.get_device_name(0)} | validation buildings: {args.count}/{len(dataset)}", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=PROCESSOR_REVISION, trust_remote_code=False)
    collator = BuildingQwenCollator(processor)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=PROCESSOR_REVISION, dtype=dtype, attn_implementation="sdpa",
        use_safetensors=True, trust_remote_code=False,
    ).to(device)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    args.output.mkdir(parents=True)
    results = []
    with (args.output / "results.jsonl").open("w", encoding="utf-8") as output:
        for number, index in enumerate(selected, start=1):
            row = dataset.rows[index]
            building_id = row["building_id"]
            sample = dataset[index]
            inputs = generation_inputs(collator, sample, device, dtype)
            print(f"[{number}/{args.count}] {building_id}: generating base and LoRA outputs", flush=True)
            with model.disable_adapter():
                base_code, base_info = generate(model, processor, inputs, args.max_new_tokens)
            lora_code, lora_info = generate(model, processor, inputs, args.max_new_tokens)

            reference = read_reference_glb(dataset.asset_path(row["mesh_path"]))
            reference_points = sample_surface(reference)
            result = {"building_id": building_id, "reference_glb": row["mesh_path"]}
            for mode, code, info in (("base", base_code, base_info), ("lora", lora_code, lora_info)):
                path = args.output / f"{building_id}.{mode}.txt"
                path.write_text(code, encoding="utf-8")
                try:
                    metrics = compare_mesh(parse_generated(code), reference, reference_points)
                except (ValueError, OverflowError) as error:
                    info.update({"mesh_status": "unscored", "reason": str(error)})
                else:
                    info.update({"mesh_status": "scored", **metrics})
                result[mode] = {**info, "generated_text_path": path.name}
                print(f"  {mode}: {info['mesh_status']} | {info['new_tokens']} tokens | "
                      + (f"surface distance {info['mean_surface_distance_m']:.3f} m"
                         if "mean_surface_distance_m" in info else info["reason"]), flush=True)
            results.append(result)
            output.write(json.dumps(result, allow_nan=False) + "\n")
            output.flush()

    common = [row for row in results if all(row[mode]["mesh_status"] == "scored" for mode in MODES)]
    summary = {"base_model": MODEL_ID, "revision": PROCESSOR_REVISION, "adapter": str(args.adapter.resolve()),
               "validation_manifest_sha256": hashlib.sha256(dataset.manifest.read_bytes()).hexdigest(),
               "selected_buildings": [row["building_id"] for row in results], "max_new_tokens": args.max_new_tokens,
               "surface_samples_per_mesh": SURFACE_SAMPLES, "scored_by_both": len(common), "models": {}}
    for mode in MODES:
        scored = sum(row[mode]["mesh_status"] == "scored" for row in results)
        summary["models"][mode] = {
            "scored": scored,
            "matched_subset_mean_surface_distance_m": float(np.mean([
                row[mode]["mean_surface_distance_m"] for row in common
            ])) if common else None,
            "matched_subset_mean_surface_f1_at_2pct": float(np.mean([
                row[mode]["surface_f1_at_2pct"] for row in common
            ])) if common else None,
        }
        print(f"{mode}: scored {scored}/{len(results)} generated meshes")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Scored by both: {len(common)}. Saved comparison to {args.output / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--adapter", required=True, type=Path, help="Checkpoint's adapter directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens < MAX_TOKENS:
        parser.error(f"--max-new-tokens must be between 1 and {MAX_TOKENS - 1}")
    if not (args.adapter / "adapter_config.json").is_file():
        parser.error("--adapter must point to a checkpoint's adapter subdirectory")
    try:
        evaluate(args)
    except (OSError, ValueError, RuntimeError, KeyError, struct.error) as error:
        parser.exit(1, f"Evaluation stopped: {error}\n")


if __name__ == "__main__":
    main()
