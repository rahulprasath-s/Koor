# Koor v3

I wanted to see whether an aerial photo and a footprint are enough to reconstruct the visible shape of a building. The first pilot asked Qwen3-VL to write Three.js directly. Its LoRA produced one mesh the evaluator could score across five validation buildings. Other outputs had problems such as out-of-range indices and unfinished modules.

The current pilot learns a JSON object with a `triangles_mm` field. Each triangle has nine integers for three XYZ points in local Y-up coordinates. The converter checks the geometry, assigns vertex indices, and writes the Three.js module. A reference GLB round-trip verified that the conversion preserves the reference shape to millimetre rounding precision. Measuring the accuracy of generated geometry on the geographic holdout comes after training.

## Published adapter

The public release contains the step-98 PEFT LoRA adapter from one epoch of JSON geometry training on SECE. Load it with Qwen3-VL-2B-Instruct at revision `89644892e4d85e24eaac8bacfd4f463576704203`. The model card accompanies `adapter_config.json` and `adapter_model.safetensors`. Held-out mesh quality has yet to be measured for this checkpoint. [Koor v3 adapter on Hugging Face](https://huggingface.co/Uruva/Koor-aerial-reconstruction-v3)

## Dataset and scripts

The pilot pairs 1,000 Dutch building meshes from 3DBAG with georeferenced 2024 PDOK aerial crops. Each sample includes a footprint mask, reference GLB, Three.js module, and metadata. Geographic splits contain 797 train, 94 validation, and 99 test buildings. The split procedure excludes 10 examples in geographic buffers. The JSON pilot uses the 782 training examples whose full input and target fit in 8,192 tokens.

- `collect_3dbag_pdok.py`, `verify_3dbag_pdok.py`, and `split_3dbag_pdok_geographic.py` build and check the paired shard.
- `building_geometry_target.py` tests reference GLB conversion. `prepare_qwen_geometry_targets.py` builds and audits the training targets.
- `train_qwen_building_pilot.py` trains the adapter and saves resumable checkpoints. `evaluate_qwen_building.py` and `retry_qwen_building_mesh.py` reproduce the earlier direct Three.js experiment.

See the [dataset card](DATASET_CARD.md) for attribution, data changes, and coverage.

## Run the JSON pilot

Use Python 3.12 and a CUDA-enabled PyTorch installation with matching torchvision. On SECE the environment used PyTorch 2.14.1+cu132, torchvision 0.29.1+cu132, Transformers 4.57.6 and PEFT 0.18.1. Install the remaining dependencies with `python -m pip install -r v3/requirements.txt`.

From the repository root, set `SHARD` to your paired data directory and keep the prepared geometry folder beside it so the manifest paths work on both Mac and SECE:

```bash
SHARD="$HOME/Koor-building-code-dataset/data/3dbag/shard-1000"
TARGETS="$(dirname "$SHARD")/geometry-training-8192"
python v3/scripts/prepare_qwen_geometry_targets.py "$SHARD" --output "$TARGETS"
python -u v3/scripts/train_qwen_building_pilot.py "$SHARD" \
  --geometry-targets "$TARGETS" \
  --run-dir runs/qwen-building-geometry-pilot \
  --epochs 1 --accumulate 8 --save-every 10
```

Use an existing audited target directory directly if it is already available. The trainer starts with the pinned Qwen3-VL-2B-Instruct base weights. It supervises the JSON and assistant end token, masking the image and prompt tokens. With these settings, it checkpoints at step 1, every 10 optimizer steps, and at the final step 98. Append `--resume` to resume the same run with its original settings.
