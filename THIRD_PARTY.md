# Credits and source history

Koor depends on research and code that others made available. This page records that work, the version used, and where the source in this repository came from.

## Point2Roof

- Repository: https://github.com/Li-Li-Whu/Point2Roof
- Recorded upstream commit: `ab3d273f1dce45e7fc2e587bd17a7b8ec2fd3aad`
- Authors: Li Li, Nan Song, Fei Sun, Xinyi Liu, Ruisheng Wang, Jian Yao, and Shapsheng Cao.
- Paper: *Point2Roof: End-to-end 3D building roof modeling from airborne LiDAR point clouds*.
- Software notice: [MIT license](v1/Point2Roof/LICENSE), preserved verbatim.

I used Point2Roof's architecture and custom CUDA operations, then adapted the code for the training environment. The changes remove obsolete THC references and update tensor access. The full trainer also handles singleton batch normalization and indexes edge pairs across buildings. The [source patch](v1/provenance/point2roof_changes.patch) and [training settings](v1/training_metadata.json) record those changes.

## Building3D

- Dataset: https://huggingface.co/datasets/Building3D/Building3D
- Project: https://building3d.ucalgary.ca/
- Authors: Ruisheng Wang, Shangfeng Huang, and Hongxin Yang.
- Paper: *Building3D: An Urban-Scale Dataset and Benchmarks for Learning Roof Structures from Point Clouds*, ICCV 2023.
- Listed data license: CC BY-NC-SA 4.0.

V1's real-data scores use a custom spatial holdout prepared from labelled Entry-level data. The official unlabelled test set was not scored. No Building3D training is claimed for v1.

V2 trains on 4,538 labelled Entry-level roofs, validates on 533 and evaluates on a separate 561-roof spatial holdout. Synthetic replay and a frozen v1 corner teacher are used during training. The official 583 unlabelled test clouds remain unused for scoring. The [v2 reports](v2/evaluation/) and [split identifiers](v2/provenance/split_ids.json) record that experiment.

## Published Koor source

The initial GitHub copy came from my published release at https://huggingface.co/Uruva/Koor-building-reconstruction-v1, revision `8dac21750a0231cede813f7f8aa52d248cb57816`. Source hashes were checked against that release's manifest. Weights remain on Hugging Face. The [provenance record](v1/provenance/source.json) keeps track of this transfer.

V2 source and aggregate reports were exported from Alex after held-out evaluation. Every exported file was checked against the archive manifest, and shared v1 source hashes matched the existing checkout. Original Alex files are preserved in [v2/provenance/alex_source](v2/provenance/alex_source/), alongside the [import record](v2/provenance/source.json). Maintained scripts have repository-relative paths; training and scoring mathematics were retained.
