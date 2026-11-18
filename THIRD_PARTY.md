# Upstream work and provenance

## Point2Roof

- Repository: https://github.com/Li-Li-Whu/Point2Roof
- Recorded upstream commit: `ab3d273f1dce45e7fc2e587bd17a7b8ec2fd3aad`
- Authors: Li Li, Nan Song, Fei Sun, Xinyi Liu, Ruisheng Wang, Jian Yao, and Shapsheng Cao.
- Paper: *Point2Roof: End-to-end 3D building roof modeling from airborne LiDAR point clouds*.
- Software notice: [MIT license](v1/Point2Roof/LICENSE), preserved verbatim.

Koor adapts that architecture and custom CUDA operations. Compatibility changes remove obsolete THC references and update tensor access. The full trainer also uses singleton-safe batch normalization and global edge-pair indexing. [Source patch](v1/provenance/point2roof_changes.patch), [training settings](v1/training_metadata.json).

## Building3D

- Dataset: https://huggingface.co/datasets/Building3D/Building3D
- Project: https://building3d.ucalgary.ca/
- Authors: Ruisheng Wang, Shangfeng Huang, and Hongxin Yang.
- Paper: *Building3D: An Urban-Scale Dataset and Benchmarks for Learning Roof Structures from Point Clouds*, ICCV 2023.
- Listed data license: CC BY-NC-SA 4.0.

V1's real-data scores use a custom spatial holdout prepared from labelled Entry-level data. The official unlabelled test set was not scored. No Building3D training is claimed for v1.

## Published Koor source

The initial v1 code was recovered from the author-published release at https://huggingface.co/Uruva/Koor-building-reconstruction-v1, revision `8dac21750a0231cede813f7f8aa52d248cb57816`. Source hashes were checked against that release's manifest. Its LFS weight object was excluded from this repository. [Provenance record](v1/provenance/source.json).
