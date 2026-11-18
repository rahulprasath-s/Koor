# Koor v2 — Building3D fine-tuning

V2 is being prepared to initialize all three stages from Koor v1's best full-roof checkpoint, using fresh optimizer state and lower learning rates. No v2 fine-tuning results have been reported.

## Prepared on the university server

Building3D `Entry_level.zip` contains 5,698 labelled training pairs (eight-column point clouds and OBJ line wireframes), plus 583 unlabelled official test point clouds.

The existing preparation workflow keeps XYZ, translates in float64 using a recorded local origin, preserves wireframe vertices/edges, and splits geographically along Y. Its counts are:

| Split | Buildings |
|---|---:|
| Training | 4,538 |
| Validation | 533 |
| Labelled holdout | 561 |
| Excluded near spatial boundaries | 66 |

The buffer is 50 source-coordinate units on each side of each cut. The source CRS has not been verified. The labelled holdout is a custom partition of the supplied training data, not the official unlabelled test set. Keep it out of fine-tuning and checkpoint selection. V1 has already been evaluated on this same holdout; it remains the frozen comparison baseline for v2.

## Source transfer pending

The v2 data-preparation scripts and current patched source still need to be exported from the server using [tools/export_gpu_sources.py](../tools/export_gpu_sources.py). This directory deliberately does not contain a copied v1 trainer advertised as completed v2 training code.

After transfer, verify alignment on training samples, add explicit dataset/full-checkpoint initialization options, run a short training check, and fine-tune with a separate run directory. Raw Building3D data and checkpoints stay outside this Git repository. The dataset requires approved access and lists CC BY-NC-SA 4.0; see its original [Hub repository](https://huggingface.co/datasets/Building3D/Building3D).
