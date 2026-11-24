# Licensing status

Point2Roof's original software is included with its [MIT notice](v1/Point2Roof/LICENSE). That notice is preserved alongside the source.

I haven't chosen a license for the original Koor code or model weights yet. Making the repository public doesn't settle those reuse terms, and the upstream software license doesn't automatically cover everything in this project.

The applicable terms for the separately hosted Point2Roof synthetic training dataset still need to be confirmed before assigning a model-weight license. Raw training data and pretrained weights aren't stored in this repository.

Building3D lists [CC BY-NC-SA 4.0](https://szusic.github.io/Building3D/). V1 used it for real-data evaluation. V2 uses labelled Entry-level development splits for training and validation, with a separate spatial holdout for evaluation. Its point clouds and annotations aren't included here; split identifiers and aggregate reports are included for provenance. The listed data terms do not establish a permissive license for the combined Koor code or weights.

The small `v1/examples/demo_roof.xyz` file was generated mathematically to try the inference code. It isn't a benchmark sample. Its generation code is in the archived exporter, and its licensing status is the same as the other original Koor material above.
