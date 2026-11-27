---
license: cc-by-4.0
tags:
  - aerial-imagery
  - 3d-reconstruction
  - threejs
---

# Koor v3 Dutch building geometry pilot

This shard pairs 1,000 Dutch buildings by location. A row includes a BAG building ID, an aerial image, a footprint mask, a reference GLB, a Three.js module, and metadata. Roof type and floor count are source estimates where available. An unknown count is recorded as missing.

The geographic split contains 797 train, 94 validation and 99 test buildings, with 10 buffer exclusions. A separate manifest identifies the 782 training examples that fit the pilot's 8,192-token limit. Crops can contain nearby structures, and aerial acquisition and geometry vintages can differ. Pairing checks confirm georeferencing and file consistency. The pilot's coverage is a limited region in the Netherlands.

## Sources and license

- © 3DBAG by tudelft3d and 3DGI, CC BY 4.0. Koor triangulated the LoD2.2 building surfaces, expressed the meshes in local Y-up coordinates, rasterized footprint masks, and exported GLB, Three.js and integer-millimetre JSON geometry. The training audit also repaired one isolated collinear face in the JSON targets. [3DBAG copyright and attribution](https://docs.3dbag.nl/en/copyright/)
- Samenwerkingsverband Beeldmateriaal via PDOK, 2024 high-resolution RGB orthophotos, WMTS layer `2024_orthoHR`, CC BY 4.0. Koor assembled tiles into building-centred crops and recompressed them as JPEG. [PDOK's 2024 imagery terms](https://www.pdok.nl/-/nu-hoge-resolutie-luchtfoto-2024-bij-pdok)

Retain the source credits, license links, and descriptions of modifications when redistributing the dataset. Each example's metadata carries its source URL and imagery provenance. [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
