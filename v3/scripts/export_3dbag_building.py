#!/usr/bin/env python3
"""Fetch one 3DBAG building and export a GLB, Three.js module, and metadata."""

import argparse
import json
import re
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import mapbox_earcut
import numpy as np
import trimesh
from pyproj import Transformer


DEFAULT_ID = "NL.IMBAG.Pand.1655100000500568"
LOD = "2.2"
ATTRIBUTION = "© 3DBAG by tudelft3d and 3DGI"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
COPYRIGHT_URL = "https://docs.3dbag.nl/en/copyright/"


def surfaces(geometry):
    boundaries = geometry["boundaries"]
    kind = geometry["type"]
    if kind in {"MultiSurface", "CompositeSurface"}:
        yield from boundaries
    elif kind == "Solid":
        for shell in boundaries:
            yield from shell
    elif kind in {"MultiSolid", "CompositeSolid"}:
        for solid in boundaries:
            for shell in solid:
                yield from shell
    else:
        raise ValueError(f"Unsupported CityJSON geometry type: {kind}")


def triangulate(rings, vertices):
    clean = [ring[:-1] if ring and ring[0] == ring[-1] else ring for ring in rings]
    if not clean or any(len(ring) < 3 for ring in clean):
        raise ValueError("Surface contains an incomplete polygon ring")
    indices = np.concatenate(clean)
    if np.any(indices < 0) or np.any(indices >= len(vertices)):
        raise ValueError("Surface references a nonexistent vertex")
    points = vertices[indices]
    outer = points[:len(clean[0])] - points[0]
    normal = np.cross(outer, np.roll(outer, -1, axis=0)).sum(axis=0)
    if not np.any(normal):
        raise ValueError("Surface has a degenerate outer ring")
    # Project onto the plane where this face has the largest area. Earcut also
    # handles inner rings, such as holes in a roof surface.
    projected = np.delete(points - points[0], int(np.argmax(np.abs(normal))), axis=1)
    ring_ends = np.cumsum([len(ring) for ring in clean], dtype=np.uint32)
    triangles = indices[mapbox_earcut.triangulate_float64(projected, ring_ends)].reshape(-1, 3)
    if not len(triangles):
        raise ValueError("Surface triangulation produced no triangles")
    a, b, c = vertices[triangles].transpose(1, 0, 2)
    alignment = np.cross(b - a, c - a) @ normal
    if np.any(alignment == 0):
        raise ValueError("Surface triangulation produced a degenerate triangle")
    flipped = alignment < 0
    triangles[flipped] = triangles[flipped][:, [0, 2, 1]]
    return triangles


def threejs_module(vertices, triangles):
    positions = json.dumps(vertices.ravel().tolist(), separators=(",", ":"))
    indices = json.dumps(triangles.ravel().tolist(), separators=(",", ":"))
    return f"""// {ATTRIBUTION}, CC BY 4.0
// {COPYRIGHT_URL}
// {LICENSE_URL}
// Adapted by triangulating surfaces and converting to local Y-up coordinates.
export function buildBuilding(THREE) {{
  const indexed = new THREE.BufferGeometry();
  indexed.setAttribute('position', new THREE.Float32BufferAttribute({positions}, 3));
  indexed.setIndex({indices});
  const geometry = indexed.toNonIndexed();
  geometry.computeVertexNormals();
  return new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({{ color: 0xb7b7b7 }}));
}}
"""


def export_building(response, building_id, source_url, destination):
    feature = response["feature"]
    objects = feature["CityObjects"]
    building = objects[building_id]
    header = response["metadata"]
    crs = header["metadata"]["referenceSystem"]
    transform = header["transform"]
    world = np.asarray(feature["vertices"], dtype=np.float64) * transform["scale"] + transform["translate"]
    if not np.isfinite(world).all():
        raise ValueError("3DBAG returned non-finite coordinates")

    pending = [building_id]
    visited = set()
    faces = []
    while pending:
        object_id = pending.pop()
        if object_id in visited:
            continue
        visited.add(object_id)
        obj = objects[object_id]
        pending.extend(obj.get("children", []))
        for geometry in obj.get("geometry", []):
            if str(geometry.get("lod")) == LOD:
                faces.extend(surfaces(geometry))
    if not faces:
        raise ValueError(f"{building_id} has no LoD{LOD} surfaces")

    triangles = np.concatenate([triangulate(face, world) for face in faces])
    used, inverse = np.unique(triangles, return_inverse=True)
    triangles = inverse.reshape(triangles.shape)
    mesh_world = world[used]
    origin = mesh_world.min(axis=0)
    # Both glTF and Three.js use Y-up. This rotation preserves winding.
    east, north, up = (mesh_world - origin).T
    y_up = np.round(np.column_stack([east, up, -north]), decimals=6)
    mesh = trimesh.Trimesh(vertices=y_up, faces=triangles, process=False)

    footprint = [
        [world[ring, :2].tolist() for ring in face]
        for geometry in building.get("geometry", []) if str(geometry.get("lod")) == "0"
        for face in surfaces(geometry)
    ]
    anchor = (mesh_world.min(axis=0) + mesh_world.max(axis=0)) / 2
    lon, lat = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True).transform(*anchor[:2])
    attributes = building.get("attributes", {})
    floors = attributes.get("b3_bouwlagen")
    metadata = {
        "building_id": building_id,
        "location": {"latitude": lat, "longitude": lon, "method": "3D bounding box center"},
        "footprint_polygons_xy": footprint,
        "source_crs": crs,
        "source_origin_xyz_m": origin.tolist(),
        "mesh_axes": ["east", "up", "south"],
        "lod": LOD,
        "height_m": round(float(np.ptp(mesh_world[:, 2])), 6),
        "roof_type": attributes.get("b3_dak_type"),
        "floor_count": floors,
        "floor_count_source": "3DBAG estimate" if floors is not None else None,
        "mesh_path": "building.glb",
        "threejs_path": "building.mjs",
        "triangle_count": len(triangles),
        "mesh_watertight": bool(mesh.is_watertight),
        "source_url": source_url,
        "attribution": ATTRIBUTION,
        "license": "CC BY 4.0",
        "license_url": LICENSE_URL,
        "attribution_url": COPYRIGHT_URL,
        "changes": "Triangulated LoD2.2 surfaces and converted geometry to local Y-up coordinates.",
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".3dbag-") as temp:
        staging = Path(temp) / "building"
        staging.mkdir()
        (staging / "building.glb").write_bytes(mesh.export(file_type="glb"))
        (staging / "building.mjs").write_text(threejs_module(y_up, triangles), encoding="utf-8")
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        staging.rename(destination)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--building-id", default=DEFAULT_ID, help="Full NL.IMBAG.Pand building ID")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if not re.fullmatch(r"NL\.IMBAG\.Pand\.[0-9]{16}", args.building_id):
        parser.error("--building-id must look like NL.IMBAG.Pand.1655100000500568")
    if args.output.exists() or args.output.is_symlink():
        parser.error("--output must be a new directory")
    source_url = f"https://api.3dbag.nl/collections/pand/items/{args.building_id}"
    try:
        request = Request(source_url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=30) as result:
            response = json.load(result)
        metadata = export_building(response, args.building_id, source_url, args.output)
    except (OSError, TimeoutError, KeyError, ValueError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(f"Exported {metadata['building_id']} with {metadata['triangle_count']} triangles to {args.output.resolve()}")


if __name__ == "__main__":
    main()
