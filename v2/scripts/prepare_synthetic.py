"""Prepare the downloaded synthetic data using Koor v1's published split code.

Run v2/scripts/prepare_synthetic.py from the repository root. Checks split membership against
v1/provenance/synthetic_split_ids.json before publishing any prepared output.
Only paths change for Alex; the original grouping and seed remain unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile


PROJECT = Path(__file__).resolve().parents[2]


def load_preparer(path):
    spec = importlib.util.spec_from_file_location("koor_v1_preparer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load preparation helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT / "data/RoofReconstructionDataset/RoofReconstructionDataset/SyntheticDataset")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/prepared")
    parser.add_argument("--expected-source-train", type=int, default=16000)
    parser.add_argument("--expected-source-test", type=int, default=1600)
    args = parser.parse_args()
    source, output = args.source.expanduser().resolve(), args.output.expanduser().resolve()
    project = PROJECT.resolve()
    source.relative_to(project)  # Published helper stores paths relative to PROJECT.
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    reference = project / "v1/provenance/synthetic_split_ids.json"
    expected = json.loads(reference.read_text())
    if set(expected) != {"train", "val", "test"}:
        raise ValueError("Unexpected published split-ID format")
    for name, extracted_name, count in (
        ("train.txt", "simulateRoof", args.expected_source_train),
        ("test.txt", "simulateRoofTest", args.expected_source_test),
    ):
        ids = [Path(line.strip()).name for line in (source / name).read_text().splitlines() if line.strip()]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError(f"Expected {count} unique entries in {name}; found {len(ids)}")
        folders = [path.parent for path in (source / extracted_name).rglob("points.xyz")]
        if len(folders) != count or {folder.name for folder in folders} != set(ids):
            raise ValueError(f"Incomplete or ambiguous {extracted_name} extraction: {len(folders)} point clouds")
        if not all((folder / "polygon.obj").is_file() for folder in folders):
            raise ValueError(f"Missing polygon.obj files in {extracted_name}")
        print(f"Found {count} extracted pairs: {extracted_name}", flush=True)

    helper = load_preparer(project / "v1/scripts/prepare_dataset.py")
    helper.PROJECT = project
    helper.SOURCE = source
    helper.SEED = 42
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    helper.OUTPUT = staging
    try:
        helper.main()
        manifest = json.loads((staging / "manifest.json").read_text())
        for split in ("train", "val", "test"):
            ids = [row["sample_id"] for row in manifest["samples"] if row["split"] == split]
            if len(ids) != len(set(ids)) or set(ids) != set(expected[split]):
                missing = sorted(set(expected[split]) - set(ids))[:5]
                extra = sorted(set(ids) - set(expected[split]))[:5]
                raise ValueError(f"{split} differs from published v1 split IDs: missing={missing}, extra={extra}")
            paths = [Path(line) for line in (staging / f"{split}.txt").read_text().splitlines() if line.strip()]
            if len(paths) != len(ids) or not all((path / "points.xyz").is_file() and (path / "polygon.obj").is_file() for path in paths):
                raise ValueError(f"Invalid prepared {split} paths")
            print(f"PASS: {split} membership matches published v1 ({len(ids)} samples)", flush=True)
        manifest["published_v1_split_ids_match"] = True
        manifest["published_split_ids_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
        manifest["preparation_helper_sha256"] = hashlib.sha256((project / "v1/scripts/prepare_dataset.py").read_bytes()).hexdigest()
        manifest["wrapper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest["source_directory"] = str(source)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print("\nPASS: synthetic data ready; original v1 split membership preserved")
    print("Final output:", output)


if __name__ == "__main__":
    main()
