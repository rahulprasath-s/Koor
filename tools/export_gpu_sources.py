"""Export Koor source/configuration from server projects; exclude datasets/weights.

No external packages, network calls, or GPU access are required. The projects are
read-only inputs. Review the resulting archive before sharing or publishing it.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZIP_DEFLATED, ZipFile


SOURCE_SUFFIXES = {".py", ".cpp", ".cu", ".h", ".cuh", ".hpp"}
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
)
MAX_FILE_BYTES = 4 * 1024 * 1024


def collect(project, version):
    """Return ZIP member names and UTF-8 source/configuration bytes."""
    members = {}

    def add(path, relative=None, content=None):
        if path.is_symlink() or not path.resolve().is_relative_to(project):
            raise ValueError(f"Refusing a link or path outside the project: {path}")
        relative = relative or path.relative_to(project).as_posix()
        member = PurePosixPath(version, relative)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"Invalid archive path: {member}")
        if content is None:
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError(f"Unexpectedly large source file: {path}")
            content = path.read_bytes()
        if len(content) > MAX_FILE_BYTES:
            raise ValueError(f"Unexpectedly large export member: {member}")
        content.decode("utf-8")
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            raise ValueError(f"Possible literal credential/private key in {path}; review locally")
        if path.suffix == ".py":
            ast.parse(content.decode("utf-8"), filename=str(path))
        key = str(member)
        if key in members:
            raise ValueError(f"Duplicate archive member: {key}")
        members[key] = content

    scripts = project / "scripts"
    if not scripts.is_dir():
        raise FileNotFoundError(scripts)
    for path in sorted(scripts.glob("*.py")):
        add(path)

    upstream = project / "Point2Roof"
    for folder in ("model", "utils", "pc_util/src"):
        for path in sorted((upstream / folder).rglob("*")):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                if any(part in {"__pycache__", "build", ".git"}
                       or part.endswith(".egg-info") for part in path.parts):
                    continue
                add(path)
    for name in ("LICENSE", "README.md", "UPSTREAM_README.md", "model_cfg.yaml", "pc_util/setup.py"):
        path = upstream / name
        if path.is_file():
            add(path)

    for name in ("README.md", "LICENSING.md", "config.json", "preprocessing.json",
                 "requirements.txt", "requirements-build.txt"):
        path = project / name
        if path.is_file():
            add(path)

    for source_name, destination in (
        ("evaluations/koor-v1-synthetic-test/summary.json", "evaluation/synthetic_test.json"),
        ("evaluations/koor-v1-building3d-zero-shot/summary.json", "evaluation/building3d_zero_shot.json"),
        ("runs/koor-roofs-v1/history.json", "provenance/training_history.json"),
    ):
        path = project / source_name
        if path.is_file():
            report = json.loads(path.read_text())
            if isinstance(report, dict):
                for key in ("checkpoint", "test_list"):
                    if key in report:
                        report[key] = Path(report[key]).name
            content = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
            add(path, destination, content)

    if not any(name.startswith(f"{version}/scripts/") for name in members):
        raise ValueError(f"No project scripts found in {project}")
    return members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path.home())
    parser.add_argument("--output", type=Path, default=Path.home() / "Koor-source-export.zip")
    args = parser.parse_args()
    base = args.base_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new --output: {output}")

    members = {}
    projects = []
    for version in ("v1",):
        name = f"Koor-building-reconstruction-{version}"
        project = (base / name).resolve()
        if not project.is_dir():
            raise FileNotFoundError(project)
        members.update(collect(project, version))
        projects.append(name)

    manifest = {
        "format_version": 1,
        "projects": projects,
        "scope": "Selected source/configuration and summary metadata; no raw datasets or weights",
        "files": [{"path": name, "bytes": len(content),
                   "sha256": hashlib.sha256(content).hexdigest()}
                  for name, content in sorted(members.items())],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacing an earlier source snapshot.
    created = False
    try:
        handle = output.open("xb")
        created = True
        with handle, ZipFile(handle, "w", compression=ZIP_DEFLATED) as archive:
            for name, content in sorted(members.items()):
                archive.writestr(name, content)
            archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
    except BaseException:
        # Only remove the partial archive created by this invocation.
        if created and output.is_file():
            output.unlink()
        raise
    print(f"PASS: exported {len(members)} source/configuration files")
    print(f"Archive: {output} ({output.stat().st_size:,} bytes)")
    print("Project inputs were not modified. Download the ZIP using VS Code, then attach it to the chat.")


if __name__ == "__main__":
    main()
