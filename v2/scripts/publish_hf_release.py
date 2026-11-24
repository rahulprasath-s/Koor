"""Upload only the verified v2 release payload; optionally make it public.

Uses stored Hugging Face login. No token is accepted on the command line.
New repositories are private until every uploaded file is checked by hash.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

PROJECT = Path(__file__).resolve().parents[1]
if PROJECT.name == "v2":
    PROJECT = PROJECT.parent
MODEL_NAME = "Koor-building-reconstruction-v2"
ROOT_FILES = {
    "model.safetensors", "config.json", "preprocessing.json", "training_metadata.json",
    "environment.json", "requirements.txt", "requirements-build.txt", "README.md", "LICENSING.md",
    "inference.py", "install_cuda_extension.py", "verify_release.py", "export_report.json",
    ".gitignore", ".gitattributes",
}
REQUIRED_FILES = ROOT_FILES | {
    "Point2Roof/LICENSE", "Point2Roof/model/roofnet.py", "Point2Roof/model/pointnet2.py",
    "Point2Roof/pc_util/setup.py", "examples/demo_roof.xyz", "examples/README.md",
    "evaluation/v1-synthetic-test.json", "evaluation/v2-synthetic-test.json",
    "evaluation/v1-building3d-holdout.json", "evaluation/v2-building3d-holdout.json",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def allowed_path(name):
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
        return False
    if name in ROOT_FILES:
        return True
    if path.parts[0] == "Point2Roof":
        return path.suffix in {".py", ".cpp", ".cu", ".h", ".cuh", ".hpp", ".md"} or name == "Point2Roof/LICENSE"
    if path.parts[0] == "evaluation":
        return len(path.parts) == 2 and path.suffix == ".json"
    return name in {"examples/demo_roof.xyz", "examples/README.md"}


def release_payload(directory):
    root = directory.resolve()
    manifest_data = (root / "SHA256SUMS.json").read_bytes()
    manifest = json.loads(manifest_data)
    if not REQUIRED_FILES.issubset(manifest):
        raise ValueError(f"Incomplete release manifest: {sorted(REQUIRED_FILES - set(manifest))}")
    payload = {}
    for name, expected in manifest.items():
        if not allowed_path(name):
            raise ValueError(f"Unexpected release file: {name}")
        path = (root / name).resolve()
        path.relative_to(root)
        data = path.read_bytes()
        if digest(data) != expected:
            raise ValueError(f"Changed release file: {name}")
        if name != "model.safetensors":
            text = data.decode("utf-8")
            if re.search(r"-----BEGIN [A-Z ]*PRIVATE KEY-----|\bhf_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b", text):
                raise ValueError(f"Possible credential content: {name}")
        payload[name] = data
    payload["SHA256SUMS.json"] = manifest_data
    for name in ("release_verification.json", "release_verification_clean.json"):
        data = (root / name).read_bytes()
        report = json.loads(data)
        if report.get("package_manifest_sha256") != digest(manifest_data):
            raise ValueError(f"Verification is for a different package: {name}")
        for check in ("file_hashes", "strict_weight_loading", "cuda_inference"):
            if report.get(check) != "passed":
                raise ValueError(f"Verification failed/missing: {name}/{check}")
        if name.endswith("_clean.json"):
            if report.get("environment_isolated") is not True or report.get("extension_built_from_package") is not True:
                raise ValueError("Clean-environment rebuild verification is missing")
        if re.search(r"-----BEGIN [A-Z ]*PRIVATE KEY-----|\bhf_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b", data.decode("utf-8")):
            raise ValueError(f"Possible credential content: {name}")
        payload[name] = data
    config = json.loads(payload["config.json"])
    metadata = json.loads(payload["training_metadata.json"])
    exported = json.loads(payload["export_report.json"])
    if config["model_name"] != MODEL_NAME or config["model_type"] != "koor_point2roof":
        raise ValueError("Expected the Koor v2 model package")
    if exported.get("weights_round_trip") != "passed_exact":
        raise ValueError("Exact safetensors conversion not verified")
    source = config["source_checkpoint_sha256"]
    if metadata["source_checkpoint_sha256"] != source or exported["source_checkpoint_sha256"] != source:
        raise ValueError("Export metadata checkpoint identities differ")
    if metadata["selected_full_roof_epoch"] != config["checkpoint_epoch"]:
        raise ValueError("Export epoch identities differ")
    for dataset, count in (("synthetic-test", 1600), ("building3d-holdout", 561)):
        report = json.loads(payload[f"evaluation/v2-{dataset}.json"])
        if report["checkpoint_sha256"] != source or report["checkpoint_epoch"] != config["checkpoint_epoch"] or report["samples"] != count:
            raise ValueError(f"Wrong held-out evaluation: {dataset}")
    return payload


def publish(api, download, addition, repo, payload, public):
    owner = repo.split("/", 1)[0]
    account = api.whoami()["name"]
    if account.casefold() != owner.casefold():
        raise PermissionError(f"Logged in as {account}; expected {owner}")
    api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
    current = api.repo_info(repo_id=repo, repo_type="model")
    existing = set(api.list_repo_files(repo_id=repo, repo_type="model", revision=current.sha))
    if existing - set(payload):
        raise ValueError(f"Repository has files outside this release: {sorted(existing-set(payload))}")
    if "model.safetensors" in existing:
        path = download(repo_id=repo, filename="model.safetensors", repo_type="model", revision=current.sha, token=True)
        if digest(Path(path).read_bytes()) != digest(payload["model.safetensors"]):
            raise ValueError("Repository already contains different weights; choose a new model repository")
    operations = [addition(path_in_repo=name, path_or_fileobj=data) for name, data in sorted(payload.items())]
    print(f"Uploading {len(operations)} verified files to {repo}", flush=True)
    commit = api.create_commit(repo_id=repo, repo_type="model", operations=operations,
                               commit_message="Release verified Koor v2 Building3D-adapted roof model",
                               parent_commit=current.sha)
    revision = commit.oid
    remote = set(api.list_repo_files(repo_id=repo, repo_type="model", revision=revision))
    if remote != set(payload):
        raise RuntimeError(f"Remote file list differs: missing={sorted(set(payload)-remote)}, extra={sorted(remote-set(payload))}")
    for name, data in sorted(payload.items()):
        path = download(repo_id=repo, filename=name, repo_type="model", revision=revision, token=True)
        if digest(Path(path).read_bytes()) != digest(data):
            raise RuntimeError(f"Remote file hash mismatch: {name}")
    print("PASS: every uploaded file matches the verified local package", flush=True)
    if api.repo_info(repo_id=repo, repo_type="model").sha != revision:
        raise RuntimeError("Repository changed during upload verification")
    if public:
        api.update_repo_settings(repo_id=repo, repo_type="model", private=False)
        if api.repo_info(repo_id=repo, repo_type="model").private:
            raise RuntimeError("Repository is still private")
    return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=PROJECT / "release" / MODEL_NAME)
    parser.add_argument("--repo", default="Uruva/" + MODEL_NAME)
    parser.add_argument("--public", action="store_true", help="Make the repo public after remote hash verification")
    args = parser.parse_args()
    if args.repo.count("/") != 1 or not all(args.repo.split("/")):
        parser.error("--repo must be namespace/model-name")
    payload = release_payload(args.release)
    print("PASS: package hashes, both GPU reports, and evaluated-checkpoint metadata verified", flush=True)
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    revision = publish(HfApi(token=True), hf_hub_download, CommitOperationAdd, args.repo, payload, args.public)
    report = {"repo_id": args.repo, "commit": revision, "public_requested": args.public,
              "remote_file_hashes": "passed", "files": {name: digest(data) for name, data in payload.items()}}
    report_path = args.release.resolve().parent / "koor-v2-publication.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(("Public" if args.public else "Uploaded") + f" model: https://huggingface.co/{args.repo}")
    print(f"Commit: https://huggingface.co/{args.repo}/commit/{revision}")
    print("Publication record:", report_path)


if __name__ == "__main__":
    main()
