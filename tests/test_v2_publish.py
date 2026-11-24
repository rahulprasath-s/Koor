"""Upload boundaries and private-to-public ordering with a fake Hub client."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v2_publish", ROOT / "v2/scripts/publish_hf_release.py")
PUBLISH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISH)


def create_fixture(root):
    files = {name: b"fixture" for name in PUBLISH.REQUIRED_FILES}
    files["config.json"] = json.dumps({"model_name": PUBLISH.MODEL_NAME, "model_type": "koor_point2roof", "source_checkpoint_sha256": "selected", "checkpoint_epoch": 14}).encode()
    files["training_metadata.json"] = json.dumps({"source_checkpoint_sha256": "selected", "selected_full_roof_epoch": 14}).encode()
    files["export_report.json"] = json.dumps({"source_checkpoint_sha256": "selected", "weights_round_trip": "passed_exact"}).encode()
    for dataset, count in (("synthetic-test", 1600), ("building3d-holdout", 561)):
        files[f"evaluation/v2-{dataset}.json"] = json.dumps({"checkpoint_sha256": "selected", "checkpoint_epoch": 14, "samples": count}).encode()
    manifest = {name: PUBLISH.digest(data) for name, data in files.items()}
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest_data = json.dumps(manifest).encode()
    (root / "SHA256SUMS.json").write_bytes(manifest_data)
    report = {"package_manifest_sha256": PUBLISH.digest(manifest_data), "file_hashes": "passed", "strict_weight_loading": "passed", "cuda_inference": "passed"}
    (root / "release_verification.json").write_text(json.dumps(report))
    report.update(environment_isolated=True, extension_built_from_package=True)
    (root / "release_verification_clean.json").write_text(json.dumps(report))


class FakeHub:
    def __init__(self, cache):
        self.cache = cache
        self.files = {}
        self.private = True
        self.sha = "initial"
        self.calls = []

    def whoami(self):
        return {"name": "Uruva"}

    def create_repo(self, **kwargs):
        assert kwargs["private"] is True
        self.calls.append("create-private")

    def repo_info(self, **kwargs):
        return SimpleNamespace(sha=self.sha, private=self.private)

    def list_repo_files(self, **kwargs):
        return list(self.files)

    def create_commit(self, **kwargs):
        assert kwargs["parent_commit"] == self.sha
        self.calls.append("commit")
        self.files.update({op.path_in_repo: op.path_or_fileobj for op in kwargs["operations"]})
        self.sha = "verified-commit"
        return SimpleNamespace(oid=self.sha)

    def download(self, **kwargs):
        self.calls.append("download")
        path = self.cache / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[kwargs["filename"]])
        return str(path)

    def update_repo_settings(self, **kwargs):
        assert kwargs["private"] is False
        self.calls.append("make-public")
        self.private = False


class PublishTests(unittest.TestCase):
    def test_only_manifest_and_bound_verification_reports_are_uploaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_fixture(root)
            (root / "do-not-upload.pt").write_bytes(b"checkpoint")
            payload = PUBLISH.release_payload(root)
            self.assertNotIn("do-not-upload.pt", payload)
            self.assertIn("release_verification_clean.json", payload)
            self.assertIn("release_verification.json", payload)

    def test_changed_file_and_outdated_verification_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_fixture(root)
            (root / "README.md").write_text("changed")
            with self.assertRaises(ValueError):
                PUBLISH.release_payload(root)
            create_fixture(root)
            (root / "release_verification_clean.json").write_text("{}")
            with self.assertRaises(ValueError):
                PUBLISH.release_payload(root)

    def test_path_boundaries(self):
        for name in ("", ".", "../token", "/tmp/token", "Point2Roof/../token.py", "data/roof.xyz", "checkpoint.pt", "Point2Roof/model/op.so", "Point2Roof//op.py"):
            self.assertFalse(PUBLISH.allowed_path(name), name)

    def test_public_only_after_all_remote_files_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = FakeHub(Path(tmp))
            payload = {"model.safetensors": b"weights", "README.md": b"model card"}
            result = PUBLISH.publish(hub, hub.download, SimpleNamespace, "Uruva/Koor-building-reconstruction-v2", payload, True)
            self.assertEqual(result, "verified-commit")
            self.assertEqual(hub.calls, ["create-private", "commit", "download", "download", "make-public"])

    def test_remote_mismatch_prevents_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = FakeHub(Path(tmp))
            def wrong_download(**kwargs):
                path = Path(tmp) / "wrong"
                path.write_bytes(b"wrong")
                return str(path)
            with self.assertRaises(RuntimeError):
                PUBLISH.publish(hub, wrong_download, SimpleNamespace, "Uruva/Koor-building-reconstruction-v2", {"README.md": b"correct"}, True)
            self.assertTrue(hub.private)
            self.assertNotIn("make-public", hub.calls)

    def test_wrong_account_and_different_existing_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = FakeHub(Path(tmp))
            hub.whoami = lambda: {"name": "SomeoneElse"}
            with self.assertRaises(PermissionError):
                PUBLISH.publish(hub, hub.download, SimpleNamespace, "Uruva/Koor-building-reconstruction-v2", {}, True)
            self.assertEqual(hub.calls, [])
            hub.whoami = lambda: {"name": "Uruva"}
            hub.files = {"model.safetensors": b"previous weights"}
            with self.assertRaises(ValueError):
                PUBLISH.publish(hub, hub.download, SimpleNamespace, "Uruva/Koor-building-reconstruction-v2", {"model.safetensors": b"different weights"}, True)
            self.assertNotIn("commit", hub.calls)
            self.assertNotIn("make-public", hub.calls)


if __name__ == "__main__":
    unittest.main()
