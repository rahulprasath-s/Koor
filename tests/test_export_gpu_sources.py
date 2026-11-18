"""Checks for the server-source transfer boundary; standard library only."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exporter", ROOT / "tools/export_gpu_sources.py")
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        for version in ("v1",):
            project = self.base / f"Koor-building-reconstruction-{version}"
            (project / "scripts").mkdir(parents=True)
            (project / "scripts/train.py").write_text("print('fixture')\n")
            (project / "Point2Roof/pc_util/src").mkdir(parents=True)
            (project / "Point2Roof/pc_util/src/kernel.cu").write_text("// fixture\n")
            (project / "Point2Roof/LICENSE").write_text("fixture license\n")
        self.project = self.base / "Koor-building-reconstruction-v1"

    def tearDown(self):
        self.temporary.cleanup()

    def run_export(self, output):
        with patch.object(sys, "argv", ["export", "--base-dir", str(self.base), "--output", str(output)]):
            EXPORTER.main()

    def test_archive_whitelist_and_content_hashes(self):
        for name in ("data/raw.xyz", "runs/model.pt", ".venv/secret.py",
                     ".hf-tools/token", "Point2Roof/pc_util/build/compiled.cpp",
                     "Point2Roof/pc_util/src/kernel.so", ".env"):
            path = self.project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("must not be exported\n")
        output = self.base / "source.zip"
        self.run_export(output)
        with ZipFile(output) as archive:
            names = set(archive.namelist())
            self.assertIn("v1/scripts/train.py", names)
            self.assertTrue(all(name.startswith("v1/") or name == "manifest.json" for name in names))
            self.assertIn("v1/Point2Roof/LICENSE", names)
            self.assertFalse(any(".venv" in name or "/data/" in name or "/runs/" in name or "/build/" in name for name in names))
            self.assertFalse(any(name.endswith((".pt", ".so")) for name in names))
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(names - {"manifest.json"}, {row["path"] for row in manifest["files"]})
            for row in manifest["files"]:
                content = archive.read(row["path"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), row["sha256"])

    def test_existing_archive_is_preserved(self):
        output = self.base / "existing.zip"
        output.write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            self.run_export(output)
        self.assertEqual(output.read_bytes(), b"original")

    def test_literal_credentials_stop_export(self):
        (self.project / "scripts/accidental.py").write_text("TOKEN = '" + "hf_" + "a" * 30 + "'\n")
        with self.assertRaises(ValueError):
            EXPORTER.collect(self.project, "v1")

    def test_external_symlink_is_rejected(self):
        outside = self.base / "outside.py"
        outside.write_text("print('outside')\n")
        (self.project / "scripts/linked.py").symlink_to(outside)
        with self.assertRaises(ValueError):
            EXPORTER.collect(self.project, "v1")


if __name__ == "__main__":
    unittest.main()
