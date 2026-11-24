"""CPU-only report and generated-source checks for the v2 release exporter."""
import ast
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v2_export", ROOT / "v2/scripts/export_hf_release.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class V2ExportTests(unittest.TestCase):
    def setUp(self):
        self.selection = json.loads((ROOT / "v2/evaluation/selection.json").read_text())

    def test_reports_match_the_selected_checkpoint(self):
        reports = EXPORT.paired_reports(ROOT / "v2/evaluation", self.selection["v2_checkpoint_sha256"], 14)
        self.assertEqual(len(reports), 4)
        for report in reports.values():
            self.assertNotIn("/", report["checkpoint"])
            self.assertNotIn("/", report["test_list"])

    def test_other_checkpoint_or_epoch_is_rejected(self):
        with self.assertRaises(ValueError):
            EXPORT.paired_reports(ROOT / "v2/evaluation", "wrong-checkpoint", 14)
        with self.assertRaises(ValueError):
            EXPORT.paired_reports(ROOT / "v2/evaluation", self.selection["v2_checkpoint_sha256"], 15)

    def test_card_keeps_measured_tradeoff(self):
        reports = EXPORT.paired_reports(ROOT / "v2/evaluation", self.selection["v2_checkpoint_sha256"], 14)
        card = EXPORT.model_card(reports, {"epoch": 14}, "2.11.0+cu128")
        self.assertIn("0.9441 | 0.9276", card)
        self.assertIn("0.0482 | 0.3979", card)
        self.assertIn("synthetic F1s declined", card)
        self.assertIn("CC BY-NC-SA 4.0", card)
        self.assertIn("has not been assigned", card)

    def test_generated_python_parses_and_inference_matches_evaluator(self):
        helper = EXPORT.load_helpers(ROOT)
        function = helper.extract_predict_function(ROOT / "v1/scripts/evaluate_roofs.py")
        inference = helper.INFERENCE_TEMPLATE.replace("__PREDICT_GRAPHS__", function)
        tree = ast.parse(inference)
        generated = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "predict_graphs")
        self.assertEqual(ast.dump(generated, include_attributes=False), ast.dump(ast.parse(function).body[0], include_attributes=False))
        installer = helper.INSTALL_TEMPLATE.replace('default="10.0", help="B200 is 10.0"', 'default="8.6", help="A40 is 8.6; A100 is 8.0"')
        self.assertIn('default="8.6"', installer)
        ast.parse(installer)
        verify = helper.VERIFY_TEMPLATE.replace('root / "verification.json"', 'root / "release_verification.json"')
        ast.parse(verify)
        self.assertIn('root / "release_verification.json"', verify)


if __name__ == "__main__":
    unittest.main()
