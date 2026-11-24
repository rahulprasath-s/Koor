"""Integrity checks for the measured v2 experiment; no GPU/dataset required."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / 'v2'

def read(path):
    return json.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class V2RecordsTests(unittest.TestCase):
    def test_archive_integrity(self):
        manifest = read(V2 / 'provenance/export_manifest.json')
        for name, expected in manifest['files'].items():
            path = V2 / ('provenance/alex_source/' + name.removeprefix('source/') if name.startswith('source/') else name)
            with self.subTest(file=name): self.assertEqual(digest(path), expected)
        for name, expected in read(V2 / 'provenance/shared_v1_source_hashes.json').items():
            with self.subTest(file=name): self.assertEqual(digest(ROOT / name), expected)

    def test_paired_evaluation_protocol_and_arithmetic(self):
        selected = read(V2 / 'evaluation/selection.json')
        for dataset, count in [('synthetic-test', 1600), ('building3d-holdout', 561)]:
            a, b = [read(V2 / f'evaluation/{m}-{dataset}/summary.json') for m in ('v1', 'v2')]
            for key in ('samples', 'test_list_sha256', 'seed', 'preprocessing', 'protocol', 'batch_size', 'environment'):
                self.assertEqual(a[key], b[key], (dataset, key))
            self.assertEqual(a['samples'], count)
            self.assertEqual(a['checkpoint_epoch'], 29)
            self.assertEqual(b['checkpoint_epoch'], 14)
            self.assertEqual(a['checkpoint_sha256'], selected['v1_adapter_sha256'])
            self.assertEqual(b['checkpoint_sha256'], selected['v2_checkpoint_sha256'])
            self.assertFalse(a['protocol']['published_protocol_reproduction'])
            comparison = read(V2 / f'evaluation/comparison-{dataset}.json')
            for tolerance in ('0.01', '0.02', '0.05'):
                for kind in ('vertex', 'edge'):
                    values = comparison['metrics_by_tolerance'][tolerance][kind]
                    self.assertEqual(values['v1_f1'], a['metrics_by_tolerance'][tolerance][kind]['f1'])
                    self.assertEqual(values['v2_f1'], b['metrics_by_tolerance'][tolerance][kind]['f1'])
                    self.assertAlmostEqual(values['absolute_change'], values['v2_f1']-values['v1_f1'], places=14)
            for name in ('empty_corner_buildings', 'empty_edge_buildings'):
                self.assertEqual(comparison[name], {'v1':a[name], 'v2':b[name]})

    def test_validation_selection_and_batch_counts(self):
        history = read(V2 / 'training/history.json')
        self.assertEqual([r['epoch'] for r in history], list(range(1,16)))
        self.assertEqual(max(history, key=lambda r:r['joint_score'])['epoch'], read(V2 / 'evaluation/selection.json')['v2_epoch'])
        self.assertFalse(any(r['retention_status']['eligible'] for r in history))
        for row in history:
            self.assertEqual(row['train']['samples'], 28800)
            self.assertEqual(row['train']['synthetic_samples'], 14400)
            self.assertEqual(row['train']['real_samples'], 14400)

    def test_split_membership_and_scored_cohorts(self):
        splits = read(V2 / 'provenance/split_ids.json')
        for split, ids in read(ROOT / 'v1/provenance/synthetic_split_ids.json').items():
            self.assertEqual(set(ids), set(splits['synthetic']['splits'][split]['sample_ids']))
        for domain, partitions in [('synthetic', ('train','val','test')), ('building3d', ('train','val','holdout'))]:
            entries = splits[domain]['splits']
            for index, split in enumerate(partitions):
                ids = entries[split]['sample_ids']
                self.assertEqual(len(ids), entries[split]['count'])
                self.assertEqual(len(ids), len(set(ids)))
                for other in partitions[index+1:]: self.assertFalse(set(ids)&set(entries[other]['sample_ids']))
            held = entries[partitions[-1]]
            dataset = 'synthetic-test' if domain=='synthetic' else 'building3d-holdout'
            report = read(V2 / f'evaluation/v2-{dataset}/summary.json')
            self.assertEqual(held['path_list_sha256'], report['test_list_sha256'])
            self.assertEqual(held['count'], report['samples'])

    def test_numerical_helpers_preserved(self):
        def definitions(path):
            tree = ast.parse(path.read_text())
            return {n.name:ast.dump(n, include_attributes=False) for n in tree.body
                    if isinstance(n,(ast.FunctionDef, ast.ClassDef)) and n.name!='main'}
        for name in ('mixed_utils.py','train_mixed_distill.py','prepare_building3d.py','prepare_synthetic.py','evaluate_v1_v2.py'):
            self.assertEqual(definitions(V2/'provenance/alex_source/scripts'/name), definitions(V2/'scripts'/name), name)

    def test_repository_root_resolution(self):
        for name in ('mixed_utils.py','prepare_building3d.py','prepare_synthetic.py'):
            path = V2/'scripts'/name
            tree = ast.parse(path.read_text())
            node = next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROJECT' for t in n.targets))
            result = eval(compile(ast.Expression(node.value),str(path),'eval'), {'Path':Path,'__file__':str(path)})
            self.assertEqual(result, ROOT)

if __name__ == '__main__':
    unittest.main()
