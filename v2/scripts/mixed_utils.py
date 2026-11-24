"""Balanced replay, deterministic augmentation, and target-free validation."""
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "v1/scripts"))
from train_roofs import (GraphDataset, RoofNet, forward_losses, graph_collate,
                        replace_batchnorm, save_checkpoint, worker_seed)
from evaluate_roofs import predict_graphs, score_graph, summarize, sha256_file as digest

TOLERANCES = (0.01, 0.02, 0.05)
NORMALIZATION = "full-cloud bbox center; max XYZ span; +0.5"


class BalancedBatchSampler:
    def __init__(self, synthetic_size, real_size, batch_size, seed, epoch, start=0):
        if min(synthetic_size, real_size) < 1 or batch_size < 2 or batch_size % 2:
            raise ValueError("Need nonempty datasets and an even batch size")
        self.synthetic_size, self.real_size = synthetic_size, real_size
        self.batch_size, self.seed, self.epoch, self.start = batch_size, seed, epoch, start
        self.total_batches = (max(synthetic_size, real_size) + batch_size // 2 - 1) // (batch_size // 2)
        if not 0 <= start <= self.total_batches:
            raise ValueError("Invalid resumed batch offset")

    def __len__(self):
        return self.total_batches - self.start

    @staticmethod
    def cycle(size, rng):
        while True:
            order = list(range(size))
            rng.shuffle(order)
            yield from order

    def __iter__(self):
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        synthetic = self.cycle(self.synthetic_size, rng)
        real = self.cycle(self.real_size, rng)
        for step in range(self.total_batches):
            batch = [next(synthetic) for _ in range(self.batch_size // 2)]
            batch += [self.synthetic_size + next(real) for _ in range(self.batch_size // 2)]
            rng.shuffle(batch)
            if step >= self.start:
                # Absolute draw IDs make each repeated real sample's noise unique
                # and reproducible, including after a mid-epoch restart.
                yield [(index, step * self.batch_size + position) for position, index in enumerate(batch)]


class ReplayDataset(Dataset):
    def __init__(self, synthetic, real, seed, epoch):
        self.synthetic, self.real, self.seed, self.epoch = synthetic, real, seed, epoch

    def __len__(self):
        return len(self.synthetic) + len(self.real)

    def __getitem__(self, item):
        index, draw = item
        synthetic = index < len(self.synthetic)
        dataset = self.synthetic if synthetic else self.real
        index = index if synthetic else index - len(self.synthetic)
        state = np.random.get_state()
        try:
            payload = f"{self.seed}:{self.epoch}:{draw}".encode()
            np.random.seed(int.from_bytes(hashlib.sha256(payload).digest()[:4], "little"))
            return dataset[index], synthetic
        finally:
            np.random.set_state(state)


def tagged_collate(samples):
    batch = graph_collate([sample for sample, _ in samples])
    batch["is_synthetic"] = torch.tensor([flag for _, flag in samples], dtype=torch.bool)
    return batch


def inspect_data(synthetic_dir, real_dir):
    partitions, counts = {}, {}
    fingerprint = hashlib.sha256()
    for domain, folder, held in (("synthetic", synthetic_dir, "test"), ("real", real_dir, "holdout")):
        counts[domain] = {}
        for name in ("manifest.json", "train.txt", "val.txt", held + ".txt"):
            fingerprint.update(f"{domain}/{name}\0".encode())
            fingerprint.update((folder / name).read_bytes())
        for split in ("train", "val", held):
            paths = [Path(line.strip()).resolve() for line in (folder / (split + ".txt")).read_text().splitlines() if line.strip()]
            if not paths or len(paths) != len(set(paths)):
                raise ValueError(f"Empty/duplicate paths: {domain}/{split}")
            partitions[(domain, split)] = set(paths)
            counts[domain][split] = len(paths)
    keys = list(partitions)
    for i, key in enumerate(keys):
        for other in keys[i + 1:]:
            if partitions[key] & partitions[other]:
                raise ValueError(f"Overlapping paths: {key}, {other}")
    return counts, fingerprint.hexdigest()


def harmonic(a, b):
    return 2 * a * b / max(1e-12, a + b)


def domain_score(report):
    metrics = report["metrics_by_tolerance"]["0.05"]
    return harmonic(metrics["vertex"]["f1"], metrics["edge"]["f1"])


def selection_status(current, baseline, retention):
    synthetic = all(current["synthetic"]["metrics_by_tolerance"][str(t)][k]["f1"] + 1e-12 >= retention * baseline["synthetic"]["metrics_by_tolerance"][str(t)][k]["f1"] for t in TOLERANCES for k in ("vertex", "edge"))
    real = all(current["real"]["metrics_by_tolerance"][str(t)][k]["f1"] + 1e-12 >= baseline["real"]["metrics_by_tolerance"][str(t)][k]["f1"] for t in TOLERANCES for k in ("vertex", "edge"))
    improved = domain_score(current["real"]) > domain_score(baseline["real"]) + 1e-12
    return {"synthetic_retained": synthetic, "real_no_regression": real,
            "real_improved": improved, "eligible": synthetic and real and improved}


def validate(model, dataset, args, should_stop):
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=graph_collate, pin_memory=True,
                        worker_init_fn=worker_seed, generator=torch.Generator().manual_seed(args.seed))
    records = {str(t): [] for t in TOLERANCES}
    samples = empty_corners = empty_edges = 0
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            if args.max_batches and step >= args.max_batches:
                break
            if should_stop():
                raise InterruptedError("Stop requested during validation")
            graphs = predict_graphs(model, batch["points"].cuda(non_blocking=True))
            for index, graph in enumerate(graphs):
                truth = batch["vectors"][index].numpy()
                truth = truth[(truth != -10).any(axis=1)]
                edges = {tuple(map(int, e)) for e in batch["edges"][index].tolist() if min(e) >= 0}
                accepted = graph["pairs"][graph["scores"] >= 0.5]
                for tolerance in TOLERANCES:
                    records[str(tolerance)].append(score_graph(graph["vertices"], accepted, truth, edges, tolerance))
                samples += 1
                empty_corners += len(graph["vertices"]) == 0
                empty_edges += len(accepted) == 0
    if not samples:
        raise RuntimeError("No validation samples")
    if should_stop():
        raise InterruptedError("Stop requested during validation")
    return {"samples": samples, "metrics_by_tolerance": {k: summarize(v) for k, v in records.items()},
            "empty_corner_buildings": empty_corners, "empty_edge_buildings": empty_edges,
            "inference": "eval mode, points only; labels used only for scoring"}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def print_validation(label, reports):
    print(label, flush=True)
    for domain, report in reports.items():
        for tolerance in TOLERANCES:
            metric = report["metrics_by_tolerance"][str(tolerance)]
            print(f"  {domain} {tolerance:.0%}: vertex={metric['vertex']['f1']:.4f}, edge={metric['edge']['f1']:.4f}", flush=True)
        print(f"  {domain}: {report['samples']} samples; empty edges={report['empty_edge_buildings']}", flush=True)
