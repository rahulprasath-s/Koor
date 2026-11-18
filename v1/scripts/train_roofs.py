"""Train Koor's corner, roof-refinement, and edge-prediction stages."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from train_corners import (
    PROJECT, RoofDataset, collate, corner_loss, save_checkpoint, worker_seed,
)
from model.roofnet import RoofNet


class SafeBatchNorm1d(torch.nn.BatchNorm1d):
    """Use running statistics when a graph contains only one feature row."""

    def forward(self, x):
        if self.training and x.numel() // x.shape[1] <= 1:
            return F.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias, training=False,
                momentum=self.momentum, eps=self.eps,
            )
        return super().forward(x)


def replace_batchnorm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm1d):
            replacement = SafeBatchNorm1d(
                child.num_features, child.eps, child.momentum,
                child.affine, child.track_running_stats,
            )
            replacement.load_state_dict(child.state_dict())
            setattr(module, name, replacement)
        else:
            replace_batchnorm(child)


def read_edges(path):
    lines = [line.split("#")[0].split() for line in path.read_text().splitlines()]
    vertex_count = sum(bool(line) and line[0] == "v" for line in lines)
    vertices_seen, edges = 0, set()

    for fields in lines:
        if not fields:
            continue
        if fields[0] == "v":
            vertices_seen += 1
            continue
        if fields[0] not in {"f", "l"}:
            continue

        indices = []
        for token in fields[1:]:
            number = int(token.split("/")[0])
            if number == 0:
                raise ValueError(f"Invalid OBJ index zero: {path}")
            index = number - 1 if number > 0 else vertices_seen + number
            if not 0 <= index < vertex_count:
                raise ValueError(f"Invalid OBJ index: {path}")
            indices.append(index)

        pairs = list(zip(indices, indices[1:]))
        if fields[0] == "f" and len(indices) > 2:
            pairs.append((indices[-1], indices[0]))
        edges.update(tuple(sorted((a, b))) for a, b in pairs if a != b)

    if not edges:
        raise ValueError(f"No edges found: {path}")
    return torch.tensor(sorted(edges), dtype=torch.float32)


class GraphDataset(RoofDataset):
    def __init__(self, path, training, seed):
        super().__init__(path, training, seed)
        self.edges = [
            read_edges(folder / "polygon.obj") for folder in self.paths
        ]

    def __getitem__(self, index):
        points, vertices = super().__getitem__(index)
        return points, vertices, self.edges[index]


def graph_collate(samples):
    batch = collate([(points, vertices) for points, vertices, _ in samples])
    edges = torch.full(
        (len(samples), max(len(sample[2]) for sample in samples), 2), -10.0
    )
    for index, sample in enumerate(samples):
        edges[index, :len(sample[2])] = sample[2]
    batch["edges"] = edges
    batch["batch_size"] = len(samples)
    return batch


def freeze_statistics(module):
    for child in module.modules():
        if isinstance(child, (
            torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d, torch.nn.Dropout,
            torch.nn.Dropout2d, torch.nn.Dropout3d,
        )):
            child.eval()


def edge_forward(model, batch):
    """Build pairs with global indices, including singleton-building cases."""
    head = model.edge_att_net
    features = batch["keypoint_features"]
    building_ids = batch["keypoint"][:, 0].long()
    matches = batch["matches"].detach().cpu().tolist()

    pairs, labels = [], []
    for building in range(batch["batch_size"]):
        rows = torch.where(building_ids == building)[0]
        if len(rows) < 2:
            continue

        building_pairs = torch.combinations(rows, r=2)
        truth = {
            tuple(map(int, edge))
            for edge in batch["edges"][building].detach().cpu().tolist()
            if min(edge) >= 0
        }
        for a, b in building_pairs.detach().cpu().tolist():
            matched_edge = tuple(sorted((matches[a], matches[b])))
            labels.append(
                min(matched_edge) >= 0 and matched_edge in truth
            )
        pairs.append(building_pairs)

    if not pairs:
        return features.sum() * 0, None, None

    pairs = torch.cat(pairs)
    targets = features.new_tensor(labels)
    pair_features = head.att_layer(
        features[pairs[:, 0]], features[pairs[:, 1]]
    )
    logits = head.cls_fc(head.drop(head.shared_fc(pair_features)))
    loss = head.get_cls_loss(
        logits, targets, head.loss_weight["cls_weight"]
    )
    return loss, pairs, logits.sigmoid().flatten()


def forward_losses(model, batch):
    batch = model.keypoint_det_net(batch)
    losses = {"corner": corner_loss(model.keypoint_det_net)}

    try:
        batch = model.cluster_refine_net(batch)
    except SystemExit:
        # Upstream exits when a batch has no usable predicted corners.
        # Retain corner supervision and count all missing predictions in F1.
        zero = losses["corner"] * 0
        losses.update(refine=zero, edge=zero)
        return losses, None, None, None

    head = model.cluster_refine_net
    values = head.train_dict
    losses["refine"] = head.get_reg_loss(
        values["keypoint_offset_pred"],
        values["keypoint_offset_label"],
        values["keypoint_cls_label"],
        head.loss_weight["reg_weight"],
    )
    losses["edge"], pairs, scores = edge_forward(model, batch)
    return losses, batch, pairs, scores


def metric_counts(batch, prediction, pairs, scores, tolerance):
    # Counts: matched vertices, predicted vertices, true vertices,
    #         matched edges, predicted edges, true edges, empty buildings.
    counts = np.zeros(7, dtype=np.int64)
    if prediction is not None:
        vertices = prediction["refined_keypoint"].detach().cpu().numpy()
        ids = prediction["keypoint"][:, 0].long().cpu().numpy()
        if not np.isfinite(vertices).all():
            raise RuntimeError("Non-finite predicted vertices")
    else:
        vertices = np.empty((0, 3))
        ids = np.empty(0, dtype=np.int64)

    if pairs is not None:
        pairs = pairs.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()

    for building in range(batch["batch_size"]):
        truth_vertices = batch["vectors"][building].detach().cpu().numpy()
        truth_vertices = truth_vertices[
            (truth_vertices != -10).any(axis=1)
        ]
        truth_edges = {
            tuple(map(int, edge))
            for edge in batch["edges"][building].detach().cpu().tolist()
            if min(edge) >= 0
        }

        rows = np.flatnonzero(ids == building)
        counts[1] += len(rows)
        counts[2] += len(truth_vertices)
        counts[5] += len(truth_edges)
        counts[6] += len(rows) == 0
        mapping = {}

        if len(rows):
            distances = np.linalg.norm(
                vertices[rows, None, :] - truth_vertices[None, :, :],
                axis=-1,
            )
            predicted, target = linear_sum_assignment(distances)
            for a, b in zip(predicted, target):
                if distances[a, b] <= tolerance:
                    mapping[int(rows[a])] = int(b)
            counts[0] += len(mapping)

        if pairs is not None:
            selected = (
                (ids[pairs[:, 0]] == building) & (scores >= 0.5)
            )
            predicted_edges = pairs[selected]
            counts[4] += len(predicted_edges)
            for a, b in predicted_edges:
                if int(a) in mapping and int(b) in mapping:
                    edge = tuple(sorted((mapping[int(a)], mapping[int(b)])))
                    counts[3] += edge in truth_edges

    return counts


def run_epoch(model, loader, optimizer, freeze_corners, max_batches, tolerance):
    training = optimizer is not None
    model.train()  # Upstream target generation depends on this flag.
    if not training:
        freeze_statistics(model)
    elif freeze_corners:
        freeze_statistics(model.keypoint_det_net)

    totals = dict(corner=0.0, refine=0.0, edge=0.0)
    counts = np.zeros(7, dtype=np.int64)
    samples, empty_batches = 0, 0

    with torch.set_grad_enabled(training):
        for step, batch in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            batch = {
                key: value.cuda(non_blocking=True)
                if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            if training:
                optimizer.zero_grad(set_to_none=True)

            losses, prediction, pairs, scores = forward_losses(model, batch)
            loss = sum(losses.values())
            if not torch.isfinite(loss).all():
                raise RuntimeError(f"Non-finite loss at batch {step}")

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 10, error_if_nonfinite=True
                )
                optimizer.step()
            else:
                counts += metric_counts(
                    batch, prediction, pairs, scores, tolerance
                )

            size = batch["batch_size"]
            samples += size
            empty_batches += prediction is None
            for key, value in losses.items():
                totals[key] += value.item() * size

            if training and step % 100 == 0:
                print(
                    f"  batch {step}: "
                    + ", ".join(f"{k}={v.item():.4f}" for k, v in losses.items()),
                    flush=True,
                )

    if not samples:
        raise RuntimeError("No samples processed")

    result = {key: value / samples for key, value in totals.items()}
    result["total"] = sum(result.values())
    result["empty_proposal_batches"] = empty_batches

    if not training:
        vertex_f1 = 2 * counts[0] / max(1, counts[1] + counts[2])
        edge_f1 = 2 * counts[3] / max(1, counts[4] + counts[5])
        result.update(
            vertex_f1=float(vertex_f1),
            edge_f1=float(edge_f1),
            score=float(
                2 * vertex_f1 * edge_f1
                / max(1e-12, vertex_f1 + edge_f1)
            ),
            empty_buildings=int(counts[6]),
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--corner-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--freeze-corner-epochs", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--corner-checkpoint", type=Path,
        default=PROJECT / "runs/koor-corners-v1/best.pt",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available()
    assert args.epochs > 0 and args.batch_size > 0
    assert args.workers >= 0 and args.max_batches >= 0
    assert args.freeze_corner_epochs >= 0 and args.tolerance > 0

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(0)

    prepared = PROJECT / "data/prepared"
    fingerprint = hashlib.sha256(
        (prepared / "manifest.json").read_bytes()
        + (prepared / "train.txt").read_bytes()
        + (prepared / "val.txt").read_bytes()
    ).hexdigest()

    with (PROJECT / "Point2Roof/model_cfg.yaml").open() as file:
        config = EasyDict(yaml.safe_load(file))
    model = RoofNet(copy.deepcopy(config.MODEL))
    replace_batchnorm(model.cluster_refine_net)
    replace_batchnorm(model.edge_att_net)
    model.cuda()

    optimizer = torch.optim.Adam([
        {
            "params": model.keypoint_det_net.parameters(),
            "lr": args.corner_lr,
        },
        {
            "params": list(model.cluster_refine_net.parameters())
            + list(model.edge_att_net.parameters()),
            "lr": args.head_lr,
        },
    ], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5
    )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.run_dir / "last.pt"
    start_epoch, best_score, history = 0, -1.0, []
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    if args.resume:
        # Only load your own generated checkpoints.
        checkpoint = torch.load(
            last_path, map_location="cpu", weights_only=False
        )
        assert checkpoint["dataset_fingerprint"] == fingerprint
        assert checkpoint["model_config"] == config
        for key in (
            "batch_size", "corner_lr", "head_lr", "freeze_corner_epochs",
            "tolerance", "seed", "max_batches",
        ):
            assert checkpoint["settings"][key] == settings[key], key

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        best_score = checkpoint["best_score"]
        history = checkpoint["history"]
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
    else:
        if last_path.exists():
            raise FileExistsError("Use --resume or a new run directory")
        checkpoint = torch.load(
            args.corner_checkpoint, map_location="cpu", weights_only=False
        )
        assert checkpoint["stage"] == "corners"
        model.keypoint_det_net.load_state_dict(
            checkpoint["model"], strict=True
        )

    train_data = GraphDataset(prepared / "train.txt", True, args.seed)
    val_data = GraphDataset(prepared / "val.txt", False, args.seed)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print(f"Train: {len(train_data)} | Validation: {len(val_data)}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        frozen = epoch < args.freeze_corner_epochs
        for parameter in model.keypoint_det_net.parameters():
            parameter.requires_grad_(not frozen)

        common = dict(
            batch_size=args.batch_size, num_workers=args.workers,
            collate_fn=graph_collate, pin_memory=True,
            worker_init_fn=worker_seed,
        )
        train_loader = DataLoader(
            train_data, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + epoch),
            **common,
        )
        val_loader = DataLoader(
            val_data, shuffle=False,
            generator=torch.Generator().manual_seed(args.seed),
            **common,
        )

        print(f"\nEpoch {epoch + 1}: corners frozen={frozen}", flush=True)
        train = run_epoch(
            model, train_loader, optimizer, frozen,
            args.max_batches, args.tolerance,
        )
        validation = run_epoch(
            model, val_loader, None, False,
            args.max_batches, args.tolerance,
        )
        scheduler.step()

        improved = validation["score"] > best_score
        best_score = max(best_score, validation["score"])
        history.append({
            "epoch": epoch + 1, "train": train, "validation": validation
        })
        state = {
            "stage": "full_roof",
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_score": best_score,
            "history": history,
            "settings": settings,
            "model_config": config,
            "preprocessing": checkpoint["preprocessing"],
            "dataset_fingerprint": fingerprint,
            "torch_version": torch.__version__,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "implementation": {
                "singleton_batchnorm_fallback": True,
                "global_edge_pair_indices": True,
                "edge_threshold": 0.5,
            },
        }
        save_checkpoint(last_path, state)
        if improved:
            save_checkpoint(args.run_dir / "best.pt", state)
        (args.run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )
        print(
            f"train={train['total']:.4f} | "
            f"val={validation['total']:.4f} | "
            f"vertex F1={validation['vertex_f1']:.4f} | "
            f"edge F1={validation['edge_f1']:.4f} | "
            f"empty buildings={validation['empty_buildings']}",
            flush=True,
        )

    print("Completed. Checkpoints:", args.run_dir, flush=True)


if __name__ == "__main__":
    main()