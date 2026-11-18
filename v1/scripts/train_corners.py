"""Train Koor's first stage using Point2Roof's corner-detection network."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader, Dataset


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "Point2Roof"))

from model.pointnet2 import PointNet2


class RoofDataset(Dataset):
    def __init__(self, list_path, training, seed):
        self.paths = [
            Path(line.strip())
            for line in list_path.read_text().splitlines()
            if line.strip()
        ]
        self.training = training
        self.seed = seed
        assert self.paths, f"Empty split: {list_path}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        folder = self.paths[index]
        points = np.loadtxt(folder / "points.xyz", ndmin=2)

        vertices = []
        for line in (folder / "polygon.obj").read_text().splitlines():
            fields = line.split()
            if fields and fields[0] == "v":
                vertices.append([float(v) for v in fields[1:4]])
        vertices = np.asarray(vertices, dtype=np.float64)

        # Preserve proportions with one shared scale for XYZ.
        lower, upper = points.min(0), points.max(0)
        center = (lower + upper) / 2
        scale = float((upper - lower).max())
        if scale <= 0:
            raise ValueError(f"Degenerate point cloud: {folder}")

        points = (points - center) / scale + 0.5
        vertices = (vertices - center) / scale + 0.5

        # Validation always uses the same sampled points.
        rng = (
            np.random if self.training
            else np.random.default_rng(self.seed + index)
        )
        if len(points) >= 1024:
            indices = rng.choice(len(points), 1024, replace=False)
        else:
            indices = np.concatenate([
                np.arange(len(points)),
                rng.choice(len(points), 1024 - len(points), replace=True),
            ])
            rng.shuffle(indices)

        points = points[indices]
        if self.training:
            noise = rng.normal(0, 0.005, points.shape)
            points = points + np.clip(noise, -0.02, 0.02)

        return (
            torch.from_numpy(points.astype(np.float32)),
            torch.from_numpy(vertices.astype(np.float32)),
        )


def collate(samples):
    points, vertices = zip(*samples)
    padded = torch.full((len(samples), max(map(len, vertices)), 3), -10.0)
    for index, item in enumerate(vertices):
        padded[index, :len(item)] = item
    return {"points": torch.stack(points), "vectors": padded}


def worker_seed(_):
    np.random.seed(torch.initial_seed() % (2**32))


def corner_loss(model):
    # Reuse upstream training losses without its diagnostic accuracy division,
    # which can divide by zero if a batch has no positive corner labels.
    values = model.train_dict
    classification = model.get_cls_loss(
        values["cls_pred"], values["cls_label"],
        model.loss_weight["cls_weight"],
    )
    regression = model.get_reg_loss(
        values["offset_pred"], values["offset_label"], values["cls_label"],
        model.loss_weight["reg_weight"],
    )
    return classification + regression


def run_epoch(model, loader, optimizer, max_batches):
    training = optimizer is not None

    # Upstream builds loss targets only when model.training is True.
    model.train()
    if not training:
        # Compute validation loss without changing batch-norm statistics
        # or applying dropout.
        for module in model.modules():
            if isinstance(module, (
                torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d, torch.nn.Dropout,
                torch.nn.Dropout2d, torch.nn.Dropout3d,
            )):
                module.eval()

    total_loss, total_samples = 0.0, 0
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(loader):
            if max_batches and step >= max_batches:
                break

            batch = {
                key: value.cuda(non_blocking=True)
                for key, value in batch.items()
            }
            if training:
                optimizer.zero_grad(set_to_none=True)

            model(batch)
            loss = corner_loss(model)
            if not torch.isfinite(loss).all():
                raise RuntimeError(f"Non-finite loss at batch {step}")

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=10,
                    error_if_nonfinite=True,
                )
                optimizer.step()

            count = batch["points"].shape[0]
            total_loss += loss.item() * count
            total_samples += count

            if training and step % 100 == 0:
                print(f"  batch {step}: loss={loss.item():.5f}", flush=True)

    if total_samples == 0:
        raise RuntimeError("No samples were processed")
    return total_loss / total_samples


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    assert args.epochs > 0 and args.batch_size > 0
    assert args.workers >= 0 and args.max_batches >= 0
    assert torch.cuda.is_available(), "CUDA is unavailable"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(0)

    with (PROJECT / "Point2Roof/model_cfg.yaml").open() as file:
        config = EasyDict(yaml.safe_load(file))

    model = PointNet2(config.MODEL.PointNet2).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5,
    )

    prepared = PROJECT / "data/prepared"
    train_data = RoofDataset(prepared / "train.txt", True, args.seed)
    val_data = RoofDataset(prepared / "val.txt", False, args.seed)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.run_dir / "last.pt"
    start_epoch, best_loss, history = 0, float("inf"), []

    if args.resume:
        # Load only checkpoints generated by this script.
        checkpoint = torch.load(
            last_path, map_location="cpu", weights_only=False,
        )
        if checkpoint["arguments"] != {
            **vars(args), "run_dir": str(args.run_dir), "resume": False
        }:
            raise ValueError("Resume arguments must match the original run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_val_loss"]
        history = checkpoint["history"]
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
    elif last_path.exists():
        raise FileExistsError("Use --resume or choose a new run directory")

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print(f"Train: {len(train_data)} | Validation: {len(val_data)}")
    print("Stage: corner detection | Precision: float32", flush=True)

    for epoch in range(start_epoch, args.epochs):
        # Epoch-based seeds also make loader ordering reproducible on resume.
        generator = torch.Generator().manual_seed(args.seed + epoch)
        common = dict(
            batch_size=args.batch_size, num_workers=args.workers,
            collate_fn=collate, pin_memory=True, worker_init_fn=worker_seed,
        )
        train_loader = DataLoader(
            train_data, shuffle=True, generator=generator, **common,
        )
        val_loader = DataLoader(
            val_data, shuffle=False,
            generator=torch.Generator().manual_seed(args.seed), **common,
        )

        train_loss = run_epoch(
            model, train_loader, optimizer, args.max_batches,
        )
        val_loss = run_epoch(model, val_loader, None, args.max_batches)
        scheduler.step()

        improved = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        state = {
            "stage": "corners",
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_loss,
            "history": history,
            "arguments": {
                **vars(args), "run_dir": str(args.run_dir), "resume": False
            },
            "model_config": config,
            "preprocessing": {
                "points": 1024,
                "normalization": "full-cloud bbox center; max XYZ span; +0.5",
                "training_noise_sigma": 0.005,
            },
            "torch_version": torch.__version__,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "numpy_rng": np.random.get_state(),
        }
        save_checkpoint(last_path, state)
        if improved:
            save_checkpoint(args.run_dir / "best.pt", state)

        (args.run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )
        print(
            f"Epoch {epoch + 1}: train={train_loss:.5f}, "
            f"val={val_loss:.5f}, best={best_loss:.5f}",
            flush=True,
        )

    print("Completed. Checkpoints:", args.run_dir, flush=True)


if __name__ == "__main__":
    main()