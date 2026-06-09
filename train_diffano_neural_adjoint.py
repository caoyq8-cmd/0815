import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffano_neural_adjoint import (
    NeuralAdjointDataset,
    NeuralAdjointOperator,
    estimate_dobs_scale,
    list_matched_npy_pairs,
)


@dataclass
class TrainConfig:
    train_dobs_root: str = "Datasets_test/AI4Scup2_simulated_CBS_sparse/dobs_500k/train"
    train_speed_root: str = "Datasets_test/AI4Scup2_simulated_CBS_sparse/speed/train"
    eval_dobs_root: str = "Datasets_test/AI4Scup2_simulated_CBS_sparse/dobs_500k/eval"
    eval_speed_root: str = "Datasets_test/AI4Scup2_simulated_CBS_sparse/speed/eval"
    output_dir: str = "diffano_runs/neural_adjoint_sparse"
    max_train_samples: int = 512
    max_eval_samples: int = 64
    batch_size: int = 4
    num_epochs: int = 30
    lr: float = 2e-4
    weight_decay: float = 1e-4
    width: int = 64
    modes: int = 16
    depth: int = 4
    dropout: float = 0.0
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def relative_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    denom = torch.mean(target ** 2).clamp_min(1e-8)
    return mse / denom


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    train: bool,
) -> Dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0

    for speed, dobs in loader:
        speed = speed.to(device, non_blocking=True)
        dobs = dobs.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            pred = model(speed)
            mse = F.mse_loss(pred, dobs)
            rel = relative_mse(pred, dobs)
            loss = mse + 0.1 * rel

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_mse += float(mse.detach().cpu())
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "mse": total_mse / max(num_batches, 1),
    }


def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Train DIFF-ANO neural adjoint measurement operator.")
    for name, value in asdict(cfg).items():
        value_type = type(value)
        if value_type is bool:
            parser.add_argument(f"--{name}", action="store_true" if not value else "store_false")
        else:
            parser.add_argument(f"--{name}", type=value_type, default=value)
    return TrainConfig(**vars(parser.parse_args()))


def main() -> None:
    cfg = parse_args()
    ensure_dir(cfg.output_dir)
    set_seed(cfg.seed)

    train_pairs = list_matched_npy_pairs(cfg.train_dobs_root, cfg.train_speed_root)
    stats = estimate_dobs_scale(train_pairs, max_files=min(len(train_pairs), 512))

    train_set = NeuralAdjointDataset(
        cfg.train_dobs_root,
        cfg.train_speed_root,
        dobs_scale=stats.scale,
        max_samples=cfg.max_train_samples,
    )
    eval_set = NeuralAdjointDataset(
        cfg.eval_dobs_root,
        cfg.eval_speed_root,
        dobs_scale=stats.scale,
        max_samples=cfg.max_eval_samples,
    )

    measurement_shape = tuple(train_set[0][1].shape[-2:])
    model = NeuralAdjointOperator(
        measurement_shape=measurement_shape,
        width=cfg.width,
        modes=cfg.modes,
        depth=cfg.depth,
        dropout=cfg.dropout,
    ).to(cfg.device)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    history = []
    best_eval = float("inf")
    for epoch in range(1, cfg.num_epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, cfg.device, train=True)
        eval_metrics = run_epoch(model, eval_loader, optimizer, cfg.device, train=False)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "eval": eval_metrics,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"[{epoch:03d}/{cfg.num_epochs}] "
            f"train_loss={train_metrics['loss']:.6e} "
            f"eval_loss={eval_metrics['loss']:.6e}"
        )

        ckpt = {
            "model_state": model.state_dict(),
            "config": asdict(cfg),
            "dobs_scale": stats.scale,
            "measurement_shape": measurement_shape,
            "history": history,
        }
        torch.save(ckpt, os.path.join(cfg.output_dir, "last.pth"))
        if eval_metrics["loss"] < best_eval:
            best_eval = eval_metrics["loss"]
            torch.save(ckpt, os.path.join(cfg.output_dir, "best.pth"))

        with open(os.path.join(cfg.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        payload = asdict(cfg)
        payload["dobs_scale"] = stats.scale
        payload["measurement_shape"] = measurement_shape
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
