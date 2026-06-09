import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_cbs_aligned import CBSSparseAlignedDataset
from train_inversionnet_baseline import InversionNetBaseline
from utils_cbs import denormalize as cbs_denormalize


@dataclass
class TrainConfig:
    train_dobs_root: str = "Datasets_train/AI4Scup2_simulated_CBS_sparse/dobs_500k/train"
    train_speed_root: str = "Datasets_train/AI4Scup2_simulated_CBS_sparse/speed/train"
    eval_dobs_root: str = "Datasets_test/AI4Scup2_simulated_CBS_sparse/dobs_500k/eval"
    eval_speed_root: str = "Datasets_test/AI4Scup2_simulated_CBS_sparse/speed/eval"

    output_dir: str = "./2080Ti/ablation_runs/inversionnet_cbs_aligned"

    batch_size: int = 8
    num_epochs: int = 60
    lr: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    base_ch: int = 32
    bottleneck_blocks: int = 4
    dropout: float = 0.0

    lambda_l1: float = 1.0
    lambda_mse: float = 0.2
    use_early_stopping: bool = True
    early_stopping_patience: int = 12
    min_delta: float = 1e-4

    save_vis_every: int = 5
    max_vis_samples: int = 4
    seed: int = 42


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(data_range / math.sqrt(mse))


class LossFn(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_mse=0.2):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_mse = lambda_mse

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)
        total = self.lambda_l1 * l1 + self.lambda_mse * mse
        return {
            "total": total,
            "l1": l1.detach(),
            "mse": mse.detach(),
        }


def build_loaders(cfg: TrainConfig):
    train_set = CBSSparseAlignedDataset(
        dobs_root=cfg.train_dobs_root,
        speed_root=cfg.train_speed_root,
    )
    eval_set = CBSSparseAlignedDataset(
        dobs_root=cfg.eval_dobs_root,
        speed_root=cfg.eval_speed_root,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_set, eval_set, train_loader, eval_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        losses = criterion(pred, y)
        losses["total"].backward()
        optimizer.step()

        total_loss += losses["total"].item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mse_480 = 0.0
    total_psnr_480 = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        losses = criterion(pred, y)
        total_loss += losses["total"].item()

        pred_480 = cbs_denormalize(pred)
        gt_480 = cbs_denormalize(y)

        mse_480 = F.mse_loss(pred_480, gt_480).item()
        psnr_480 = compute_psnr(pred_480, gt_480, data_range=1595.1279 - 1408.692)

        total_mse_480 += mse_480
        total_psnr_480 += psnr_480
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "mse_480": total_mse_480 / max(n_batches, 1),
        "psnr_480": total_psnr_480 / max(n_batches, 1),
    }


@torch.no_grad()
def save_visualizations(model, loader, output_dir, epoch, device, max_vis_samples=4):
    model.eval()
    vis_dir = os.path.join(output_dir, "visuals")
    ensure_dir(vis_dir)

    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)

    pred = model(x)

    pred_480 = cbs_denormalize(pred).cpu()
    gt_480 = cbs_denormalize(y).cpu()
    x = x.cpu()

    n = min(max_vis_samples, x.shape[0])
    for i in range(n):
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
        axes[0].imshow(x[i, 0], cmap="gray")
        axes[0].set_title("input ch0")
        axes[1].imshow(x[i, 1], cmap="gray")
        axes[1].set_title("input ch1")
        axes[2].imshow(gt_480[i, 0], cmap="inferno")
        axes[2].set_title("gt 480")
        axes[3].imshow(pred_480[i, 0], cmap="inferno")
        axes[3].set_title("pred 480")
        err = pred_480[i, 0] - gt_480[i, 0]
        im = axes[4].imshow(err, cmap="bwr")
        axes[4].set_title("error")

        for ax in axes:
            ax.axis("off")

        fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"epoch_{epoch:03d}_sample_{i:02d}.png"), dpi=150)
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dobs_root", type=str, required=True)
    parser.add_argument("--train_speed_root", type=str, required=True)
    parser.add_argument("--eval_dobs_root", type=str, required=True)
    parser.add_argument("--eval_speed_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--bottleneck_blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_mse", type=float, default=0.2)

    parser.add_argument("--use_early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=12)
    parser.add_argument("--min_delta", type=float, default=1e-4)

    parser.add_argument("--save_vis_every", type=int, default=5)
    parser.add_argument("--max_vis_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        train_dobs_root=args.train_dobs_root,
        train_speed_root=args.train_speed_root,
        eval_dobs_root=args.eval_dobs_root,
        eval_speed_root=args.eval_speed_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        base_ch=args.base_ch,
        bottleneck_blocks=args.bottleneck_blocks,
        dropout=args.dropout,
        lambda_l1=args.lambda_l1,
        lambda_mse=args.lambda_mse,
        use_early_stopping=args.use_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        save_vis_every=args.save_vis_every,
        max_vis_samples=args.max_vis_samples,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)
    ensure_dir(os.path.join(cfg.output_dir, "checkpoints"))

    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    train_set, eval_set, train_loader, eval_loader = build_loaders(cfg)

    print(f"train size = {len(train_set)}")
    print(f"eval size  = {len(eval_set)}")
    print(f"device     = {cfg.device}")

    model = InversionNetBaseline(
        in_channels=2,
        out_channels=1,
        base_ch=cfg.base_ch,
        bottleneck_blocks=cfg.bottleneck_blocks,
        dropout=cfg.dropout,
    ).to(cfg.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"num_params = {num_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    criterion = LossFn(lambda_l1=cfg.lambda_l1, lambda_mse=cfg.lambda_mse)

    history = {
        "train_loss": [],
        "eval_loss": [],
        "eval_mse_480": [],
        "eval_psnr_480": [],
    }

    best_val = float("inf")
    best_epoch = -1
    wait = 0

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)
        eval_metrics = evaluate(model, eval_loader, criterion, cfg.device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["eval_loss"].append(eval_metrics["loss"])
        history["eval_mse_480"].append(eval_metrics["mse_480"])
        history["eval_psnr_480"].append(eval_metrics["psnr_480"])

        print(
            f"Epoch [{epoch:03d}/{cfg.num_epochs:03d}] | "
            f"train_loss={train_loss:.6f} | "
            f"eval_loss={eval_metrics['loss']:.6f} | "
            f"eval_mse_480={eval_metrics['mse_480']:.6f} | "
            f"eval_psnr_480={eval_metrics['psnr_480']:.4f}"
        )

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val": best_val,
            "config": asdict(cfg),
        }

        torch.save(state, os.path.join(cfg.output_dir, "checkpoints", "last.pth"))

        improved = eval_metrics["loss"] < (best_val - cfg.min_delta)
        if improved:
            best_val = eval_metrics["loss"]
            best_epoch = epoch
            wait = 0
            torch.save(state, os.path.join(cfg.output_dir, "checkpoints", "best.pth"))
        else:
            wait += 1

        if (epoch % cfg.save_vis_every == 0) or improved or (epoch == 1):
            save_visualizations(
                model=model,
                loader=eval_loader,
                output_dir=cfg.output_dir,
                epoch=epoch,
                device=cfg.device,
                max_vis_samples=cfg.max_vis_samples,
            )

        with open(os.path.join(cfg.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        if cfg.use_early_stopping and wait >= cfg.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}. Best epoch = {best_epoch}")
            break

    print("Training finished.")
    print(f"Best epoch = {best_epoch}, best eval_loss = {best_val:.6f}")


if __name__ == "__main__":
    main()