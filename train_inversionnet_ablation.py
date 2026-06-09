import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_openbreastus_oldstyle import OpenBreastUSOldStyleDataset
from train_inversionnet_baseline import InversionNetBaseline


# =========================
# 配置
# =========================
@dataclass
class TrainConfig:
    data_root: str = "/home/featurize/datasets/90024ebe-ceca-4e0d-aab7-55f496b4b5f5"
    output_dir: str = "./ablation_runs/inversionnet_ablation"

    batch_size: int = 8
    num_epochs: int = 60
    lr: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    save_vis_every: int = 5
    max_vis_samples: int = 4

    target_min: float = 1400.0
    target_max: float = 1600.0

    lambda_l1: float = 1.0
    lambda_mse: float = 0.2
    lambda_grad: float = 0.1

    use_early_stopping: bool = True
    early_stopping_patience: int = 12
    min_delta: float = 1e-4

    base_ch: int = 32
    bottleneck_blocks: int = 4
    dropout: float = 0.0

    seed: int = 42


# =========================
# 工具
# =========================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_target(y: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    y = (y - vmin) / (vmax - vmin)
    y = y * 2.0 - 1.0
    return y.clamp(-1.0, 1.0)


def denormalize_target(y_norm: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    y = (y_norm + 1.0) * 0.5
    return y * (vmax - vmin) + vmin


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(data_range / math.sqrt(mse))


def image_gradients(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return dx, dy


# =========================
# 损失
# =========================
class GradientLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dx, pred_dy = image_gradients(pred)
        target_dx, target_dy = image_gradients(target)
        return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


class CompositeLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_mse=0.2, lambda_grad=0.1):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_mse = lambda_mse
        self.lambda_grad = lambda_grad
        self.grad_loss = GradientLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        l1 = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)
        grad = self.grad_loss(pred, target)
        total = self.lambda_l1 * l1 + self.lambda_mse * mse + self.lambda_grad * grad
        return {
            "total": total,
            "l1": l1.detach(),
            "mse": mse.detach(),
            "grad": grad.detach(),
        }


# =========================
# 数据
# =========================
def build_loaders(cfg: TrainConfig):
    train_set = OpenBreastUSOldStyleDataset(
        root_dir=cfg.data_root,
        split="train",
        normalize_input=True,
        normalize_target=False,
    )
    test_set = OpenBreastUSOldStyleDataset(
        root_dir=cfg.data_root,
        split="test",
        normalize_input=True,
        normalize_target=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_set, test_set, train_loader, test_loader


# =========================
# 可视化
# =========================
def save_curves(history: Dict[str, list], outdir: str):
    ensure_dir(outdir)

    def _plot(keys, title, filename):
        plt.figure(figsize=(8, 5))
        for key in keys:
            if key in history and len(history[key]) > 0:
                plt.plot(history[key], label=key)
        plt.title(title)
        plt.xlabel("epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, filename), dpi=150)
        plt.close()

    _plot(["train_loss", "val_loss"], "Train / Val Loss", "curve_loss.png")
    _plot(["val_psnr"], "Val PSNR", "curve_val_psnr.png")
    _plot(["train_l1", "train_grad"], "Train L1 / Grad", "curve_train_components.png")


@torch.no_grad()
def save_visualizations(model: nn.Module, loader: DataLoader, cfg: TrainConfig, epoch: int, device: str):
    model.eval()
    vis_dir = os.path.join(cfg.output_dir, "visuals")
    ensure_dir(vis_dir)

    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)

    pred_norm = model(x)
    pred = denormalize_target(pred_norm, cfg.target_min, cfg.target_max).cpu()
    gt = y.cpu()
    x = x.cpu()

    n = min(cfg.max_vis_samples, x.shape[0])
    for i in range(n):
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
        axes[0].imshow(x[i, 0], cmap="gray")
        axes[0].set_title("input ch0")

        axes[1].imshow(x[i, 1], cmap="gray")
        axes[1].set_title("input ch1")

        axes[2].imshow(gt[i, 0], cmap="magma")
        axes[2].set_title("gt")

        axes[3].imshow(pred[i, 0], cmap="magma")
        axes[3].set_title("pred")

        err = pred[i, 0] - gt[i, 0]
        im = axes[4].imshow(err, cmap="bwr")
        axes[4].set_title("error")

        for ax in axes:
            ax.axis("off")

        fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"epoch_{epoch:03d}_sample_{i:02d}.png"), dpi=150)
        plt.close(fig)


# =========================
# 训练 / 验证
# =========================
def train_one_epoch(model, loader, optimizer, criterion, cfg, device):
    model.train()
    total_loss, total_l1, total_grad = 0.0, 0.0, 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_norm = normalize_target(y, cfg.target_min, cfg.target_max)

        optimizer.zero_grad(set_to_none=True)
        pred_norm = model(x)
        losses = criterion(pred_norm, y_norm)
        losses["total"].backward()
        optimizer.step()

        total_loss += losses["total"].item()
        total_l1 += losses["l1"].item()
        total_grad += losses["grad"].item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "l1": total_l1 / max(n_batches, 1),
        "grad": total_grad / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, cfg, device):
    model.eval()
    total_loss, total_psnr, total_mse = 0.0, 0.0, 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_norm = normalize_target(y, cfg.target_min, cfg.target_max)

        pred_norm = model(x)
        losses = criterion(pred_norm, y_norm)
        total_loss += losses["total"].item()

        pred = denormalize_target(pred_norm, cfg.target_min, cfg.target_max)
        mse = F.mse_loss(pred, y).item()
        total_mse += mse
        total_psnr += compute_psnr(pred, y, data_range=cfg.target_max - cfg.target_min)
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "mse": total_mse / max(n_batches, 1),
        "psnr": total_psnr / max(n_batches, 1),
    }


# =========================
# 参数
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
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
    parser.add_argument("--lambda_grad", type=float, default=0.1)

    parser.add_argument("--use_early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=12)
    parser.add_argument("--min_delta", type=float, default=1e-4)

    parser.add_argument("--target_min", type=float, default=1400.0)
    parser.add_argument("--target_max", type=float, default=1600.0)

    parser.add_argument("--save_vis_every", type=int, default=5)
    parser.add_argument("--max_vis_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# =========================
# 主函数
# =========================
def main():
    args = parse_args()
    cfg = TrainConfig(
        data_root=args.data_root,
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
        lambda_grad=args.lambda_grad,
        use_early_stopping=args.use_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        target_min=args.target_min,
        target_max=args.target_max,
        save_vis_every=args.save_vis_every,
        max_vis_samples=args.max_vis_samples,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)
    ensure_dir(os.path.join(cfg.output_dir, "checkpoints"))

    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    train_set, test_set, train_loader, test_loader = build_loaders(cfg)

    print(f"train size = {len(train_set)}")
    print(f"test size  = {len(test_set)}")
    print(f"device     = {cfg.device}")

    model = InversionNetBaseline(
        in_channels=2,
        out_channels=1,
        base_ch=cfg.base_ch,
        bottleneck_blocks=cfg.bottleneck_blocks,
        dropout=cfg.dropout,
    ).to(cfg.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"base_ch           = {cfg.base_ch}")
    print(f"bottleneck_blocks = {cfg.bottleneck_blocks}")
    print(f"num_params        = {num_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    criterion = CompositeLoss(
        lambda_l1=cfg.lambda_l1,
        lambda_mse=cfg.lambda_mse,
        lambda_grad=cfg.lambda_grad,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "train_l1": [],
        "train_grad": [],
    }

    best_val = float("inf")
    best_epoch = -1
    wait = 0

    for epoch in range(1, cfg.num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, cfg, cfg.device)
        val_metrics = evaluate(model, test_loader, criterion, cfg, cfg.device)
        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_psnr"].append(val_metrics["psnr"])
        history["train_l1"].append(train_metrics["l1"])
        history["train_grad"].append(train_metrics["grad"])

        print(
            f"Epoch [{epoch:03d}/{cfg.num_epochs:03d}] | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_mse={val_metrics['mse']:.6f} | "
            f"val_psnr={val_metrics['psnr']:.4f}"
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

        improved = val_metrics["loss"] < (best_val - cfg.min_delta)
        if improved:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            wait = 0
            torch.save(state, os.path.join(cfg.output_dir, "checkpoints", "best.pth"))
        else:
            wait += 1

        if (epoch % cfg.save_vis_every == 0) or improved or (epoch == 1):
            save_visualizations(model, test_loader, cfg, epoch, cfg.device)

        with open(os.path.join(cfg.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        save_curves(history, cfg.output_dir)

        if cfg.use_early_stopping and wait >= cfg.early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best epoch = {best_epoch}, best val_loss = {best_val:.6f}"
            )
            break

    print("Training finished.")
    print(f"Best epoch = {best_epoch}, best val_loss = {best_val:.6f}")


if __name__ == "__main__":
    main()