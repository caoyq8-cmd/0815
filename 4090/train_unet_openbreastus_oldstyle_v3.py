import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_openbreastus_oldstyle import OpenBreastUSOldStyleDataset


@dataclass
class TrainConfig:
    data_root: str = r"/home/featurize/datasets/90024ebe-ceca-4e0d-aab7-55f496b4b5f5"
    output_dir: str = r"./results_unet_oldstyle_v3_48"

    batch_size: int = 8
    num_epochs: int = 50
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

    early_stopping_patience: int = 10
    min_delta: float = 1e-4

    seed: int = 42


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


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, x1_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(x1_ch, x1_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(x1_ch // 2 + skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 1, base_ch: int = 48):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_ch)              # 32
        self.down1 = Down(base_ch, base_ch * 2)                  # 64
        self.down2 = Down(base_ch * 2, base_ch * 4)              # 128
        self.down3 = Down(base_ch * 4, base_ch * 8)              # 256
        self.down4 = Down(base_ch * 8, base_ch * 8)              # 256

        self.up1 = Up(base_ch * 8, base_ch * 8, base_ch * 4)     # 256 + 256 -> 128
        self.up2 = Up(base_ch * 4, base_ch * 4, base_ch * 2)     # 128 + 128 -> 64
        self.up3 = Up(base_ch * 2, base_ch * 2, base_ch)         # 64 + 64 -> 32
        self.up4 = Up(base_ch, base_ch, base_ch)                 # 32 + 32 -> 32

        self.outc = nn.Conv2d(base_ch, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outc(x)
        return self.tanh(x)


def image_gradients(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return dx, dy


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


def save_curves(history: Dict[str, list], outdir: str):
    ensure_dir(outdir)

    def _plot(key: str, title: str, filename: str):
        if key not in history or len(history[key]) == 0:
            return
        plt.figure(figsize=(8, 5))
        plt.plot(history[key])
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel(key)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, filename), dpi=150)
        plt.close()

    _plot("train_loss", "Train Loss", "curve_train_loss.png")
    _plot("val_loss", "Val Loss", "curve_val_loss.png")
    _plot("val_psnr", "Val PSNR", "curve_val_psnr.png")
    _plot("train_l1", "Train L1", "curve_train_l1.png")
    _plot("train_grad", "Train Gradient Loss", "curve_train_grad.png")


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


def train_one_epoch(model, loader, optimizer, criterion, cfg, device):
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_grad = 0.0
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
    total_loss = 0.0
    total_psnr = 0.0
    total_mse = 0.0
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


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    ensure_dir(cfg.output_dir)
    ensure_dir(os.path.join(cfg.output_dir, "checkpoints"))

    print(f"data_root   = {cfg.data_root}")
    print(f"train dir   = {os.path.join(cfg.data_root, 'train')}")
    print(f"test dir    = {os.path.join(cfg.data_root, 'test')}")
    print(f"train exist = {os.path.isdir(os.path.join(cfg.data_root, 'train'))}")
    print(f"test exist  = {os.path.isdir(os.path.join(cfg.data_root, 'test'))}")

    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    train_set, test_set, train_loader, test_loader = build_loaders(cfg)
    print(f"train size = {len(train_set)}")
    print(f"test size  = {len(test_set)}")
    print(f"device     = {cfg.device}")

    model = UNet(in_channels=2, out_channels=1, base_ch=32).to(cfg.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"num_params = {num_params}")

    x_sanity = torch.randn(2, 2, 256, 256).to(cfg.device)
    with torch.no_grad():
        y_sanity = model(x_sanity)
    print("sanity check:", x_sanity.shape, "->", y_sanity.shape)

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

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val": best_val,
                "config": asdict(cfg),
            },
            os.path.join(cfg.output_dir, "checkpoints", "last.pth"),
        )

        improved = val_metrics["loss"] < (best_val - cfg.min_delta)
        if improved:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val": best_val,
                    "config": asdict(cfg),
                },
                os.path.join(cfg.output_dir, "checkpoints", "best.pth"),
            )
        else:
            wait += 1

        if (epoch % cfg.save_vis_every == 0) or improved or (epoch == 1):
            save_visualizations(model, test_loader, cfg, epoch, cfg.device)

        with open(os.path.join(cfg.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        save_curves(history, cfg.output_dir)

        if wait >= cfg.early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best epoch = {best_epoch}, best val_loss = {best_val:.6f}"
            )
            break

    print("Training finished.")
    print(f"Best epoch = {best_epoch}, best val_loss = {best_val:.6f}")


if __name__ == "__main__":
    main()