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
    output_dir: str = "./ablation_runs/inversion_refiner"

    inversion_ckpt: str = "./ablation_runs/inversionnet_b32/checkpoints/best.pth"

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

    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    min_delta: float = 1e-4

    refiner_base_ch: int = 32
    seed: int = 42


# =========================
# 通用工具
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
# 损失函数
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
# Refiner UNet
# 输入 3 通道: input_2ch + inversion_pred_norm
# 输出 1 通道残差 delta_norm
# =========================
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
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResidualRefinerUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_ch=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 8)

        self.up1 = Up(base_ch * 8, base_ch * 8, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 4, base_ch * 2)
        self.up3 = Up(base_ch * 2, base_ch * 2, base_ch)
        self.up4 = Up(base_ch, base_ch, base_ch)

        self.outc = nn.Conv2d(base_ch, out_channels, kernel_size=1)

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

        # 残差幅度不要太大，避免一开始破坏 inversion 初值
        return 0.3 * torch.tanh(x)


# =========================
# 两阶段模型
# inversion 冻结，refiner 训练
# =========================
class TwoStageModel(nn.Module):
    def __init__(self, inversion_model: nn.Module, refiner_model: nn.Module):
        super().__init__()
        self.inversion_model = inversion_model
        self.refiner_model = refiner_model

        for p in self.inversion_model.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            inv_pred_norm = self.inversion_model(x)

        refiner_input = torch.cat([x, inv_pred_norm], dim=1)
        delta_norm = self.refiner_model(refiner_input)
        final_pred_norm = (inv_pred_norm + delta_norm).clamp(-1.0, 1.0)

        return {
            "inv_pred_norm": inv_pred_norm,
            "delta_norm": delta_norm,
            "final_pred_norm": final_pred_norm,
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


def load_frozen_inversion_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    inv_cfg = ckpt["config"]

    model = InversionNetBaseline(
        in_channels=2,
        out_channels=1,
        base_ch=inv_cfg["base_ch"],
        bottleneck_blocks=inv_cfg["bottleneck_blocks"],
        dropout=inv_cfg["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# =========================
# 可视化
# =========================
@torch.no_grad()
def save_visualizations(model: nn.Module, loader: DataLoader, cfg: TrainConfig, epoch: int, device: str):
    model.eval()
    vis_dir = os.path.join(cfg.output_dir, "visuals")
    ensure_dir(vis_dir)

    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)

    outputs = model(x)
    inv_pred = denormalize_target(outputs["inv_pred_norm"], cfg.target_min, cfg.target_max).cpu()
    final_pred = denormalize_target(outputs["final_pred_norm"], cfg.target_min, cfg.target_max).cpu()
    gt = y.cpu()
    x_cpu = x.cpu()

    n = min(cfg.max_vis_samples, x.shape[0])
    for i in range(n):
        fig, axes = plt.subplots(1, 6, figsize=(20, 3.5))

        axes[0].imshow(x_cpu[i, 0], cmap="gray")
        axes[0].set_title("input ch0")

        axes[1].imshow(x_cpu[i, 1], cmap="gray")
        axes[1].set_title("input ch1")

        axes[2].imshow(gt[i, 0], cmap="magma")
        axes[2].set_title("gt")

        axes[3].imshow(inv_pred[i, 0], cmap="magma")
        axes[3].set_title("inv pred")

        axes[4].imshow(final_pred[i, 0], cmap="magma")
        axes[4].set_title("final pred")

        err = final_pred[i, 0] - gt[i, 0]
        im = axes[5].imshow(err, cmap="bwr")
        axes[5].set_title("final error")

        for ax in axes:
            ax.axis("off")

        fig.colorbar(im, ax=axes[5], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"epoch_{epoch:03d}_sample_{i:02d}.png"), dpi=150)
        plt.close(fig)


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
    _plot(["val_psnr_final", "val_psnr_inv"], "Val PSNR: final vs inversion", "curve_val_psnr.png")
    _plot(["val_mse_final", "val_mse_inv"], "Val MSE: final vs inversion", "curve_val_mse.png")


# =========================
# 训练 / 验证
# =========================
def train_one_epoch(model, loader, optimizer, criterion, cfg, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_norm = normalize_target(y, cfg.target_min, cfg.target_max)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        losses = criterion(outputs["final_pred_norm"], y_norm)
        losses["total"].backward()
        optimizer.step()

        total_loss += losses["total"].item()
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def evaluate(model, loader, criterion, cfg, device):
    model.eval()

    total_loss = 0.0
    total_mse_inv = 0.0
    total_psnr_inv = 0.0
    total_mse_final = 0.0
    total_psnr_final = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_norm = normalize_target(y, cfg.target_min, cfg.target_max)

        outputs = model(x)
        losses = criterion(outputs["final_pred_norm"], y_norm)
        total_loss += losses["total"].item()

        inv_pred = denormalize_target(outputs["inv_pred_norm"], cfg.target_min, cfg.target_max)
        final_pred = denormalize_target(outputs["final_pred_norm"], cfg.target_min, cfg.target_max)

        mse_inv = F.mse_loss(inv_pred, y).item()
        mse_final = F.mse_loss(final_pred, y).item()

        psnr_inv = compute_psnr(inv_pred, y, data_range=cfg.target_max - cfg.target_min)
        psnr_final = compute_psnr(final_pred, y, data_range=cfg.target_max - cfg.target_min)

        total_mse_inv += mse_inv
        total_psnr_inv += psnr_inv
        total_mse_final += mse_final
        total_psnr_final += psnr_final
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "mse_inv": total_mse_inv / max(n_batches, 1),
        "psnr_inv": total_psnr_inv / max(n_batches, 1),
        "mse_final": total_mse_final / max(n_batches, 1),
        "psnr_final": total_psnr_final / max(n_batches, 1),
    }


# =========================
# 参数
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--inversion_ckpt", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--refiner_base_ch", type=int, default=32)

    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_mse", type=float, default=0.2)
    parser.add_argument("--lambda_grad", type=float, default=0.1)

    parser.add_argument("--use_early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=10)
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
        inversion_ckpt=args.inversion_ckpt,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        refiner_base_ch=args.refiner_base_ch,
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

    inversion_model = load_frozen_inversion_model(cfg.inversion_ckpt, cfg.device)
    refiner_model = ResidualRefinerUNet(
        in_channels=3,
        out_channels=1,
        base_ch=cfg.refiner_base_ch,
    ).to(cfg.device)

    model = TwoStageModel(inversion_model, refiner_model).to(cfg.device)

    num_params_total = sum(p.numel() for p in model.parameters())
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"refiner_base_ch      = {cfg.refiner_base_ch}")
    print(f"num_params_total     = {num_params_total}")
    print(f"num_params_trainable = {num_params_trainable}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    criterion = CompositeLoss(
        lambda_l1=cfg.lambda_l1,
        lambda_mse=cfg.lambda_mse,
        lambda_grad=cfg.lambda_grad,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mse_inv": [],
        "val_psnr_inv": [],
        "val_mse_final": [],
        "val_psnr_final": [],
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
        history["val_mse_inv"].append(val_metrics["mse_inv"])
        history["val_psnr_inv"].append(val_metrics["psnr_inv"])
        history["val_mse_final"].append(val_metrics["mse_final"])
        history["val_psnr_final"].append(val_metrics["psnr_final"])

        print(
            f"Epoch [{epoch:03d}/{cfg.num_epochs:03d}] | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"inv_mse={val_metrics['mse_inv']:.6f} | "
            f"inv_psnr={val_metrics['psnr_inv']:.4f} | "
            f"final_mse={val_metrics['mse_final']:.6f} | "
            f"final_psnr={val_metrics['psnr_final']:.4f}"
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