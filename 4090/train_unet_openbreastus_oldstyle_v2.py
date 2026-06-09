import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_openbreastus_oldstyle import OpenBreastUSOldStyleDataset


@dataclass
class Config:
    data_root: str = r"D:\OpenBreastUS_processed_oldstyle_2x256x256_fullresize"
    save_dir: str = "./results_openbreastus_oldstyle_unet_v2"
    batch_size: int = 8
    num_epochs: int = 80
    lr: float = 2e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    normalize_input: bool = True
    target_mode: str = "global_minmax"   # none | global_minmax | zscore
    target_min: float = 1408.692
    target_max: float = 1595.1279
    target_mean: float = 1500.0
    target_std: float = 50.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    print_freq: int = 50
    vis_count: int = 4
    amp: bool = True
    scheduler_tmax: int = 80


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
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
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.up1 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up3 = Up(64, 32, 32)
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


class WrappedDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, cfg: Config):
        self.base_ds = base_ds
        self.cfg = cfg

    def __len__(self):
        return len(self.base_ds)

    def normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        if self.cfg.target_mode == "none":
            return y
        if self.cfg.target_mode == "global_minmax":
            y = 2.0 * (y - self.cfg.target_min) / (self.cfg.target_max - self.cfg.target_min) - 1.0
            return torch.clamp(y, -1.0, 1.0)
        if self.cfg.target_mode == "zscore":
            return (y - self.cfg.target_mean) / self.cfg.target_std
        raise ValueError(f"Unknown target_mode: {self.cfg.target_mode}")

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        y = self.normalize_y(y)
        return x, y


def denormalize_y(y: torch.Tensor, cfg: Config) -> torch.Tensor:
    if cfg.target_mode == "none":
        return y
    if cfg.target_mode == "global_minmax":
        return (y + 1.0) * 0.5 * (cfg.target_max - cfg.target_min) + cfg.target_min
    if cfg.target_mode == "zscore":
        return y * cfg.target_std + cfg.target_mean
    raise ValueError(f"Unknown target_mode: {cfg.target_mode}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(pred_phys: torch.Tensor, target_phys: torch.Tensor) -> Dict[str, float]:
    mse = torch.mean((pred_phys - target_phys) ** 2).item()
    mae = torch.mean(torch.abs(pred_phys - target_phys)).item()
    rmse = math.sqrt(mse)
    peak = max(target_phys.max().item() - target_phys.min().item(), 1e-8)
    psnr = 20.0 * math.log10(peak / max(rmse, 1e-8))
    return {"mse": mse, "mae": mae, "rmse": rmse, "psnr": psnr}


def save_curve(values: List[float], path: str, title: str, ylabel: str):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(values) + 1), values)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_preview(x, y_true_phys, y_pred_phys, path: str):
    x_np = x.detach().cpu().numpy()
    y_true_np = y_true_phys.detach().cpu().numpy()[0]
    y_pred_np = y_pred_phys.detach().cpu().numpy()[0]
    err = y_pred_np - y_true_np

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes[0].imshow(x_np[0], cmap="gray")
    axes[0].set_title("input ch0")
    axes[1].imshow(x_np[1], cmap="gray")
    axes[1].set_title("input ch1")
    axes[2].imshow(y_true_np, cmap="inferno")
    axes[2].set_title("gt")
    axes[3].imshow(y_pred_np, cmap="inferno")
    axes[3].set_title("pred")
    im = axes[4].imshow(err, cmap="bwr")
    axes[4].set_title("error")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def train_one_epoch(model, loader, optimizer, scaler, criterion_l1, criterion_mse, device, cfg):
    model.train()
    total_loss = 0.0

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=(cfg.amp and str(device).startswith("cuda"))):
            pred = model(x)
            loss_l1 = criterion_l1(pred, y)
            loss_mse = criterion_mse(pred, y)
            loss = loss_l1 + 0.2 * loss_mse

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if step % cfg.print_freq == 0 or step == len(loader):
            print(f"  step {step:04d}/{len(loader):04d} | loss={loss.item():.6f} | l1={loss_l1.item():.6f} | mse={loss_mse.item():.6f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, cfg, save_dir=None, epoch=None):
    model.eval()
    total_loss = 0.0
    all_metrics = {"mse": 0.0, "mae": 0.0, "rmse": 0.0, "psnr": 0.0}
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    preview_saved = 0
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)

        loss = criterion_l1(pred, y) + 0.2 * criterion_mse(pred, y)
        total_loss += loss.item()

        pred_phys = denormalize_y(pred, cfg)
        y_phys = denormalize_y(y, cfg)

        metrics = compute_metrics(pred_phys, y_phys)
        for k, v in metrics.items():
            all_metrics[k] += v

        if save_dir is not None and preview_saved < cfg.vis_count:
            for i in range(x.shape[0]):
                if preview_saved >= cfg.vis_count:
                    break
                preview_path = os.path.join(save_dir, f"epoch_{epoch:03d}_sample_{preview_saved:02d}.png")
                save_preview(x[i], y_phys[i], pred_phys[i], preview_path)
                preview_saved += 1

    num_batches = len(loader)
    out = {"loss": total_loss / num_batches}
    for k, v in all_metrics.items():
        out[k] = v / num_batches
    return out


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_metric, cfg, path):
    torch.save(
        {
            "epoch": epoch,
            "best_psnr": best_metric,
            "config": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        path,
    )


def main():
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt_dir = os.path.join(cfg.save_dir, "checkpoints")
    vis_dir = os.path.join(cfg.save_dir, "val_previews")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    with open(os.path.join(cfg.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    print(f"Using device: {cfg.device}")

    train_base = OpenBreastUSOldStyleDataset(
        root_dir=cfg.data_root,
        split="train",
        normalize_input=cfg.normalize_input,
        normalize_target=False,
    )
    test_base = OpenBreastUSOldStyleDataset(
        root_dir=cfg.data_root,
        split="test",
        normalize_input=cfg.normalize_input,
        normalize_target=False,
    )
    train_set = WrappedDataset(train_base, cfg)
    test_set = WrappedDataset(test_base, cfg)

    print(f"Train samples: {len(train_set)}")
    print(f"Test samples : {len(test_set)}")

    x0, y0 = train_set[0]
    print(f"Input shape  : {tuple(x0.shape)}")
    print(f"Target shape : {tuple(y0.shape)}")
    print(f"Input min/max: {x0.min().item():.4f}, {x0.max().item():.4f}")
    print(f"Target(min,max after norm): {y0.min().item():.4f}, {y0.max().item():.4f}")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = SmallUNet(in_channels=2, out_channels=1).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.scheduler_tmax)
    scaler = torch.amp.GradScaler(enabled=(cfg.amp and str(cfg.device).startswith("cuda")))
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_mse": [],
        "val_mae": [],
        "val_rmse": [],
        "val_psnr": [],
        "lr": [],
    }

    best_psnr = -1e9

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\n===== Epoch {epoch:03d}/{cfg.num_epochs:03d} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion_l1, criterion_mse, cfg.device, cfg)
        val_metrics = evaluate(model, test_loader, cfg.device, cfg, save_dir=vis_dir, epoch=epoch)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mse"].append(val_metrics["mse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_psnr"].append(val_metrics["psnr"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"train_loss={train_loss:.6f} | val_loss={val_metrics['loss']:.6f} | "
            f"mse={val_metrics['mse']:.4f} | mae={val_metrics['mae']:.4f} | "
            f"rmse={val_metrics['rmse']:.4f} | psnr={val_metrics['psnr']:.4f}"
        )

        with open(os.path.join(cfg.save_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_psnr, cfg, os.path.join(ckpt_dir, "last.pth"))

        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_psnr, cfg, os.path.join(ckpt_dir, "best.pth"))
            print(f"Saved new best checkpoint, best_psnr={best_psnr:.4f}")

        save_curve(history["train_loss"], os.path.join(cfg.save_dir, "curve_train_loss.png"), "Train Loss", "loss")
        save_curve(history["val_loss"], os.path.join(cfg.save_dir, "curve_val_loss.png"), "Val Loss", "loss")
        save_curve(history["val_psnr"], os.path.join(cfg.save_dir, "curve_val_psnr.png"), "Val PSNR", "PSNR")

    print("Training finished.")
    print(f"Best val PSNR = {best_psnr:.4f}")


if __name__ == "__main__":
    main()
