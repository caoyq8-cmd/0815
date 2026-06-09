import os
import sys
import math
import glob
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 路径工具
# -----------------------------
def add_repo_to_path(repo_root: str):
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


# -----------------------------
# 数据集：尽量兼容 npy / pt / png / jpg
# 输出统一为 [1, H, W] float32, 值域约到 [-1, 1]
# -----------------------------
class SpeedImageDataset(Dataset):
    def __init__(self, data_dir, image_size=256):
        self.data_dir = Path(data_dir)
        self.image_size = image_size

        exts = ["*.npy", "*.pt", "*.pth", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
        files = []
        for ext in exts:
            files.extend(sorted(self.data_dir.rglob(ext)))
        self.files = files

        if len(self.files) == 0:
            raise RuntimeError(f"在 {data_dir} 下没有找到可读取的数据文件。")

        print(f"[Dataset] 找到样本数: {len(self.files)}")
        print(f"[Dataset] 前3个样本:")
        for p in self.files[:3]:
            print("   ", p)

    def __len__(self):
        return len(self.files)

    def _load_one(self, path: Path):
        suffix = path.suffix.lower()

        if suffix == ".npy":
            arr = np.load(path)
        elif suffix in [".pt", ".pth"]:
            arr = torch.load(path, map_location="cpu")
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            else:
                arr = np.array(arr)
        else:
            img = Image.open(path).convert("L")
            arr = np.array(img)

        arr = np.array(arr, dtype=np.float32)
        arr = np.squeeze(arr)

        if arr.ndim != 2:
            raise RuntimeError(f"样本 {path} 读取后维度不是2D，而是 {arr.shape}")

    # 与 notebook / utils_cbs.normalize 对齐：
    # 先从 480x480 中裁中心 300x300，再缩放到 256x256
        if arr.shape == (480, 480):
            arr = arr[90:390, 90:390]

        img = Image.fromarray(arr)
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)

    # 固定全局归一化，不再使用每张图自己的 min/max
        X_MIN = 1408.692
        X_MAX = 1595.1279
        arr = np.clip(arr, X_MIN, X_MAX)
        arr = 2.0 * (arr - X_MIN) / (X_MAX - X_MIN) - 1.0
        arr = np.clip(arr, -1.0, 1.0)

        arr = arr[None, :, :]  # [1, H, W]
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        x = self._load_one(self.files[idx])
        return x


# -----------------------------
# EMA
# -----------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[name])
        self.backup = {}


# -----------------------------
# VP-DDPM 常用噪声注入
# 这里只做一个最小可行训练版本
# 目标：产出兼容 checkpoint，而不是追求最终效果
# -----------------------------
def make_beta_schedule(num_scales, beta_min=1e-4, beta_max=2e-2, device="cpu"):
    betas = torch.linspace(beta_min, beta_max, num_scales, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def q_sample(x0, t, noise, alphas_cumprod):
    """
    x0: [B, C, H, W]
    t : [B] in [0, T-1]
    """
    a_bar = alphas_cumprod[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--data_dir", type=str,
                        default="Datasets_test/AI4Scup2_simulated_CBS_sparse/speed/train")
    parser.add_argument("--save_dir", type=str, default="./mini_score_ckpt")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use_dataparallel", action="store_true")
    args = parser.parse_args()

    add_repo_to_path(args.repo_root)

    # 导入仓库配置与模型
    from configs.vp import AI4Scup2_ddpm_continuous
    from models import ddpm as ddpm_model

    config = AI4Scup2_ddpm_continuous.get_config()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 80)
    print("[1] 配置")
    print("=" * 80)
    print("config.model.name        =", config.model.name)
    print("config.data.image_size   =", config.data.image_size)
    print("config.data.num_channels =", config.data.num_channels)
    print("config.model.nf          =", config.model.nf)
    print("config.model.ch_mult     =", config.model.ch_mult)
    print("config.model.num_scales  =", config.model.num_scales)
    print("config.training.continuous =", config.training.continuous)
    print("device =", device)

    # 数据集
    dataset = SpeedImageDataset(
        data_dir=args.data_dir,
        image_size=config.data.image_size
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    print("=" * 80)
    print("[2] 构建模型")
    print("=" * 80)
    model = ddpm_model.DDPM(config)

    if args.use_dataparallel and torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=config.device_ids)

    model = model.to(device)
    print("model class =", model.__class__.__name__)
    n_params = sum(p.numel() for p in model.parameters())
    print("num params  =", n_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = EMA(model, decay=0.999)

    T = config.model.num_scales
    betas, alphas, alphas_cumprod = make_beta_schedule(
        num_scales=T,
        beta_min=1e-4,
        beta_max=2e-2,
        device=device
    )

    print("=" * 80)
    print("[3] 开始训练")
    print("=" * 80)

    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in loader:
            x0 = batch.to(device)   # [B,1,256,256]
            bsz = x0.shape[0]

            t = torch.randint(low=0, high=T, size=(bsz,), device=device)
            noise = torch.randn_like(x0)
            xt = q_sample(x0, t, noise, alphas_cumprod)

            # DDPM 模型通常输入的是 xt 和离散时间标签 t
            pred = model(xt, t)

            # 最小可行版本：直接预测噪声
            loss = torch.mean((pred - noise) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)

            running_loss += loss.item()
            n_batches += 1
            global_step += 1

        epoch_loss = running_loss / max(n_batches, 1)
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | train_loss = {epoch_loss:.6f}")

        # 保存最新 checkpoint
        ckpt_last = os.path.join(args.save_dir, "checkpoint_last.pth")
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "step": global_step,
            "train_loss": epoch_loss,
        }, ckpt_last)

        # 保存 EMA 版本（更推荐后续加载这个试）
        ema.apply_shadow(model)
        ckpt_ema = os.path.join(args.save_dir, "checkpoint_ema.pth")
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "step": global_step,
            "train_loss": epoch_loss,
        }, ckpt_ema)
        ema.restore(model)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt_best = os.path.join(args.save_dir, "checkpoint_best.pth")
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "train_loss": epoch_loss,
            }, ckpt_best)

    print("=" * 80)
    print("[完成]")
    print("=" * 80)
    print("训练结束。保存目录：", args.save_dir)
    print("建议优先测试：checkpoint_ema.pth 或 checkpoint_best.pth")


if __name__ == "__main__":
    main()