import os
import re
import json
import math
import copy
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 工具函数
# ============================================================

def numeric_sort_key(filename):
    nums = re.findall(r"\d+", filename)
    return int(nums[-1]) if nums else -1


def read_mat_v73_target(path, key="target_256"):
    """
    读取 MATLAB v7.3 / HDF5 mat 文件中的 target_256。
    """
    with h5py.File(path, "r") as f:
        arr = f[key][()]
        arr = np.asarray(arr)

        # MATLAB v7.3/HDF5 常常维度反过来；target 是 256x256，转置无伤大雅
        if arr.ndim >= 2:
            arr = np.transpose(arr, axes=list(range(arr.ndim - 1, -1, -1)))

        arr = arr.astype(np.float32)
    return arr


def normalize_speed(x, vmin=1400.0, vmax=1605.0):
    """
    声速图归一化到 [-1, 1]。
    """
    mid = 0.5 * (vmin + vmax)
    half = 0.5 * (vmax - vmin)
    return (x - mid) / half


def denormalize_speed(x, vmin=1400.0, vmax=1605.0):
    """
    [-1, 1] 反归一化回声速。
    """
    mid = 0.5 * (vmin + vmax)
    half = 0.5 * (vmax - vmin)
    return x * half + mid


def save_image_grid(samples, path, title=None, vmin=1400.0, vmax=1605.0):
    """
    samples: [B, 1, H, W], numpy 或 tensor，物理声速尺度。
    """
    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()

    samples = samples[:, 0]
    n = samples.shape[0]
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    plt.figure(figsize=(3 * cols, 3 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(samples[i], cmap="inferno", vmin=vmin, vmax=vmax)
        plt.axis("off")
        plt.title(f"{i}")
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_loss_curve(history, output_dir):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.xlabel("epoch")
    plt.ylabel("noise prediction MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()


# ============================================================
# 2. Dataset
# ============================================================

class SpeedMapDataset(Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        norm_min=1400.0,
        norm_max=1605.0,
        augment=False,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.augment = augment

        self.split_dir = self.data_root / split
        self.files = sorted(
            [p for p in self.split_dir.glob(f"{split}_*.mat")],
            key=lambda p: numeric_sort_key(p.name),
        )

        if len(self.files) == 0:
            raise RuntimeError(f"No mat files found in {self.split_dir}")

        print(f"[Dataset] split={split}, num_files={len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        target = read_mat_v73_target(str(path), key="target_256")

        # [H, W] -> [1, H, W]
        x = normalize_speed(target, self.norm_min, self.norm_max)
        x = np.clip(x, -1.0, 1.0).astype(np.float32)
        x = torch.from_numpy(x).unsqueeze(0)

        if self.augment:
            # 对声速图先验训练来说，简单翻转增强是可以接受的
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[1])
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[2])

        return x


# ============================================================
# 3. DDPM 模型
# ============================================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        time: [B]
        """
        device = time.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = time[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


def make_group_norm(ch):
    g = min(8, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.0):
        super().__init__()

        self.norm1 = make_group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )

        self.norm2 = make_group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        time_add = self.time_mlp(t_emb)[:, :, None, None]
        h = h + time_add
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class SimpleDDPMUNet(nn.Module):
    """
    输入:
        x_t: [B, 1, 256, 256]
        t:   [B]
    输出:
        predicted noise: [B, 1, 256, 256]
    """

    def __init__(self, in_ch=1, base_ch=48, time_dim=192, dropout=0.0):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        b = base_ch

        self.init_conv = nn.Conv2d(in_ch, b, 3, padding=1)

        self.down1a = ResBlock(b, b, time_dim, dropout)
        self.down1b = ResBlock(b, b, time_dim, dropout)
        self.down1s = Downsample(b)

        self.down2a = ResBlock(b, 2 * b, time_dim, dropout)
        self.down2b = ResBlock(2 * b, 2 * b, time_dim, dropout)
        self.down2s = Downsample(2 * b)

        self.down3a = ResBlock(2 * b, 4 * b, time_dim, dropout)
        self.down3b = ResBlock(4 * b, 4 * b, time_dim, dropout)
        self.down3s = Downsample(4 * b)

        self.down4a = ResBlock(4 * b, 4 * b, time_dim, dropout)
        self.down4b = ResBlock(4 * b, 4 * b, time_dim, dropout)
        self.down4s = Downsample(4 * b)

        self.mid1 = ResBlock(4 * b, 4 * b, time_dim, dropout)
        self.mid2 = ResBlock(4 * b, 4 * b, time_dim, dropout)

        self.up4s = Upsample(4 * b)
        self.up4a = ResBlock(8 * b, 4 * b, time_dim, dropout)
        self.up4b = ResBlock(4 * b, 4 * b, time_dim, dropout)

        self.up3s = Upsample(4 * b)
        self.up3a = ResBlock(8 * b, 4 * b, time_dim, dropout)
        self.up3b = ResBlock(4 * b, 4 * b, time_dim, dropout)

        self.up2s = Upsample(4 * b)
        self.up2a = ResBlock(6 * b, 2 * b, time_dim, dropout)
        self.up2b = ResBlock(2 * b, 2 * b, time_dim, dropout)

        self.up1s = Upsample(2 * b)
        self.up1a = ResBlock(3 * b, b, time_dim, dropout)
        self.up1b = ResBlock(b, b, time_dim, dropout)

        self.out_norm = make_group_norm(b)
        self.out_conv = nn.Conv2d(b, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        x = self.init_conv(x)

        x = self.down1a(x, t_emb)
        x = self.down1b(x, t_emb)
        skip1 = x
        x = self.down1s(x)

        x = self.down2a(x, t_emb)
        x = self.down2b(x, t_emb)
        skip2 = x
        x = self.down2s(x)

        x = self.down3a(x, t_emb)
        x = self.down3b(x, t_emb)
        skip3 = x
        x = self.down3s(x)

        x = self.down4a(x, t_emb)
        x = self.down4b(x, t_emb)
        skip4 = x
        x = self.down4s(x)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        x = self.up4s(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.up4a(x, t_emb)
        x = self.up4b(x, t_emb)

        x = self.up3s(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.up3a(x, t_emb)
        x = self.up3b(x, t_emb)

        x = self.up2s(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up2a(x, t_emb)
        x = self.up2b(x, t_emb)

        x = self.up1s(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up1a(x, t_emb)
        x = self.up1b(x, t_emb)

        return self.out_conv(F.silu(self.out_norm(x)))


# ============================================================
# 4. Diffusion 工具
# ============================================================

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Improved DDPM cosine schedule.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-5, 0.999)


class GaussianDiffusion:
    def __init__(self, timesteps=1000, device="cuda"):
        self.timesteps = timesteps
        self.device = device

        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_ac * x_start + sqrt_om * noise

    def p_losses(self, model, x_start, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = model(x_noisy, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def ddim_sample(self, model, shape, steps=50, eta=0.0):
        """
        DDIM 少步采样。
        eta=0 为确定性 DDIM。
        """
        model.eval()
        b = shape[0]
        x = torch.randn(shape, device=self.device)

        times = torch.linspace(self.timesteps - 1, 0, steps, device=self.device).long()
        time_pairs = list(zip(times[:-1], times[1:])) + [(times[-1], torch.tensor(-1, device=self.device))]

        for t_now, t_next in time_pairs:
            t = torch.full((b,), int(t_now.item()), device=self.device, dtype=torch.long)
            pred_noise = model(x, t)

            alpha_now = self.alphas_cumprod[t_now]
            sqrt_alpha_now = torch.sqrt(alpha_now)
            sqrt_one_minus_now = torch.sqrt(1 - alpha_now)

            x0_pred = (x - sqrt_one_minus_now * pred_noise) / sqrt_alpha_now
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            if t_next < 0:
                x = x0_pred
                continue

            alpha_next = self.alphas_cumprod[t_next]

            sigma = eta * torch.sqrt(
                (1 - alpha_next) / (1 - alpha_now) * (1 - alpha_now / alpha_next)
            )
            c = torch.sqrt(torch.clamp(1 - alpha_next - sigma ** 2, min=0.0))

            noise = torch.randn_like(x) if eta > 0 else 0.0
            x = torch.sqrt(alpha_next) * x0_pred + c * pred_noise + sigma * noise

        return torch.clamp(x, -1.0, 1.0)


# ============================================================
# 5. EMA
# ============================================================

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))
            else:
                v.copy_(msd[k])


# ============================================================
# 6. 训练与采样
# ============================================================

def evaluate_val_loss(model, diffusion, loader, device, max_batches=20):
    model.eval()
    losses = []

    with torch.no_grad():
        for i, x in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device).long()
            loss = diffusion.p_losses(model, x, t)
            losses.append(float(loss.detach().cpu()))

    model.train()
    return float(np.mean(losses))


def save_checkpoint(path, model, ema, optimizer, epoch, args, best_val_loss):
    ckpt = {
        "model": model.state_dict(),
        "ema_model": ema.ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
        "best_val_loss": best_val_loss,
    }
    torch.save(ckpt, path)


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    train_set = SpeedMapDataset(
        args.data_root,
        split="train",
        norm_min=args.norm_min,
        norm_max=args.norm_max,
        augment=args.augment,
    )
    val_set = SpeedMapDataset(
        args.data_root,
        split="test",
        norm_min=args.norm_min,
        norm_max=args.norm_max,
        augment=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = SimpleDDPMUNet(
        in_ch=1,
        base_ch=args.base_ch,
        time_dim=args.time_dim,
        dropout=args.dropout,
    ).to(device)

    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)
    ema = EMA(model, decay=args.ema_decay)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters = {num_params / 1e6:.2f} M")

    best_val_loss = float("inf")
    history = []

    fixed_noise_shape = (args.num_sample_images, 1, 256, 256)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for step, x in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device).long()

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                loss = diffusion.p_losses(model, x, t)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            train_losses.append(float(loss.detach().cpu()))

            if step % args.log_every == 0:
                print(
                    f"Epoch [{epoch:03d}/{args.epochs:03d}] "
                    f"Step [{step:04d}/{len(train_loader):04d}] "
                    f"loss={np.mean(train_losses[-args.log_every:]):.6f}"
                )

        train_loss = float(np.mean(train_losses))
        val_loss = evaluate_val_loss(
            ema.ema_model,
            diffusion,
            val_loader,
            device,
            max_batches=args.val_batches,
        )

        item = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(item)

        print("=" * 80)
        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )
        print("=" * 80)

        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        save_loss_curve(history, args.output_dir)

        # 保存 latest
        save_checkpoint(
            os.path.join(args.output_dir, "checkpoints", "latest.pth"),
            model,
            ema,
            optimizer,
            epoch,
            args,
            best_val_loss,
        )

        # 保存 best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                os.path.join(args.output_dir, "checkpoints", "best.pth"),
                model,
                ema,
                optimizer,
                epoch,
                args,
                best_val_loss,
            )
            print(f"Saved best checkpoint at epoch {epoch}, val_loss={val_loss:.6f}")

        # 采样可视化
        if epoch % args.sample_every == 0 or epoch == 1:
            samples_norm = diffusion.ddim_sample(
                ema.ema_model,
                fixed_noise_shape,
                steps=args.ddim_steps,
                eta=args.ddim_eta,
            )
            samples_speed = denormalize_speed(samples_norm, args.norm_min, args.norm_max)
            save_image_grid(
                samples_speed,
                os.path.join(args.output_dir, "samples", f"epoch_{epoch:03d}.png"),
                title=f"DDIM samples epoch {epoch}",
                vmin=args.norm_min,
                vmax=args.norm_max,
            )
            print(f"Saved samples for epoch {epoch}")

    print("[Done] training finished.")


@torch.no_grad()
def sample(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})

    base_ch = args.base_ch if args.base_ch is not None else ckpt_args.get("base_ch", 48)
    time_dim = args.time_dim if args.time_dim is not None else ckpt_args.get("time_dim", 192)
    dropout = ckpt_args.get("dropout", 0.0)
    timesteps = ckpt_args.get("timesteps", args.timesteps)

    norm_min = ckpt_args.get("norm_min", args.norm_min)
    norm_max = ckpt_args.get("norm_max", args.norm_max)

    model = SimpleDDPMUNet(
        in_ch=1,
        base_ch=base_ch,
        time_dim=time_dim,
        dropout=dropout,
    ).to(device)

    if "ema_model" in ckpt:
        model.load_state_dict(ckpt["ema_model"])
        print("loaded ema_model")
    else:
        model.load_state_dict(ckpt["model"])
        print("loaded model")

    model.eval()

    diffusion = GaussianDiffusion(timesteps=timesteps, device=device)

    os.makedirs(args.output_dir, exist_ok=True)

    samples_norm = diffusion.ddim_sample(
        model,
        (args.num_sample_images, 1, 256, 256),
        steps=args.ddim_steps,
        eta=args.ddim_eta,
    )
    samples_speed = denormalize_speed(samples_norm, norm_min, norm_max)

    np.save(os.path.join(args.output_dir, "samples_norm.npy"), samples_norm.cpu().numpy())
    np.save(os.path.join(args.output_dir, "samples_speed.npy"), samples_speed.cpu().numpy())

    save_image_grid(
        samples_speed,
        os.path.join(args.output_dir, "samples_grid.png"),
        title="DDPM speed prior samples",
        vmin=norm_min,
        vmax=norm_max,
    )

    print("samples saved to:", os.path.abspath(args.output_dir))
    print("sample speed min/max/mean/std =",
          float(samples_speed.min().cpu()),
          float(samples_speed.max().cpu()),
          float(samples_speed.mean().cpu()),
          float(samples_speed.std().cpu()))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample"])

    parser.add_argument("--data_root", type=str, default="/home/featurize/datasets/90024ebe-ceca-4e0d-aab7-55f496b4b5f5")
    parser.add_argument("--output_dir", type=str, default="./ddpm_prior_runs/speed_ddpm_v73")
    parser.add_argument("--ckpt_path", type=str, default="")

    parser.add_argument("--norm_min", type=float, default=1400.0)
    parser.add_argument("--norm_max", type=float, default=1605.0)

    parser.add_argument("--base_ch", type=int, default=48)
    parser.add_argument("--time_dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--augment", action="store_true")

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--num_sample_images", type=int, default=8)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.0)

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        if args.ckpt_path == "":
            raise ValueError("--mode sample 需要指定 --ckpt_path")
        sample(args)


if __name__ == "__main__":
    main()