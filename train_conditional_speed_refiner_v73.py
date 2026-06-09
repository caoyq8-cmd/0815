import os
import re
import json
import math
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
# basic utils
# ============================================================

def numeric_sort_key(path):
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


def normalize_speed(x, vmin=1400.0, vmax=1605.0):
    mid = 0.5 * (vmin + vmax)
    half = 0.5 * (vmax - vmin)
    return (x - mid) / half


def denormalize_speed(x, vmin=1400.0, vmax=1605.0):
    mid = 0.5 * (vmin + vmax)
    half = 0.5 * (vmax - vmin)
    return x * half + mid


def compute_metrics_np(pred, target, data_range=205.0):
    diff = pred - target
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    psnr = 99.0 if mse <= 1e-12 else float(20.0 * math.log10(data_range / rmse))
    return mse, mae, rmse, psnr


def image_gradients(img):
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return dx, dy


def save_curves(history, outdir):
    def plot(keys, filename, ylabel):
        plt.figure(figsize=(7, 4))
        for k in keys:
            if k in history:
                plt.plot(history[k], label=k)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, filename), dpi=150)
        plt.close()

    plot(["train_loss", "val_loss"], "curve_loss.png", "loss")
    plot(["val_psnr", "val_cond_psnr"], "curve_psnr.png", "PSNR")
    plot(["val_mse", "val_cond_mse"], "curve_mse.png", "MSE")


def save_vis(cond, pred, target, path, vmin=1400.0, vmax=1605.0, max_n=4):
    cond = cond.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    n = min(max_n, cond.shape[0])

    for i in range(n):
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

        imgs = [
            cond[i, 0],
            target[i, 0],
            pred[i, 0],
            pred[i, 0] - target[i, 0],
        ]
        titles = ["condition", "target", "refined", "error"]
        cmaps = ["inferno", "inferno", "inferno", "bwr"]

        for ax, im, title, cmap in zip(axes, imgs, titles, cmaps):
            if title == "error":
                m = max(abs(float(im.min())), abs(float(im.max())), 1.0)
                h = ax.imshow(im, cmap=cmap, vmin=-m, vmax=m)
            else:
                h = ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(path.replace(".png", f"_sample_{i:02d}.png"), dpi=150)
        plt.close(fig)


# ============================================================
# dataset
# ============================================================

class ConditionalRefinerDataset(Dataset):
    def __init__(self, condition_root, split="train"):
        self.condition_root = Path(condition_root)
        self.split = split
        self.split_dir = self.condition_root / split

        self.files = sorted(
            list(self.split_dir.glob(f"{split}_*.npz")),
            key=numeric_sort_key,
        )

        if len(self.files) == 0:
            raise RuntimeError(f"No npz files found in {self.split_dir}")

        print(f"[Dataset] split={split}, num_files={len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        d = np.load(p)

        condition_norm = d["condition_norm"].astype(np.float32)
        target_norm = d["target_norm"].astype(np.float32)
        condition_speed = d["condition_speed"].astype(np.float32)
        target_speed = d["target_speed"].astype(np.float32)

        return {
            "condition_norm": torch.from_numpy(condition_norm),
            "target_norm": torch.from_numpy(target_norm),
            "condition_speed": torch.from_numpy(condition_speed),
            "target_speed": torch.from_numpy(target_speed),
            "index": torch.tensor(idx, dtype=torch.long),
        }


# ============================================================
# diffusion noising
# ============================================================

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    ac = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - ac[1:] / ac[:-1]
    return torch.clip(betas, 1e-5, 0.999)


class NoiseScheduler:
    def __init__(self, timesteps=1000, device="cuda"):
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        self.sqrt_ac = torch.sqrt(ac)
        self.sqrt_om = torch.sqrt(1.0 - ac)

    def q_sample(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        a = self.sqrt_ac[t][:, None, None, None]
        b = self.sqrt_om[t][:, None, None, None]
        return a * x + b * noise


# ============================================================
# model
# ============================================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half = self.dim // 2
        emb_scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb_scale)
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

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
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


class ConditionalUNet(nn.Module):
    """
    input channels:
        channel 0: noisy condition/current x_t
        channel 1: condition reconstruction
    output:
        clean target norm
    """
    def __init__(self, in_ch=2, out_ch=1, base_ch=32, time_dim=128, dropout=0.0):
        super().__init__()

        b = base_ch

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init = nn.Conv2d(in_ch, b, 3, padding=1)

        self.d1a = ResBlock(b, b, time_dim, dropout)
        self.d1b = ResBlock(b, b, time_dim, dropout)
        self.s1 = Downsample(b)

        self.d2a = ResBlock(b, 2 * b, time_dim, dropout)
        self.d2b = ResBlock(2 * b, 2 * b, time_dim, dropout)
        self.s2 = Downsample(2 * b)

        self.d3a = ResBlock(2 * b, 4 * b, time_dim, dropout)
        self.d3b = ResBlock(4 * b, 4 * b, time_dim, dropout)
        self.s3 = Downsample(4 * b)

        self.d4a = ResBlock(4 * b, 4 * b, time_dim, dropout)
        self.d4b = ResBlock(4 * b, 4 * b, time_dim, dropout)
        self.s4 = Downsample(4 * b)

        self.mid1 = ResBlock(4 * b, 4 * b, time_dim, dropout)
        self.mid2 = ResBlock(4 * b, 4 * b, time_dim, dropout)

        self.u4s = Upsample(4 * b)
        self.u4a = ResBlock(8 * b, 4 * b, time_dim, dropout)
        self.u4b = ResBlock(4 * b, 4 * b, time_dim, dropout)

        self.u3s = Upsample(4 * b)
        self.u3a = ResBlock(8 * b, 4 * b, time_dim, dropout)
        self.u3b = ResBlock(4 * b, 4 * b, time_dim, dropout)

        self.u2s = Upsample(4 * b)
        self.u2a = ResBlock(6 * b, 2 * b, time_dim, dropout)
        self.u2b = ResBlock(2 * b, 2 * b, time_dim, dropout)

        self.u1s = Upsample(2 * b)
        self.u1a = ResBlock(3 * b, b, time_dim, dropout)
        self.u1b = ResBlock(b, b, time_dim, dropout)

        self.out_norm = make_group_norm(b)
        self.out = nn.Sequential(
            nn.Conv2d(b, out_ch, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x_noisy, condition, t):
        t_emb = self.time_mlp(t)
        x = torch.cat([x_noisy, condition], dim=1)

        x = self.init(x)

        x = self.d1a(x, t_emb)
        x = self.d1b(x, t_emb)
        sk1 = x
        x = self.s1(x)

        x = self.d2a(x, t_emb)
        x = self.d2b(x, t_emb)
        sk2 = x
        x = self.s2(x)

        x = self.d3a(x, t_emb)
        x = self.d3b(x, t_emb)
        sk3 = x
        x = self.s3(x)

        x = self.d4a(x, t_emb)
        x = self.d4b(x, t_emb)
        sk4 = x
        x = self.s4(x)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        x = self.u4s(x)
        x = torch.cat([x, sk4], dim=1)
        x = self.u4a(x, t_emb)
        x = self.u4b(x, t_emb)

        x = self.u3s(x)
        x = torch.cat([x, sk3], dim=1)
        x = self.u3a(x, t_emb)
        x = self.u3b(x, t_emb)

        x = self.u2s(x)
        x = torch.cat([x, sk2], dim=1)
        x = self.u2a(x, t_emb)
        x = self.u2b(x, t_emb)

        x = self.u1s(x)
        x = torch.cat([x, sk1], dim=1)
        x = self.u1a(x, t_emb)
        x = self.u1b(x, t_emb)

        return self.out(F.silu(self.out_norm(x)))


# ============================================================
# loss / eval
# ============================================================

class CompositeLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_mse=0.2, lambda_grad=0.05, lambda_residual=0.0):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_mse = lambda_mse
        self.lambda_grad = lambda_grad
        self.lambda_residual = lambda_residual

    def forward(self, pred, target, condition=None):
        l1 = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)

        pdx, pdy = image_gradients(pred)
        tdx, tdy = image_gradients(target)
        grad = F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy)

        total = self.lambda_l1 * l1 + self.lambda_mse * mse + self.lambda_grad * grad

        residual = torch.tensor(0.0, device=pred.device)
        if self.lambda_residual > 0 and condition is not None:
            residual = F.l1_loss(pred - condition, target - condition)
            total = total + self.lambda_residual * residual

        return {
            "total": total,
            "l1": l1.detach(),
            "mse": mse.detach(),
            "grad": grad.detach(),
            "residual": residual.detach(),
        }


@torch.no_grad()
def evaluate(model, loader, scheduler, criterion, args, device, save_visual=False, epoch=0):
    model.eval()

    pred_all = []
    target_all = []
    cond_all = []

    losses = []

    for batch_idx, batch in enumerate(loader):
        condition_norm = batch["condition_norm"].to(device)
        target_norm = batch["target_norm"].to(device)
        condition_speed = batch["condition_speed"].to(device)
        target_speed = batch["target_speed"].to(device)

        # eval 使用 t=0，即从 coarse condition 直接 refine
        t = torch.zeros(condition_norm.shape[0], device=device, dtype=torch.long)
        pred_norm = model(condition_norm, condition_norm, t)
        pred_norm = torch.clamp(pred_norm, -1.0, 1.0)

        loss_dict = criterion(pred_norm, target_norm, condition_norm)
        losses.append(float(loss_dict["total"].detach().cpu()))

        pred_speed = denormalize_speed(pred_norm, args.speed_min, args.speed_max)
        pred_speed = torch.clamp(pred_speed, args.speed_min, args.speed_max)

        pred_all.append(pred_speed.detach().cpu().numpy())
        target_all.append(target_speed.detach().cpu().numpy())
        cond_all.append(condition_speed.detach().cpu().numpy())

        if save_visual and batch_idx == 0:
            vis_dir = os.path.join(args.output_dir, "visuals")
            os.makedirs(vis_dir, exist_ok=True)
            save_vis(
                condition_speed,
                pred_speed,
                target_speed,
                os.path.join(vis_dir, f"epoch_{epoch:03d}.png"),
                vmin=args.speed_min,
                vmax=args.speed_max,
                max_n=args.max_vis_samples,
            )

    pred_all = np.concatenate(pred_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)
    cond_all = np.concatenate(cond_all, axis=0)

    mse, mae, rmse, psnr = compute_metrics_np(pred_all, target_all, data_range=args.speed_max - args.speed_min)
    cond_mse, cond_mae, cond_rmse, cond_psnr = compute_metrics_np(cond_all, target_all, data_range=args.speed_max - args.speed_min)

    return {
        "loss": float(np.mean(losses)),
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "cond_mse": cond_mse,
        "cond_mae": cond_mae,
        "cond_rmse": cond_rmse,
        "cond_psnr": cond_psnr,
    }


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visuals"), exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    train_set = ConditionalRefinerDataset(args.condition_root, "train")
    test_set = ConditionalRefinerDataset(args.condition_root, "test")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = ConditionalUNet(
        in_ch=2,
        out_ch=1,
        base_ch=args.base_ch,
        time_dim=args.time_dim,
        dropout=args.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters = {num_params / 1e6:.2f}M")

    scheduler_noise = NoiseScheduler(args.timesteps, device=device)
    criterion = CompositeLoss(
        lambda_l1=args.lambda_l1,
        lambda_mse=args.lambda_mse,
        lambda_grad=args.lambda_grad,
        lambda_residual=args.lambda_residual,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    best_val = float("inf")
    best_epoch = -1
    wait = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mse": [],
        "val_psnr": [],
        "val_cond_mse": [],
        "val_cond_psnr": [],
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for step, batch in enumerate(train_loader, start=1):
            condition_norm = batch["condition_norm"].to(device)
            target_norm = batch["target_norm"].to(device)

            # 训练时：对 condition 加噪声，要求模型恢复 clean target
            t = torch.randint(
                low=args.t_min,
                high=args.t_max + 1,
                size=(condition_norm.shape[0],),
                device=device,
                dtype=torch.long,
            )

            noisy_cond = scheduler_noise.q_sample(condition_norm, t)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                pred_norm = model(noisy_cond, condition_norm, t)
                loss_dict = criterion(pred_norm, target_norm, condition_norm)
                loss = loss_dict["total"]

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.detach().cpu()))

            if step % args.log_every == 0:
                print(
                    f"Epoch [{epoch:03d}/{args.epochs:03d}] "
                    f"Step [{step:04d}/{len(train_loader):04d}] "
                    f"loss={np.mean(train_losses[-args.log_every:]):.6f}"
                )

        lr_scheduler.step()

        train_loss = float(np.mean(train_losses))
        val_metrics = evaluate(
            model,
            test_loader,
            scheduler_noise,
            criterion,
            args,
            device,
            save_visual=(epoch == 1 or epoch % args.save_vis_every == 0),
            epoch=epoch,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mse"].append(val_metrics["mse"])
        history["val_psnr"].append(val_metrics["psnr"])
        history["val_cond_mse"].append(val_metrics["cond_mse"])
        history["val_cond_psnr"].append(val_metrics["cond_psnr"])

        print("=" * 80)
        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_mse={val_metrics['mse']:.4f} "
            f"val_psnr={val_metrics['psnr']:.4f} | "
            f"cond_mse={val_metrics['cond_mse']:.4f} "
            f"cond_psnr={val_metrics['cond_psnr']:.4f}"
        )
        print("=" * 80)

        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        save_curves(history, args.output_dir)

        state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "best_val": best_val,
            "val_metrics": val_metrics,
        }

        torch.save(state, os.path.join(args.output_dir, "checkpoints", "latest.pth"))

        improved = val_metrics["mse"] < best_val - args.min_delta
        if improved:
            best_val = val_metrics["mse"]
            best_epoch = epoch
            wait = 0
            torch.save(state, os.path.join(args.output_dir, "checkpoints", "best.pth"))

            with open(os.path.join(args.output_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "best_val_mse": best_val,
                        "metrics": val_metrics,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(f"Saved best checkpoint at epoch {epoch}, val_mse={best_val:.4f}")
        else:
            wait += 1

        if args.use_early_stopping and wait >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch}, best_mse={best_val:.4f}")
            break

    print("[Done] training finished.")
    print("best_epoch =", best_epoch)
    print("best_val_mse =", best_val)


@torch.no_grad()
def eval_only(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", vars(args))

    base_ch = ckpt_args.get("base_ch", args.base_ch)
    time_dim = ckpt_args.get("time_dim", args.time_dim)
    dropout = ckpt_args.get("dropout", args.dropout)

    model = ConditionalUNet(
        in_ch=2,
        out_ch=1,
        base_ch=base_ch,
        time_dim=time_dim,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    test_set = ConditionalRefinerDataset(args.condition_root, "test")
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    scheduler_noise = NoiseScheduler(args.timesteps, device=device)
    criterion = CompositeLoss(
        lambda_l1=args.lambda_l1,
        lambda_mse=args.lambda_mse,
        lambda_grad=args.lambda_grad,
        lambda_residual=args.lambda_residual,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    metrics = evaluate(
        model,
        test_loader,
        scheduler_noise,
        criterion,
        args,
        device,
        save_visual=True,
        epoch=0,
    )

    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])

    parser.add_argument("--condition_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default="")

    parser.add_argument("--speed_min", type=float, default=1400.0)
    parser.add_argument("--speed_max", type=float, default=1605.0)

    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--t_min", type=int, default=0)
    parser.add_argument("--t_max", type=int, default=500)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_mse", type=float, default=0.2)
    parser.add_argument("--lambda_grad", type=float, default=0.05)
    parser.add_argument("--lambda_residual", type=float, default=0.1)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_vis_every", type=int, default=5)
    parser.add_argument("--max_vis_samples", type=int, default=4)

    parser.add_argument("--use_early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=1e-4)

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        if args.ckpt_path == "":
            raise ValueError("--mode eval 需要指定 --ckpt_path")
        eval_only(args)


if __name__ == "__main__":
    main()