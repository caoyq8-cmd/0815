import os
import re
import json
import math
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def numeric_sort_key(path):
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


def denormalize_speed(x, vmin=1400.0, vmax=1605.0):
    mid = 0.5 * (vmin + vmax)
    half = 0.5 * (vmax - vmin)
    return x * half + mid


def normalize_speed(x, vmin=1400.0, vmax=1605.0):
    mid = 0.5 * (vmin + vmax)
    half = 0.5 * (vmax - vmin)
    return (x - mid) / half


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
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curve_loss.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(history["val_mse"], label="refined_mse")
    plt.plot(history["val_cond_mse"], label="condition_mse")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curve_mse.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(history["val_psnr"], label="refined_psnr")
    plt.plot(history["val_cond_psnr"], label="condition_psnr")
    plt.xlabel("epoch")
    plt.ylabel("PSNR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curve_psnr.png"), dpi=150)
    plt.close()


def save_vis(cond, pred, target, path, vmin=1400.0, vmax=1605.0, max_n=4):
    cond = cond.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    n = min(max_n, cond.shape[0])

    for i in range(n):
        fig, axes = plt.subplots(1, 5, figsize=(17, 3.5))

        residual = pred[i, 0] - cond[i, 0]
        error_before = cond[i, 0] - target[i, 0]
        error_after = pred[i, 0] - target[i, 0]

        imgs = [
            cond[i, 0],
            target[i, 0],
            pred[i, 0],
            residual,
            error_after,
        ]
        titles = ["condition", "target", "refined", "learned residual", "after error"]
        cmaps = ["inferno", "inferno", "inferno", "bwr", "bwr"]

        for ax, im, title, cmap in zip(axes, imgs, titles, cmaps):
            if cmap == "bwr":
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


class ResidualRefinerDataset(Dataset):
    def __init__(self, condition_root, split="train"):
        self.root = Path(condition_root)
        self.split = split
        self.dir = self.root / split

        self.files = sorted(
            list(self.dir.glob(f"{split}_*.npz")),
            key=numeric_sort_key,
        )

        if len(self.files) == 0:
            raise RuntimeError(f"No npz files found in {self.dir}")

        print(f"[Dataset] split={split}, num_files={len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])

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


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -scale)
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


class ResidualUNet(nn.Module):
    """
    输入:
        condition_norm + noisy_condition_norm
    输出:
        residual_norm
    最终:
        pred_norm = condition_norm + residual_scale * residual_norm
    """

    def __init__(self, in_ch=2, base_ch=32, time_dim=128, dropout=0.0, residual_scale=0.25):
        super().__init__()

        self.residual_scale = residual_scale
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

        self.mid1 = ResBlock(4 * b, 4 * b, time_dim, dropout)
        self.mid2 = ResBlock(4 * b, 4 * b, time_dim, dropout)

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
        self.out_conv = nn.Conv2d(b, 1, 3, padding=1)

        # 关键：最后一层置零，初始 pred ≈ condition
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, condition, noisy_condition, t):
        t_emb = self.time_mlp(t)

        x = torch.cat([condition, noisy_condition], dim=1)
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

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

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

        residual = torch.tanh(self.out_conv(F.silu(self.out_norm(x))))
        pred = condition + self.residual_scale * residual
        return torch.clamp(pred, -1.0, 1.0), residual


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


class ResidualLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_mse=0.2, lambda_grad=0.05, lambda_residual=0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_mse = lambda_mse
        self.lambda_grad = lambda_grad
        self.lambda_residual = lambda_residual

    def forward(self, pred, target, condition):
        l1 = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)

        pdx, pdy = image_gradients(pred)
        tdx, tdy = image_gradients(target)
        grad = F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy)

        # 直接约束“学 residual”，而不是整图重学
        res_l1 = F.l1_loss(pred - condition, target - condition)

        total = (
            self.lambda_l1 * l1
            + self.lambda_mse * mse
            + self.lambda_grad * grad
            + self.lambda_residual * res_l1
        )

        return total


@torch.no_grad()
def evaluate(model, loader, criterion, args, device, save_visual=False, epoch=0):
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

        t = torch.zeros(condition_norm.shape[0], device=device, dtype=torch.long)
        pred_norm, residual = model(condition_norm, condition_norm, t)

        loss = criterion(pred_norm, target_norm, condition_norm)
        losses.append(float(loss.detach().cpu()))

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

    mse, mae, rmse, psnr = compute_metrics_np(pred_all, target_all, args.speed_max - args.speed_min)
    cmse, cmae, crmse, cpsnr = compute_metrics_np(cond_all, target_all, args.speed_max - args.speed_min)

    return {
        "loss": float(np.mean(losses)),
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "cond_mse": cmse,
        "cond_mae": cmae,
        "cond_rmse": crmse,
        "cond_psnr": cpsnr,
    }


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visuals"), exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    train_set = ResidualRefinerDataset(args.condition_root, "train")
    test_set = ResidualRefinerDataset(args.condition_root, "test")

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

    model = ResidualUNet(
        in_ch=2,
        base_ch=args.base_ch,
        time_dim=args.time_dim,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
    ).to(device)

    print(f"model parameters = {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    noise_scheduler = NoiseScheduler(args.timesteps, device)
    criterion = ResidualLoss(
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

    best_mse = float("inf")
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
        losses = []

        for step, batch in enumerate(train_loader, start=1):
            condition_norm = batch["condition_norm"].to(device)
            target_norm = batch["target_norm"].to(device)

            # 大部分时候 t=0，少量时候加轻噪声，避免 train/eval mismatch
            if torch.rand(()) < args.zero_t_prob:
                t = torch.zeros(condition_norm.shape[0], device=device, dtype=torch.long)
                noisy_condition = condition_norm
            else:
                t = torch.randint(
                    args.t_min,
                    args.t_max + 1,
                    size=(condition_norm.shape[0],),
                    device=device,
                    dtype=torch.long,
                )
                noisy_condition = noise_scheduler.q_sample(condition_norm, t)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                pred_norm, residual = model(condition_norm, noisy_condition, t)
                loss = criterion(pred_norm, target_norm, condition_norm)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.detach().cpu()))

            if step % args.log_every == 0:
                print(
                    f"Epoch [{epoch:03d}/{args.epochs:03d}] "
                    f"Step [{step:04d}/{len(train_loader):04d}] "
                    f"loss={np.mean(losses[-args.log_every:]):.6f}"
                )

        lr_scheduler.step()

        train_loss = float(np.mean(losses))

        val = evaluate(
            model,
            test_loader,
            criterion,
            args,
            device,
            save_visual=(epoch == 1 or epoch % args.save_vis_every == 0),
            epoch=epoch,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_mse"].append(val["mse"])
        history["val_psnr"].append(val["psnr"])
        history["val_cond_mse"].append(val["cond_mse"])
        history["val_cond_psnr"].append(val["cond_psnr"])

        print("=" * 80)
        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val['loss']:.6f} "
            f"val_mse={val['mse']:.4f} "
            f"val_psnr={val['psnr']:.4f} | "
            f"cond_mse={val['cond_mse']:.4f} "
            f"cond_psnr={val['cond_psnr']:.4f}"
        )
        print("=" * 80)

        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        save_curves(history, args.output_dir)

        state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "val_metrics": val,
            "best_mse": best_mse,
        }

        torch.save(state, os.path.join(args.output_dir, "checkpoints", "latest.pth"))

        if val["mse"] < best_mse - args.min_delta:
            best_mse = val["mse"]
            best_epoch = epoch
            wait = 0

            torch.save(state, os.path.join(args.output_dir, "checkpoints", "best.pth"))

            with open(os.path.join(args.output_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "best_mse": best_mse,
                        "metrics": val,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(f"Saved best checkpoint at epoch {epoch}, val_mse={best_mse:.4f}")
        else:
            wait += 1

        if args.use_early_stopping and wait >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}, best_epoch={best_epoch}, best_mse={best_mse:.4f}")
            break

    print("[Done]")
    print("best_epoch =", best_epoch)
    print("best_mse =", best_mse)


@torch.no_grad()
def eval_only(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", vars(args))

    model = ResidualUNet(
        in_ch=2,
        base_ch=ckpt_args.get("base_ch", args.base_ch),
        time_dim=ckpt_args.get("time_dim", args.time_dim),
        dropout=ckpt_args.get("dropout", args.dropout),
        residual_scale=ckpt_args.get("residual_scale", args.residual_scale),
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    test_set = ResidualRefinerDataset(args.condition_root, "test")
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    criterion = ResidualLoss(
        lambda_l1=args.lambda_l1,
        lambda_mse=args.lambda_mse,
        lambda_grad=args.lambda_grad,
        lambda_residual=args.lambda_residual,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    val = evaluate(
        model,
        test_loader,
        criterion,
        args,
        device,
        save_visual=True,
        epoch=0,
    )

    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)

    print(json.dumps(val, indent=2, ensure_ascii=False))


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
    parser.add_argument("--residual_scale", type=float, default=0.25)

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--t_min", type=int, default=1)
    parser.add_argument("--t_max", type=int, default=100)
    parser.add_argument("--zero_t_prob", type=float, default=0.7)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_mse", type=float, default=0.2)
    parser.add_argument("--lambda_grad", type=float, default=0.05)
    parser.add_argument("--lambda_residual", type=float, default=0.5)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_vis_every", type=int, default=5)
    parser.add_argument("--max_vis_samples", type=int, default=4)

    parser.add_argument("--use_early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=20)
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