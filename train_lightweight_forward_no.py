import os
import re
import json
import math
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# basic utils
# ============================================================

def numeric_key(path):
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


def normalize_speed_np(x, center=1500.0, scale=100.0):
    return ((x.astype(np.float32) - center) / scale).astype(np.float32)


def make_source_map_np(shape, src_xy, sigma=2.0):
    """
    shape = (H,W)
    src_xy = [row, col]，与 CBS indexing 保持一致。
    """
    h, w = shape
    rr = np.arange(h, dtype=np.float32)[:, None]
    cc = np.arange(w, dtype=np.float32)[None, :]

    r0 = float(src_xy[0])
    c0 = float(src_xy[1])

    g = np.exp(-((rr - r0) ** 2 + (cc - c0) ** 2) / (2.0 * sigma ** 2))
    g = g.astype(np.float32)
    if g.max() > 0:
        g = g / g.max()
    return g


def complex_rrmse(pred_complex, target_complex, eps=1e-12):
    num = np.sqrt(np.mean(np.abs(pred_complex - target_complex) ** 2))
    den = np.sqrt(np.mean(np.abs(target_complex) ** 2)) + eps
    return float(num / den)


def complex_mae(pred_complex, target_complex):
    return float(np.mean(np.abs(pred_complex - target_complex)))


def save_image(img, path, title=None, cmap="viridis", vmin=None, vmax=None):
    plt.figure(figsize=(5, 4))
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_vis(speed, source_map, pred_complex, target_complex, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)

    save_image(speed, os.path.join(out_dir, f"{prefix}_speed.png"), "speed", "inferno")
    save_image(source_map, os.path.join(out_dir, f"{prefix}_source.png"), "source map", "gray")

    save_image(target_complex.real, os.path.join(out_dir, f"{prefix}_target_real.png"), "target real", "seismic")
    save_image(target_complex.imag, os.path.join(out_dir, f"{prefix}_target_imag.png"), "target imag", "seismic")
    save_image(np.abs(target_complex), os.path.join(out_dir, f"{prefix}_target_abs.png"), "target abs", "viridis")

    save_image(pred_complex.real, os.path.join(out_dir, f"{prefix}_pred_real.png"), "pred real", "seismic")
    save_image(pred_complex.imag, os.path.join(out_dir, f"{prefix}_pred_imag.png"), "pred imag", "seismic")
    save_image(np.abs(pred_complex), os.path.join(out_dir, f"{prefix}_pred_abs.png"), "pred abs", "viridis")

    err = pred_complex - target_complex
    save_image(np.abs(err), os.path.join(out_dir, f"{prefix}_error_abs.png"), "abs error", "magma")


# ============================================================
# LRU cache for npz files
# ============================================================

class NPZCache:
    def __init__(self, max_size=2):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, path):
        path = str(path)

        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]

        d = np.load(path)
        item = {
            "target_480": d["target_480"].astype(np.float32),
            "wavefields": d["wavefields"].astype(np.complex64),
            "dobs_complex": d["dobs_complex"].astype(np.complex64),
            "src_indices": d["src_indices"].astype(np.int64),
            "rec_indices": d["rec_indices"].astype(np.int64),
        }

        self.cache[path] = item

        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

        return item


# ============================================================
# dataset
# ============================================================

class WavefieldForwardDataset(Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        image_size=480,
        speed_center=1500.0,
        speed_scale=100.0,
        wave_scale=0.1,
        source_sigma=2.0,
        max_files=-1,
        source_stride=1,
        cache_size=2,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.speed_center = speed_center
        self.speed_scale = speed_scale
        self.wave_scale = wave_scale
        self.source_sigma = source_sigma
        self.source_stride = source_stride

        self.files = sorted(
            list((self.data_root / split).glob(f"{split}_*.npz")),
            key=numeric_key,
        )

        if max_files > 0:
            self.files = self.files[:max_files]

        if len(self.files) == 0:
            raise RuntimeError(f"No files found in {self.data_root / split}")

        first = np.load(self.files[0])
        self.num_sources = first["wavefields"].shape[0]
        self.src_indices_first = first["src_indices"].astype(np.int64)
        self.rec_indices_first = first["rec_indices"].astype(np.int64)

        self.source_ids = list(range(0, self.num_sources, source_stride))

        # cache-friendly index: file 外层，source 内层；DataLoader 默认 shuffle=False
        self.index = []
        for fi in range(len(self.files)):
            for sid in self.source_ids:
                self.index.append((fi, sid))

        self.cache = NPZCache(max_size=cache_size)

        print(
            f"[Dataset] split={split}, files={len(self.files)}, "
            f"num_sources={self.num_sources}, used_sources={len(self.source_ids)}, "
            f"items={len(self.index)}, image_size={image_size}"
        )

    def __len__(self):
        return len(self.index)

    def _resize_tensor(self, x, mode="bilinear"):
        """
        x: torch [C,H,W]
        """
        if x.shape[-1] == self.image_size and x.shape[-2] == self.image_size:
            return x

        x4 = x.unsqueeze(0)
        x4 = F.interpolate(x4, size=(self.image_size, self.image_size), mode=mode, align_corners=False)
        return x4[0]

    def __getitem__(self, idx):
        file_idx, src_id = self.index[idx]
        path = self.files[file_idx]
        d = self.cache.get(path)

        speed = d["target_480"]
        wave = d["wavefields"][src_id]
        src_indices = d["src_indices"]
        rec_indices = d["rec_indices"]

        src_xy = src_indices[src_id]
        source_map = make_source_map_np(speed.shape, src_xy, sigma=self.source_sigma)

        speed_norm = normalize_speed_np(speed, self.speed_center, self.speed_scale)
        wave_real = wave.real.astype(np.float32) / self.wave_scale
        wave_imag = wave.imag.astype(np.float32) / self.wave_scale

        inp = np.stack([speed_norm, source_map], axis=0).astype(np.float32)
        out = np.stack([wave_real, wave_imag], axis=0).astype(np.float32)

        inp_t = torch.from_numpy(inp)
        out_t = torch.from_numpy(out)

        inp_t = self._resize_tensor(inp_t)
        out_t = self._resize_tensor(out_t)

        # receiver indices 如果 image_size != 480，需要缩放
        if self.image_size != 480:
            scale = self.image_size / 480.0
            rec_scaled = np.round(rec_indices.astype(np.float32) * scale).astype(np.int64)
            rec_scaled = np.clip(rec_scaled, 0, self.image_size - 1)
        else:
            rec_scaled = rec_indices

        return {
            "input": inp_t,
            "target": out_t,
            "source_id": torch.tensor(src_id, dtype=torch.long),
            "file_idx": torch.tensor(file_idx, dtype=torch.long),
            "rec_indices": torch.from_numpy(rec_scaled.astype(np.int64)),
        }


# ============================================================
# model
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, mid_ch), mid_ch),
            nn.SiLU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # 防止尺寸边界问题
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LightweightForwardUNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, base_ch=16):
        super().__init__()

        b = base_ch

        self.inc = DoubleConv(in_ch, b)
        self.d1 = Down(b, 2 * b)
        self.d2 = Down(2 * b, 4 * b)
        self.d3 = Down(4 * b, 8 * b)
        self.d4 = Down(8 * b, 8 * b)

        self.mid = DoubleConv(8 * b, 8 * b)

        # ===================== 已按你的要求修改 =====================
        self.u4 = Up(8 * b, 8 * b, 8 * b)
        self.u3 = Up(8 * b, 4 * b, 4 * b)
        self.u2 = Up(4 * b, 2 * b, 2 * b)
        self.u1 = Up(2 * b, b, b)
        # ===========================================================

        self.out = nn.Conv2d(b, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        s1 = self.inc(x)
        s2 = self.d1(s1)
        s3 = self.d2(s2)
        s4 = self.d3(s3)
        x = self.d4(s4)

        x = self.mid(x)

        x = self.u4(x, s4)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)

        return self.out(x)


# ============================================================
# loss and metrics
# ============================================================

def receiver_loss(pred, target, rec_indices):
    """
    pred/target: [B,2,H,W]
    rec_indices: [B,R,2]
    """
    b = pred.shape[0]
    losses = []

    for i in range(b):
        rr = rec_indices[i, :, 0].long().to(pred.device)
        cc = rec_indices[i, :, 1].long().to(pred.device)

        pred_rec = pred[i, :, rr, cc]      # [2,R]
        target_rec = target[i, :, rr, cc]  # [2,R]

        losses.append(F.mse_loss(pred_rec, target_rec))

    return torch.stack(losses).mean()


def compute_loss(pred, target, rec_indices, lambda_l1=1.0, lambda_mse=1.0, lambda_rec=2.0):
    l1 = F.l1_loss(pred, target)
    mse = F.mse_loss(pred, target)
    rec = receiver_loss(pred, target, rec_indices)

    total = lambda_l1 * l1 + lambda_mse * mse + lambda_rec * rec

    return total, {
        "l1": float(l1.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "rec": float(rec.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


@torch.no_grad()
def evaluate(model, loader, args, device, max_batches=-1, save_visual=False, epoch=0):
    model.eval()

    losses = []
    wave_rrmse_list = []
    wave_mae_list = []
    dobs_rrmse_list = []
    dobs_mae_list = []

    vis_saved = False

    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break

        x = batch["input"].to(device)
        y = batch["target"].to(device)
        rec_indices = batch["rec_indices"].to(device)

        pred = model(x)

        loss, loss_items = compute_loss(
            pred,
            y,
            rec_indices,
            lambda_l1=args.lambda_l1,
            lambda_mse=args.lambda_mse,
            lambda_rec=args.lambda_rec,
        )
        losses.append(loss_items["total"])

        pred_np = pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        rec_np = batch["rec_indices"].numpy()
        x_np = x.detach().cpu().numpy()

        for i in range(pred_np.shape[0]):
            pred_c = (pred_np[i, 0] + 1j * pred_np[i, 1]) * args.wave_scale
            true_c = (y_np[i, 0] + 1j * y_np[i, 1]) * args.wave_scale

            wave_rrmse_list.append(complex_rrmse(pred_c, true_c))
            wave_mae_list.append(complex_mae(pred_c, true_c))

            rr = rec_np[i, :, 0]
            cc = rec_np[i, :, 1]

            pred_dobs = pred_c[rr, cc]
            true_dobs = true_c[rr, cc]

            dobs_rrmse_list.append(complex_rrmse(pred_dobs, true_dobs))
            dobs_mae_list.append(complex_mae(pred_dobs, true_dobs))

            if save_visual and not vis_saved:
                vis_dir = os.path.join(args.output_dir, "visuals")
                os.makedirs(vis_dir, exist_ok=True)

                speed_norm = x_np[i, 0]
                speed = speed_norm * args.speed_scale + args.speed_center
                source_map = x_np[i, 1]

                save_vis(
                    speed,
                    source_map,
                    pred_c,
                    true_c,
                    vis_dir,
                    prefix=f"epoch_{epoch:03d}",
                )
                vis_saved = True

    model.train()

    return {
        "loss": float(np.mean(losses)),
        "wave_rrmse": float(np.mean(wave_rrmse_list)),
        "wave_rrmse_std": float(np.std(wave_rrmse_list)),
        "wave_mae": float(np.mean(wave_mae_list)),
        "dobs_rrmse": float(np.mean(dobs_rrmse_list)),
        "dobs_rrmse_std": float(np.std(dobs_rrmse_list)),
        "dobs_mae": float(np.mean(dobs_mae_list)),
    }


def save_curves(history, outdir):
    os.makedirs(outdir, exist_ok=True)

    def plot(keys, filename, ylabel):
        plt.figure(figsize=(7, 4))
        for k in keys:
            plt.plot(history[k], label=k)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, filename), dpi=150)
        plt.close()

    plot(["train_loss", "val_loss"], "curve_loss.png", "loss")
    plot(["val_wave_rrmse"], "curve_wave_rrmse.png", "wave RRMSE")
    plot(["val_dobs_rrmse"], "curve_dobs_rrmse.png", "dobs RRMSE")


# ============================================================
# train / eval
# ============================================================

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visuals"), exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    train_set = WavefieldForwardDataset(
        data_root=args.data_root,
        split="train",
        image_size=args.image_size,
        speed_center=args.speed_center,
        speed_scale=args.speed_scale,
        wave_scale=args.wave_scale,
        source_sigma=args.source_sigma,
        max_files=args.max_train_files,
        source_stride=args.source_stride,
        cache_size=args.cache_size,
    )
    val_set = WavefieldForwardDataset(
        data_root=args.data_root,
        split="test",
        image_size=args.image_size,
        speed_center=args.speed_center,
        speed_scale=args.speed_scale,
        wave_scale=args.wave_scale,
        source_sigma=args.source_sigma,
        max_files=args.max_test_files,
        source_stride=args.source_stride,
        cache_size=args.cache_size,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,   # cache-friendly
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = LightweightForwardUNet(
        in_ch=2,
        out_ch=2,
        base_ch=args.base_ch,
    ).to(device)

    print(f"model parameters = {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    best_metric = float("inf")
    best_epoch = -1

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_wave_rrmse": [],
        "val_dobs_rrmse": [],
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for step, batch in enumerate(train_loader, start=1):
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            rec_indices = batch["rec_indices"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                pred = model(x)
                loss, loss_items = compute_loss(
                    pred,
                    y,
                    rec_indices,
                    lambda_l1=args.lambda_l1,
                    lambda_mse=args.lambda_mse,
                    lambda_rec=args.lambda_rec,
                )

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

        scheduler.step()

        train_loss = float(np.mean(train_losses))

        val = evaluate(
            model,
            val_loader,
            args,
            device,
            max_batches=args.val_batches,
            save_visual=(epoch == 1 or epoch % args.save_vis_every == 0),
            epoch=epoch,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_wave_rrmse"].append(val["wave_rrmse"])
        history["val_dobs_rrmse"].append(val["dobs_rrmse"])

        print("=" * 80)
        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val['loss']:.6f} "
            f"wave_rrmse={val['wave_rrmse']:.6f} "
            f"dobs_rrmse={val['dobs_rrmse']:.6f}"
        )
        print("=" * 80)

        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        save_curves(history, args.output_dir)

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "val": val,
            "best_metric": best_metric,
        }

        torch.save(ckpt, os.path.join(args.output_dir, "checkpoints", "latest.pth"))

        metric = val["dobs_rrmse"]
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save(ckpt, os.path.join(args.output_dir, "checkpoints", "best.pth"))

            with open(os.path.join(args.output_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "best_metric_dobs_rrmse": best_metric,
                        "val": val,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(f"Saved best checkpoint at epoch {epoch}, dobs_rrmse={best_metric:.6f}")

    print("[Done]")
    print("best_epoch =", best_epoch)
    print("best_dobs_rrmse =", best_metric)


@torch.no_grad()
def eval_only(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", vars(args))

    model = LightweightForwardUNet(
        in_ch=2,
        out_ch=2,
        base_ch=ckpt_args.get("base_ch", args.base_ch),
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    test_set = WavefieldForwardDataset(
        data_root=args.data_root,
        split=args.eval_split,
        image_size=ckpt_args.get("image_size", args.image_size),
        speed_center=ckpt_args.get("speed_center", args.speed_center),
        speed_scale=ckpt_args.get("speed_scale", args.speed_scale),
        wave_scale=ckpt_args.get("wave_scale", args.wave_scale),
        source_sigma=ckpt_args.get("source_sigma", args.source_sigma),
        max_files=args.max_test_files,
        source_stride=args.source_stride,
        cache_size=args.cache_size,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    val = evaluate(
        model,
        test_loader,
        args,
        device,
        max_batches=-1,
        save_visual=True,
        epoch=0,
    )

    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)

    print(json.dumps(val, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--eval_split", type=str, default="test")

    parser.add_argument("--image_size", type=int, default=480)
    parser.add_argument("--speed_center", type=float, default=1500.0)
    parser.add_argument("--speed_scale", type=float, default=100.0)
    parser.add_argument("--wave_scale", type=float, default=0.1)
    parser.add_argument("--source_sigma", type=float, default=2.0)

    parser.add_argument("--base_ch", type=int, default=16)

    parser.add_argument("--max_train_files", type=int, default=-1)
    parser.add_argument("--max_test_files", type=int, default=-1)
    parser.add_argument("--source_stride", type=int, default=1)
    parser.add_argument("--cache_size", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    parser.add_argument("--lambda_rec", type=float, default=2.0)

    parser.add_argument("--val_batches", type=int, default=20)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_vis_every", type=int, default=5)

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        if args.ckpt_path == "":
            raise ValueError("--mode eval 需要指定 --ckpt_path")
        eval_only(args)


if __name__ == "__main__":
    main()