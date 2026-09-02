#!/usr/bin/env python3
"""Leakage-controlled IC-RFM v2 for OpenBreastUS v7.3.

The model learns a continuous clean-space flow from an InversionNet condition
to the ground-truth sound-speed map.  It intentionally uses only image-domain
condition caches produced by ``precompute_inversionnet_conditions_v73.py``;
no legacy measured data are mixed with the self-consistent CBS data.

Primary protocol used by this project:
  * train: 5-fold out-of-fold InversionNet conditions
  * validation / checkpoint selection: test_1 ... test_20
  * frozen DEV30 report: test_21 ... test_50
  * primary sampler: deterministic 4-step Heun
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------------
# Reproducibility and I/O
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def numeric_id(path: Path) -> int:
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else -1


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def file_manifest(files: Sequence[Path]) -> List[Dict]:
    return [
        {
            "sample_id": numeric_id(p),
            "name": p.name,
            "path": str(p.resolve()),
            "size": int(p.stat().st_size),
        }
        for p in files
    ]


def manifest_digest(items: Sequence[Dict]) -> str:
    payload = json.dumps(items, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_speed(x: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    return (x - 0.5 * (vmin + vmax)) / (0.5 * (vmax - vmin))


def denormalize_speed(x: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    return x * (0.5 * (vmax - vmin)) + 0.5 * (vmin + vmax)


class ConditionCacheDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        id_start: int = -1,
        id_end: int = -1,
        max_samples: int = -1,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.augment = augment
        split_dir = self.root / split
        files = sorted(split_dir.glob(f"{split}_*.npz"), key=numeric_id)
        if id_start >= 0:
            files = [p for p in files if numeric_id(p) >= id_start]
        if id_end >= 0:
            files = [p for p in files if numeric_id(p) <= id_end]
        if max_samples > 0:
            files = files[:max_samples]
        if not files:
            raise RuntimeError(
                f"No condition-cache files in {split_dir} for IDs "
                f"[{id_start}, {id_end}]"
            )
        ids = [numeric_id(p) for p in files]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Duplicate sample IDs in {split_dir}")
        self.files = files
        print(
            f"[Dataset] split={split} n={len(files)} "
            f"IDs={ids[0]}..{ids[-1]} augment={augment}"
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.files[index]
        with np.load(path) as d:
            cond = d["condition_norm"].astype(np.float32)
            target = d["target_norm"].astype(np.float32)
            cond_speed = d["condition_speed"].astype(np.float32)
            target_speed = d["target_speed"].astype(np.float32)

        arrays = [cond, target, cond_speed, target_speed]
        arrays = [a[None] if a.ndim == 2 else a for a in arrays]
        if any(a.shape != (1, 256, 256) for a in arrays):
            raise ValueError(f"Unexpected cache shape in {path}: {[a.shape for a in arrays]}")

        cond_t, target_t, cond_speed_t, target_speed_t = [
            torch.from_numpy(a.copy()) for a in arrays
        ]
        if self.augment:
            if torch.rand(()) < 0.5:
                cond_t = torch.flip(cond_t, (1,))
                target_t = torch.flip(target_t, (1,))
                cond_speed_t = torch.flip(cond_speed_t, (1,))
                target_speed_t = torch.flip(target_speed_t, (1,))
            if torch.rand(()) < 0.5:
                cond_t = torch.flip(cond_t, (2,))
                target_t = torch.flip(target_t, (2,))
                cond_speed_t = torch.flip(cond_speed_t, (2,))
                target_speed_t = torch.flip(target_speed_t, (2,))
        return {
            "condition": cond_t,
            "target": target_t,
            "condition_speed": cond_speed_t,
            "target_speed": target_speed_t,
            "sample_id": torch.tensor(numeric_id(path), dtype=torch.long),
        }


# -----------------------------------------------------------------------------
# Conditional velocity network
# -----------------------------------------------------------------------------


def make_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class FourierTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        half = dim // 2
        freq = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), max(half, 1))
        )
        self.register_buffer("freq", freq, persistent=False)
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * math.pi * t[:, None] * self.freq[None]
        emb = torch.cat((phase.sin(), phase.cos()), dim=1)
        return F.pad(emb, (0, self.dim - emb.shape[1]))


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = make_group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = make_group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(temb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class ConditionalVelocityUNet(nn.Module):
    """U-Net velocity field v_theta(x_t, condition, t).

    The third input channel is x_t-condition.  A zero-initialized output layer
    makes the initial ODE an identity map, which is a useful safety property for
    refinement from an already strong InversionNet reconstruction.
    """

    def __init__(self, base_ch: int = 32, time_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        b = base_ch
        self.time_mlp = nn.Sequential(
            FourierTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv2d(3, b, 3, padding=1)

        self.d1a, self.d1b = ResBlock(b, b, time_dim, dropout), ResBlock(b, b, time_dim, dropout)
        self.ds1 = nn.Conv2d(b, b, 4, 2, 1)
        self.d2a, self.d2b = ResBlock(b, 2*b, time_dim, dropout), ResBlock(2*b, 2*b, time_dim, dropout)
        self.ds2 = nn.Conv2d(2*b, 2*b, 4, 2, 1)
        self.d3a, self.d3b = ResBlock(2*b, 4*b, time_dim, dropout), ResBlock(4*b, 4*b, time_dim, dropout)
        self.ds3 = nn.Conv2d(4*b, 4*b, 4, 2, 1)
        self.d4a, self.d4b = ResBlock(4*b, 4*b, time_dim, dropout), ResBlock(4*b, 4*b, time_dim, dropout)
        self.ds4 = nn.Conv2d(4*b, 4*b, 4, 2, 1)

        self.mid1 = ResBlock(4*b, 4*b, time_dim, dropout)
        self.mid2 = ResBlock(4*b, 4*b, time_dim, dropout)

        self.us4 = nn.ConvTranspose2d(4*b, 4*b, 4, 2, 1)
        self.u4a, self.u4b = ResBlock(8*b, 4*b, time_dim, dropout), ResBlock(4*b, 4*b, time_dim, dropout)
        self.us3 = nn.ConvTranspose2d(4*b, 4*b, 4, 2, 1)
        self.u3a, self.u3b = ResBlock(8*b, 4*b, time_dim, dropout), ResBlock(4*b, 4*b, time_dim, dropout)
        self.us2 = nn.ConvTranspose2d(4*b, 4*b, 4, 2, 1)
        self.u2a, self.u2b = ResBlock(6*b, 2*b, time_dim, dropout), ResBlock(2*b, 2*b, time_dim, dropout)
        self.us1 = nn.ConvTranspose2d(2*b, 2*b, 4, 2, 1)
        self.u1a, self.u1b = ResBlock(3*b, b, time_dim, dropout), ResBlock(b, b, time_dim, dropout)

        self.out_norm = make_group_norm(b)
        self.out_conv = nn.Conv2d(b, 1, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(t)
        h = self.in_conv(torch.cat((x, condition, x - condition), dim=1))
        h = self.d1b(self.d1a(h, temb), temb); s1 = h; h = self.ds1(h)
        h = self.d2b(self.d2a(h, temb), temb); s2 = h; h = self.ds2(h)
        h = self.d3b(self.d3a(h, temb), temb); s3 = h; h = self.ds3(h)
        h = self.d4b(self.d4a(h, temb), temb); s4 = h; h = self.ds4(h)
        h = self.mid2(self.mid1(h, temb), temb)
        h = self.us4(h); h = self.u4b(self.u4a(torch.cat((h, s4), 1), temb), temb)
        h = self.us3(h); h = self.u3b(self.u3a(torch.cat((h, s3), 1), temb), temb)
        h = self.us2(h); h = self.u2b(self.u2a(torch.cat((h, s2), 1), temb), temb)
        h = self.us1(h); h = self.u1b(self.u1a(torch.cat((h, s1), 1), temb), temb)
        return self.out_conv(F.silu(self.out_norm(h)))


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        for key, value in self.model.state_dict().items():
            if value.is_floating_point():
                value.mul_(self.decay).add_(source[key].detach(), alpha=1.0-self.decay)
            else:
                value.copy_(source[key])


# -----------------------------------------------------------------------------
# Flow objective and solver
# -----------------------------------------------------------------------------


def image_gradients(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return x[:, :, :, 1:] - x[:, :, :, :-1], x[:, :, 1:, :] - x[:, :, :-1, :]


def flow_loss(
    model: nn.Module,
    condition: torch.Tensor,
    target: torch.Tensor,
    sigma_max: float,
    zero_start_prob: float,
    lambda_endpoint_l1: float,
    lambda_endpoint_grad: float,
) -> Dict[str, torch.Tensor]:
    b = condition.shape[0]
    sigma = torch.rand((b, 1, 1, 1), device=condition.device) * sigma_max
    zero_mask = torch.rand((b, 1, 1, 1), device=condition.device) < zero_start_prob
    sigma = torch.where(zero_mask, torch.zeros_like(sigma), sigma)
    start = torch.clamp(condition + sigma * torch.randn_like(condition), -1.0, 1.0)
    t = torch.rand((b,), device=condition.device)
    tb = t[:, None, None, None]
    xt = (1.0 - tb) * start + tb * target
    target_v = target - start
    pred_v = model(xt, condition, t)
    velocity_mse = F.mse_loss(pred_v, target_v)

    endpoint = xt + (1.0 - tb) * pred_v
    endpoint_l1 = F.l1_loss(endpoint, target)
    ex, ey = image_gradients(endpoint)
    tx, ty = image_gradients(target)
    endpoint_grad = F.l1_loss(ex, tx) + F.l1_loss(ey, ty)
    total = (
        velocity_mse
        + lambda_endpoint_l1 * endpoint_l1
        + lambda_endpoint_grad * endpoint_grad
    )
    return {
        "total": total,
        "velocity_mse": velocity_mse.detach(),
        "endpoint_l1": endpoint_l1.detach(),
        "endpoint_grad": endpoint_grad.detach(),
    }


@torch.no_grad()
def integrate_flow(
    model: nn.Module,
    condition: torch.Tensor,
    steps: int,
    solver: str = "heun",
    noise_scale: float = 0.0,
    generator: Optional[torch.Generator] = None,
    return_states: bool = False,
):
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if noise_scale > 0:
        noise = torch.randn(
            condition.shape, device=condition.device, dtype=condition.dtype,
            generator=generator,
        )
        x = torch.clamp(condition + noise_scale * noise, -1.0, 1.0)
    else:
        x = condition.clone()
    states = [x.clone()]
    dt = 1.0 / steps
    for k in range(steps):
        t0 = torch.full((x.shape[0],), k / steps, device=x.device, dtype=x.dtype)
        v0 = model(x, condition, t0)
        if solver == "euler":
            x = x + dt * v0
        elif solver == "heun":
            proposal = x + dt * v0
            t1 = torch.full((x.shape[0],), (k+1) / steps, device=x.device, dtype=x.dtype)
            v1 = model(proposal, condition, t1)
            x = x + 0.5 * dt * (v0 + v1)
        else:
            raise ValueError(f"Unknown solver: {solver}")
        x = torch.clamp(x, -1.05, 1.05)
        states.append(x.clone())
    x = torch.clamp(x, -1.0, 1.0)
    states[-1] = x
    return (x, states) if return_states else x


# -----------------------------------------------------------------------------
# Metrics and statistics
# -----------------------------------------------------------------------------


def metric_row(pred: np.ndarray, target: np.ndarray, data_range: float) -> Dict[str, float]:
    diff = pred - target
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    psnr = 99.0 if mse <= 1e-16 else float(20.0 * np.log10(data_range / rmse))
    ssim = float(structural_similarity(target, pred, data_range=data_range))
    return {"mse": mse, "mae": mae, "rmse": rmse, "psnr": psnr, "ssim": ssim}


def aggregate_rows(rows: Sequence[Dict]) -> Dict:
    metrics = ["mse", "mae", "rmse", "psnr", "ssim"]
    summary: Dict[str, object] = {"num_samples": len(rows)}
    for prefix in ("condition", "flow"):
        for metric in metrics:
            values = np.asarray([r[f"{prefix}_{metric}"] for r in rows], np.float64)
            summary[f"{prefix}_{metric}"] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    for metric in metrics:
        # Positive means flow is better for every metric.
        if metric in ("mse", "mae", "rmse"):
            delta = np.asarray(
                [r[f"condition_{metric}"] - r[f"flow_{metric}"] for r in rows]
            )
        else:
            delta = np.asarray(
                [r[f"flow_{metric}"] - r[f"condition_{metric}"] for r in rows]
            )
        summary[f"improvement_{metric}"] = {
            "mean": float(delta.mean()),
            "std": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
            "wins": int(np.sum(delta > 0)),
            "ties": int(np.sum(delta == 0)),
            "losses": int(np.sum(delta < 0)),
        }
    return summary


def bootstrap_paired(delta: np.ndarray, seed: int, n_boot: int = 10000) -> Dict:
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = len(delta)
    for i in range(n_boot):
        means[i] = rng.choice(delta, size=n, replace=True).mean()
    try:
        w_two = float(wilcoxon(delta, alternative="two-sided").pvalue)
        w_greater = float(wilcoxon(delta, alternative="greater").pvalue)
    except ValueError:
        w_two, w_greater = 1.0, 1.0
    return {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "bootstrap_mean_95ci": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "wilcoxon_two_sided_p": w_two,
        "wilcoxon_improvement_p": w_greater,
        "wins": int(np.sum(delta > 0)),
        "losses": int(np.sum(delta < 0)),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    speed_min: float,
    speed_max: float,
    steps: int,
    solver: str,
    noise_scale: float = 0.0,
) -> Tuple[List[Dict], Dict]:
    model.eval()
    rows: List[Dict] = []
    for batch in loader:
        condition = batch["condition"].to(device, non_blocking=True)
        target_speed = batch["target_speed"].numpy()
        condition_speed = batch["condition_speed"].numpy()
        pred = integrate_flow(model, condition, steps, solver, noise_scale)
        pred_speed = denormalize_speed(pred, speed_min, speed_max).cpu().numpy()
        ids = batch["sample_id"].numpy()
        for i, sid in enumerate(ids):
            cm = metric_row(condition_speed[i, 0], target_speed[i, 0], speed_max-speed_min)
            fm = metric_row(pred_speed[i, 0], target_speed[i, 0], speed_max-speed_min)
            row: Dict[str, object] = {"sample_id": int(sid)}
            row.update({f"condition_{k}": v for k, v in cm.items()})
            row.update({f"flow_{k}": v for k, v in fm.items()})
            row["mse_improvement"] = cm["mse"] - fm["mse"]
            row["mae_improvement"] = cm["mae"] - fm["mae"]
            row["psnr_improvement"] = fm["psnr"] - cm["psnr"]
            row["ssim_improvement"] = fm["ssim"] - cm["ssim"]
            rows.append(row)
    return rows, aggregate_rows(rows)


def save_visuals(
    model: nn.Module,
    dataset: Dataset,
    path: Path,
    device: torch.device,
    args,
    max_samples: int = 4,
) -> None:
    n = min(max_samples, len(dataset))
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.1*n), squeeze=False)
    model.eval()
    for i in range(n):
        item = dataset[i]
        condition = item["condition"][None].to(device)
        pred = integrate_flow(model, condition, args.steps, args.solver)
        pred = denormalize_speed(pred, args.speed_min, args.speed_max)[0, 0].cpu().numpy()
        cond = item["condition_speed"][0].numpy()
        target = item["target_speed"][0].numpy()
        images = [cond, target, pred, pred-target]
        titles = ["InversionNet", "Target", f"IC-RFM ({args.steps} steps)", "Error"]
        for j, (image, title) in enumerate(zip(images, titles)):
            if j < 3:
                h = axes[i, j].imshow(image, cmap="inferno", vmin=args.speed_min, vmax=args.speed_max)
            else:
                lim = max(float(np.abs(image).max()), 1.0)
                h = axes[i, j].imshow(image, cmap="bwr", vmin=-lim, vmax=lim)
            axes[i, j].set_title(title)
            axes[i, j].axis("off")
            fig.colorbar(h, ax=axes[i, j], fraction=0.046, pad=0.03)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Train / eval entry points
# -----------------------------------------------------------------------------


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, args) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) >= batch_size,
        persistent_workers=args.num_workers > 0,
    )


def save_checkpoint(path: Path, model, ema, optimizer, scheduler, epoch, args, metrics) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "val_metrics": metrics,
        },
        path,
    )


def load_model(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, Dict, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("args", {})
    model = ConditionalVelocityUNet(
        base_ch=int(cfg.get("base_ch", 32)),
        time_dim=int(cfg.get("time_dim", 128)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    state = ckpt.get("ema_model", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg, ckpt


def train(args) -> None:
    out = Path(args.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "visuals").mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    config_path = out / "config.json"
    latest = out / "checkpoints" / "latest.pth"
    current_config = vars(args).copy()
    if config_path.exists():
        old_config = json.loads(config_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise RuntimeError(
                f"Output directory already has a config: {out}. "
                "Use a new RUN_ROOT, or use --resume with identical protected settings."
            )
        mutable = {
            "epochs", "device", "num_workers", "log_every", "visual_every",
            "early_stop_patience", "resume",
        }
        changed = sorted(
            key for key, old_value in old_config.items()
            if key not in mutable and key in current_config
            and current_config[key] != old_value
        )
        if changed:
            raise RuntimeError(
                f"Resume guard failed for {out}: protected settings changed: {changed}. "
                "Use a new RUN_ROOT for a new experiment."
            )
        if not latest.exists():
            raise RuntimeError(f"--resume requested but checkpoint is missing: {latest}")
    elif args.resume:
        raise RuntimeError(f"--resume requested but config is missing: {config_path}")

    train_set = ConditionCacheDataset(
        args.condition_root, args.train_split, args.train_start, args.train_end,
        args.max_train, args.augment,
    )
    val_set = ConditionCacheDataset(
        args.condition_root, args.val_split, args.val_start, args.val_end,
        args.max_val, False,
    )
    overlap = set(map(numeric_id, train_set.files)) & set(map(numeric_id, val_set.files))
    if args.train_split == args.val_split and overlap:
        raise RuntimeError(f"Training/validation overlap: {sorted(overlap)[:10]}")
    manifests = {
        "train": file_manifest(train_set.files),
        "validation": file_manifest(val_set.files),
    }
    manifests["train_sha256"] = manifest_digest(manifests["train"])
    manifests["validation_sha256"] = manifest_digest(manifests["validation"])
    json_dump(out / "split_manifest.json", manifests)
    json_dump(config_path, current_config)

    train_loader = make_loader(train_set, args.batch_size, True, args)
    val_loader = make_loader(val_set, args.eval_batch_size, False, args)
    model = ConditionalVelocityUNet(args.base_ch, args.time_dim, args.dropout).to(device)
    ema = EMA(model, args.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = args.use_amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    print("="*100)
    print("IC-RFM training")
    print("device             =", device)
    print("parameters (M)     =", sum(p.numel() for p in model.parameters())/1e6)
    print("train / validation=", len(train_set), len(val_set))
    print("primary sampler    =", args.steps, args.solver)
    print("start sigma max    =", args.start_sigma_max)
    print("="*100)

    history: List[Dict] = []
    best_mse = float("inf")
    best_guard_mse = float("inf")
    best_guard_epoch = -1
    wait = 0
    start_epoch = 1
    if args.resume and latest.exists():
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        if (out / "history.json").exists():
            history = json.loads((out / "history.json").read_text(encoding="utf-8"))
        if history:
            best_mse = min(h["flow_mse"] for h in history)
            guarded = [h for h in history if h["structure_guard_pass"]]
            if guarded:
                best_guard_mse = min(h["flow_mse"] for h in guarded)
                best_guard_epoch = min(guarded, key=lambda h: h["flow_mse"])["epoch"]
        print(f"[resume] epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs+1):
        model.train()
        running: List[float] = []
        t0 = time.time()
        for step, batch in enumerate(train_loader, 1):
            condition = batch["condition"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            try:
                amp_context = torch.amp.autocast(device_type="cuda", enabled=amp_enabled)
            except AttributeError:
                amp_context = torch.cuda.amp.autocast(enabled=amp_enabled)
            with amp_context:
                losses = flow_loss(
                    model, condition, target, args.start_sigma_max,
                    args.zero_start_prob, args.lambda_endpoint_l1,
                    args.lambda_endpoint_grad,
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            running.append(float(losses["total"].detach().cpu()))
            if step % args.log_every == 0:
                print(
                    f"epoch {epoch:03d}/{args.epochs:03d} "
                    f"step {step:04d}/{len(train_loader):04d} "
                    f"loss={np.mean(running[-args.log_every:]):.6f}"
                )
        scheduler.step()
        rows, summary = evaluate(
            ema.model, val_loader, device, args.speed_min, args.speed_max,
            args.steps, args.solver,
        )
        cond_mse = summary["condition_mse"]["mean"]
        flow_mse = summary["flow_mse"]["mean"]
        cond_ssim = summary["condition_ssim"]["mean"]
        flow_ssim = summary["flow_ssim"]["mean"]
        guard_pass = flow_ssim >= cond_ssim - args.ssim_guard
        item = {
            "epoch": epoch,
            "train_loss": float(np.mean(running)),
            "condition_mse": cond_mse,
            "flow_mse": flow_mse,
            "condition_mae": summary["condition_mae"]["mean"],
            "flow_mae": summary["flow_mae"]["mean"],
            "condition_psnr": summary["condition_psnr"]["mean"],
            "flow_psnr": summary["flow_psnr"]["mean"],
            "condition_ssim": cond_ssim,
            "flow_ssim": flow_ssim,
            "structure_guard_pass": bool(guard_pass),
            "seconds": float(time.time()-t0),
        }
        history.append(item)
        json_dump(out / "history.json", history)
        save_checkpoint(latest, model, ema, optimizer, scheduler, epoch, args, summary)
        write_csv(out / "validation_latest.csv", rows)
        json_dump(out / "validation_latest.json", summary)

        if flow_mse < best_mse:
            best_mse = flow_mse
            save_checkpoint(out/"checkpoints"/"best_mse.pth", model, ema, optimizer, scheduler, epoch, args, summary)
        improved_guard = guard_pass and flow_mse < best_guard_mse - args.min_delta
        if improved_guard:
            best_guard_mse = flow_mse
            best_guard_epoch = epoch
            wait = 0
            save_checkpoint(out/"checkpoints"/"best_guard.pth", model, ema, optimizer, scheduler, epoch, args, summary)
            shutil.copy2(out/"checkpoints"/"best_guard.pth", out/"checkpoints"/"best.pth")
        else:
            wait += 1

        print(
            f"[epoch {epoch:03d}] loss={item['train_loss']:.6f} | "
            f"MSE {cond_mse:.3f}->{flow_mse:.3f} | "
            f"MAE {item['condition_mae']:.4f}->{item['flow_mae']:.4f} | "
            f"SSIM {cond_ssim:.6f}->{flow_ssim:.6f} | "
            f"guard={guard_pass} | {item['seconds']:.1f}s"
        )
        if epoch == 1 or epoch % args.visual_every == 0 or improved_guard:
            save_visuals(
                ema.model, val_set, out/"visuals"/f"epoch_{epoch:03d}.png",
                device, args,
            )
        if args.early_stop_patience > 0 and wait >= args.early_stop_patience:
            if best_guard_epoch > 0:
                print(f"[early stop] best guarded epoch={best_guard_epoch}")
            else:
                print("[early stop] no checkpoint passed the SSIM guard")
            break

    if not (out/"checkpoints"/"best.pth").exists():
        shutil.copy2(out/"checkpoints"/"best_mse.pth", out/"checkpoints"/"best.pth")
        print("[warning] No checkpoint passed the SSIM guard; best.pth uses best MSE.")
    print("[DONE]", out.resolve())


def eval_mode(args) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg, ckpt = load_model(args.ckpt_path, device)
    # Normalization and architecture are properties of the checkpoint.
    speed_min = float(cfg.get("speed_min", args.speed_min))
    speed_max = float(cfg.get("speed_max", args.speed_max))
    dataset = ConditionCacheDataset(
        args.condition_root, args.eval_split, args.eval_start, args.eval_end,
        args.max_eval, False,
    )
    loader = make_loader(dataset, args.eval_batch_size, False, args)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, summary = evaluate(
        model, loader, device, speed_min, speed_max,
        args.steps, args.solver, args.inference_noise_scale,
    )
    write_csv(out/"sample_results.csv", rows)
    json_dump(out/"summary.json", summary)
    paired = {}
    for metric in ("mse", "mae", "psnr", "ssim"):
        if metric in ("mse", "mae"):
            delta = np.asarray([r[f"condition_{metric}"]-r[f"flow_{metric}"] for r in rows])
        else:
            delta = np.asarray([r[f"flow_{metric}"]-r[f"condition_{metric}"] for r in rows])
        paired[metric] = bootstrap_paired(delta, args.seed+numeric_id(Path(args.ckpt_path)), args.bootstrap_samples)
    json_dump(out/"paired_statistics.json", paired)
    json_dump(out/"evaluation_config.json", {
        **vars(args),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_training_args": cfg,
        "evaluated_files": file_manifest(dataset.files),
    })
    save_visuals(model, dataset, out/"examples.png", device, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[DONE]", out.resolve())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("train", "eval"), default="train")
    p.add_argument("--condition_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ckpt_path", default="")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--speed_min", type=float, default=1400.0)
    p.add_argument("--speed_max", type=float, default=1605.0)

    p.add_argument("--train_split", default="train")
    p.add_argument("--train_start", type=int, default=-1)
    p.add_argument("--train_end", type=int, default=-1)
    p.add_argument("--max_train", type=int, default=-1)
    p.add_argument("--val_split", default="test")
    p.add_argument("--val_start", type=int, default=1)
    p.add_argument("--val_end", type=int, default=20)
    p.add_argument("--max_val", type=int, default=-1)
    p.add_argument("--eval_split", default="test")
    p.add_argument("--eval_start", type=int, default=21)
    p.add_argument("--eval_end", type=int, default=50)
    p.add_argument("--max_eval", type=int, default=-1)

    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--time_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--start_sigma_max", type=float, default=0.0)
    p.add_argument("--zero_start_prob", type=float, default=1.0)
    p.add_argument("--lambda_endpoint_l1", type=float, default=1.0)
    p.add_argument("--lambda_endpoint_grad", type=float, default=0.20)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--solver", choices=("euler", "heun"), default="heun")
    p.add_argument("--inference_noise_scale", type=float, default=0.0)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ssim_guard", type=float, default=0.001)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--visual_every", type=int, default=5)
    p.add_argument("--bootstrap_samples", type=int, default=10000)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "train":
        train(args)
    else:
        if not args.ckpt_path:
            raise ValueError("--mode eval requires --ckpt_path")
        eval_mode(args)


if __name__ == "__main__":
    main()
