import argparse
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Utilities
# ============================================================

def numeric_key(p):
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else -1


def estimate_wave_scale(files):
    """
    Global RMS scale of complex wavefield:
        sqrt(mean(|u|^2))
    """
    ss = 0.0
    n = 0

    for p in files:
        with np.load(p, allow_pickle=True) as z:
            w = z["wavefields"].astype(np.complex64)

        ss += float(
            np.sum(np.abs(w).astype(np.float64) ** 2)
        )
        n += int(w.size)

    return math.sqrt(ss / max(n, 1))


# ============================================================
# Dataset
# ============================================================

class WavefieldPairDataset(Dataset):

    def __init__(
        self,
        root,
        split,
        wave_scale,
        max_images=0,
        source_sigma=2.0,
    ):
        self.root = Path(root)
        self.split = split
        self.wave_scale = float(wave_scale)
        self.source_sigma = float(source_sigma)

        files = sorted(
            (self.root / split).glob("*.npz"),
            key=numeric_key,
        )

        if max_images > 0:
            files = files[:max_images]

        self.files = files

        self.items = []

        for p in self.files:
            with np.load(p, allow_pickle=True) as z:
                ns = int(z["wavefields"].shape[0])

            for s in range(ns):
                self.items.append((p, s))

        # normalized coordinate grids
        yy = np.linspace(
            -1.0, 1.0, 480,
            dtype=np.float32
        )

        xx = np.linspace(
            -1.0, 1.0, 480,
            dtype=np.float32
        )

        self.yy, self.xx = np.meshgrid(
            yy,
            xx,
            indexing="ij",
        )

        py = np.arange(
            480,
            dtype=np.float32
        )[:, None]

        px = np.arange(
            480,
            dtype=np.float32
        )[None, :]

        self.py = py
        self.px = px

        print(
            f"[Dataset] split={split} "
            f"images={len(self.files)} "
            f"pairs={len(self.items)} "
            f"wave_scale={self.wave_scale:.8e}"
        )

    def __len__(self):
        return len(self.items)

    @lru_cache(maxsize=8)
    def _load(self, path_str):
        p = Path(path_str)

        with np.load(p, allow_pickle=True) as z:
            data = {
                "target_480":
                    z["target_480"].astype(np.float32),

                "wavefields":
                    z["wavefields"].astype(np.complex64),

                "src_indices":
                    z["src_indices"].astype(np.int64),

                "rec_indices":
                    z["rec_indices"].astype(np.int64),

                "dobs_complex":
                    z["dobs_complex"].astype(np.complex64),
            }

        return data

    def __getitem__(self, idx):

        p, source_idx = self.items[idx]

        d = self._load(str(p))

        speed = d["target_480"]

        src = d["src_indices"][source_idx]

        wave = d["wavefields"][source_idx]

        dobs = d["dobs_complex"][source_idx]

        rec = d["rec_indices"]

        # ----------------------------------------------------
        # speed normalization consistent with current project
        # nominal speed interval [1400,1605]
        # ----------------------------------------------------

        speed_norm = (
            speed - 1502.5
        ) / 102.5

        sy = int(src[0])
        sx = int(src[1])

        sy_norm = (
            2.0 * sy / 479.0 - 1.0
        )

        sx_norm = (
            2.0 * sx / 479.0 - 1.0
        )

        # relative coordinates
        dy = (
            self.yy - sy_norm
        ) / 2.0

        dx = (
            self.xx - sx_norm
        ) / 2.0

        radius = np.sqrt(
            dx ** 2 + dy ** 2
        ) / math.sqrt(2.0)

        # localized source map
        dist2 = (
            (self.py - sy) ** 2
            +
            (self.px - sx) ** 2
        )

        source_map = np.exp(
            -dist2 /
            (
                2.0
                *
                self.source_sigma ** 2
            )
        ).astype(np.float32)

        inp = np.stack(
            [
                speed_norm,
                dx,
                dy,
                radius,
                source_map,
            ],
            axis=0,
        ).astype(np.float32)

        target = np.stack(
            [
                wave.real,
                wave.imag,
            ],
            axis=0,
        ).astype(np.float32)

        target /= self.wave_scale

        dobs_2ch = np.stack(
            [
                dobs.real,
                dobs.imag,
            ],
            axis=0,
        ).astype(np.float32)

        dobs_2ch /= self.wave_scale

        return {
            "input":
                torch.from_numpy(inp),

            "target":
                torch.from_numpy(target),

            "dobs":
                torch.from_numpy(dobs_2ch),

            "rec":
                torch.from_numpy(rec),

            "source_idx":
                torch.tensor(
                    source_idx,
                    dtype=torch.long,
                ),
        }


# ============================================================
# FNO
# ============================================================

class SpectralConv2d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / math.sqrt(
            in_channels * out_channels
        )

        shape = (
            in_channels,
            out_channels,
            modes,
            modes,
        )

        self.weight_pos = nn.Parameter(
            scale
            *
            torch.randn(
                *shape,
                dtype=torch.cfloat,
            )
        )

        self.weight_neg = nn.Parameter(
            scale
            *
            torch.randn(
                *shape,
                dtype=torch.cfloat,
            )
        )

    def compl_mul2d(
        self,
        x,
        weights,
    ):
        return torch.einsum(
            "bixy,ioxy->boxy",
            x,
            weights,
        )

    def forward(self, x):

        b, _, h, w = x.shape

        x_ft = torch.fft.rfft2(
            x,
            norm="ortho",
        )

        out_ft = torch.zeros(
            b,
            self.out_channels,
            h,
            w // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        mx = min(
            self.modes,
            h // 2,
        )

        my = min(
            self.modes,
            w // 2 + 1,
        )

        out_ft[
            :,
            :,
            :mx,
            :my
        ] = self.compl_mul2d(
            x_ft[
                :,
                :,
                :mx,
                :my
            ],
            self.weight_pos[
                :,
                :,
                :mx,
                :my
            ],
        )

        out_ft[
            :,
            :,
            -mx:,
            :my
        ] = self.compl_mul2d(
            x_ft[
                :,
                :,
                -mx:,
                :my
            ],
            self.weight_neg[
                :,
                :,
                :mx,
                :my
            ],
        )

        return torch.fft.irfft2(
            out_ft,
            s=(h, w),
            norm="ortho",
        )


class FNOBlock(nn.Module):

    def __init__(
        self,
        width,
        modes,
    ):
        super().__init__()

        self.spectral = SpectralConv2d(
            width,
            width,
            modes,
        )

        self.local = nn.Conv2d(
            width,
            width,
            kernel_size=1,
        )

        self.norm = nn.GroupNorm(
            num_groups=4,
            num_channels=width,
        )

    def forward(self, x):

        y = (
            self.spectral(x)
            +
            self.local(x)
        )

        y = self.norm(y)

        y = F.gelu(y)

        return x + y


class WavefieldFNO(nn.Module):

    def __init__(
        self,
        in_channels=5,
        width=16,
        modes=96,
        depth=4,
    ):
        super().__init__()

        self.lift = nn.Sequential(
            nn.Conv2d(
                in_channels,
                width,
                kernel_size=1,
            ),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList(
            [
                FNOBlock(
                    width,
                    modes,
                )
                for _ in range(depth)
            ]
        )

        self.proj = nn.Sequential(
            nn.Conv2d(
                width,
                width,
                kernel_size=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                width,
                2,
                kernel_size=1,
            ),
        )

    def forward(self, x):

        x = self.lift(x)

        for block in self.blocks:
            x = block(x)

        return self.proj(x)


# ============================================================
# Receiver extraction
# ============================================================

def sample_receivers(
    field,
    rec,
):
    """
    field [B,2,H,W]
    rec   [B,L,2]

    return [B,2,L]
    """

    b, c, h, w = field.shape

    flat = field.reshape(
        b,
        c,
        h * w,
    )

    idx = (
        rec[:, :, 0] * w
        +
        rec[:, :, 1]
    )

    idx = (
        idx
        .unsqueeze(1)
        .expand(-1, c, -1)
    )

    return torch.gather(
        flat,
        2,
        idx,
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    full_num = 0.0
    full_den = 0.0

    rec_num = 0.0
    rec_den = 0.0

    for batch in loader:

        x = batch["input"].to(
            device,
            non_blocking=True,
        )

        y = batch["target"].to(
            device,
            non_blocking=True,
        )

        dobs = batch["dobs"].to(
            device,
            non_blocking=True,
        )

        rec = batch["rec"].to(
            device,
            non_blocking=True,
        )

        pred = model(x)

        full_num += float(
            torch.sum(
                (pred - y) ** 2
            ).cpu()
        )

        full_den += float(
            torch.sum(
                y ** 2
            ).cpu()
        )

        pred_rec = sample_receivers(
            pred,
            rec,
        )

        rec_num += float(
            torch.sum(
                (pred_rec - dobs) ** 2
            ).cpu()
        )

        rec_den += float(
            torch.sum(
                dobs ** 2
            ).cpu()
        )

    full_rrmse = math.sqrt(
        full_num /
        max(full_den, 1e-12)
    )

    rec_rrmse = math.sqrt(
        rec_num /
        max(rec_den, 1e-12)
    )

    return {
        "full_rrmse":
            full_rrmse,

        "receiver_rrmse":
            rec_rrmse,
    }


# ============================================================
# Main
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data_root",
        required=True,
    )

    ap.add_argument(
        "--output_dir",
        required=True,
    )

    ap.add_argument(
        "--modes",
        type=int,
        default=96,
    )

    ap.add_argument(
        "--width",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--depth",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )

    ap.add_argument(
        "--weight_decay",
        type=float,
        default=1e-6,
    )

    ap.add_argument(
        "--receiver_weight",
        type=float,
        default=0.1,
    )

    ap.add_argument(
        "--max_train_images",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--max_val_images",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260904,
    )

    ap.add_argument(
        "--device",
        default="cuda:0",
    )

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    root = Path(
        args.data_root
    )

    out = Path(
        args.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_files = sorted(
        (root / "train").glob(
            "*.npz"
        ),
        key=numeric_key,
    )

    if args.max_train_images > 0:
        train_files = train_files[
            :args.max_train_images
        ]

    wave_scale = estimate_wave_scale(
        train_files
    )

    print("=" * 100)
    print(
        "C5.1c SOURCE-CONDITIONED "
        "HIGH-MODE FNO"
    )
    print("=" * 100)

    print(
        "wave_scale =",
        f"{wave_scale:.8e}",
    )

    print(
        "modes      =",
        args.modes,
    )

    print(
        "width      =",
        args.width,
    )

    print(
        "depth      =",
        args.depth,
    )

    train_set = WavefieldPairDataset(
        root,
        "train",
        wave_scale,
        max_images=
            args.max_train_images,
    )

    val_set = WavefieldPairDataset(
        root,
        "val",
        wave_scale,
        max_images=
            args.max_val_images,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    train_eval_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    model = WavefieldFNO(
        width=args.width,
        modes=args.modes,
        depth=args.depth,
    ).to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "parameters =",
        f"{n_params:,}",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=
            args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    )

    history = []

    best_train = float("inf")
    best_val = float("inf")

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        total_loss = 0.0

        for batch in train_loader:

            x = batch["input"].to(
                device,
                non_blocking=True,
            )

            y = batch["target"].to(
                device,
                non_blocking=True,
            )

            dobs = batch["dobs"].to(
                device,
                non_blocking=True,
            )

            rec = batch["rec"].to(
                device,
                non_blocking=True,
            )

            pred = model(x)

            full_loss = F.mse_loss(
                pred,
                y,
            )

            pred_rec = sample_receivers(
                pred,
                rec,
            )

            receiver_loss = (
                F.mse_loss(
                    pred_rec,
                    dobs,
                )
            )

            loss = (
                full_loss
                +
                args.receiver_weight
                *
                receiver_loss
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            total_loss += float(
                loss.detach().cpu()
            )

        scheduler.step()

        train_metrics = evaluate(
            model,
            train_eval_loader,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        row = {
            "epoch":
                epoch,

            "loss":
                total_loss /
                max(
                    len(train_loader),
                    1,
                ),

            "train_full_rrmse":
                train_metrics[
                    "full_rrmse"
                ],

            "train_receiver_rrmse":
                train_metrics[
                    "receiver_rrmse"
                ],

            "val_full_rrmse":
                val_metrics[
                    "full_rrmse"
                ],

            "val_receiver_rrmse":
                val_metrics[
                    "receiver_rrmse"
                ],

            "lr":
                scheduler.get_last_lr()[0],
        }

        history.append(row)

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"loss={row['loss']:.6e} | "
            f"train full="
            f"{row['train_full_rrmse']:.4f} "
            f"rec="
            f"{row['train_receiver_rrmse']:.4f} | "
            f"val full="
            f"{row['val_full_rrmse']:.4f} "
            f"rec="
            f"{row['val_receiver_rrmse']:.4f}"
        )

        ckpt = {
            "model_state":
                model.state_dict(),

            "args":
                vars(args),

            "wave_scale":
                wave_scale,

            "history":
                history,
        }

        torch.save(
            ckpt,
            out / "last.pth",
        )

        if (
            train_metrics[
                "full_rrmse"
            ]
            <
            best_train
        ):

            best_train = (
                train_metrics[
                    "full_rrmse"
                ]
            )

            torch.save(
                ckpt,
                out /
                "best_train.pth",
            )

        if (
            val_metrics[
                "full_rrmse"
            ]
            <
            best_val
        ):

            best_val = (
                val_metrics[
                    "full_rrmse"
                ]
            )

            torch.save(
                ckpt,
                out /
                "best_val.pth",
            )

        with open(
            out / "history.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                history,
                f,
                indent=2,
            )

    print()
    print("=" * 100)
    print(
        "C5.1c MICRO-OVERFIT SUMMARY"
    )
    print("=" * 100)

    best_train_row = min(
        history,
        key=lambda r:
            r[
                "train_full_rrmse"
            ],
    )

    best_val_row = min(
        history,
        key=lambda r:
            r[
                "val_full_rrmse"
            ],
    )

    print(
        "best train full RRMSE =",
        f"{best_train_row['train_full_rrmse']:.6f}",
        "@ epoch",
        best_train_row["epoch"],
    )

    print(
        "best train receiver RRMSE =",
        f"{best_train_row['train_receiver_rrmse']:.6f}",
    )

    print(
        "best val full RRMSE =",
        f"{best_val_row['val_full_rrmse']:.6f}",
        "@ epoch",
        best_val_row["epoch"],
    )

    print(
        "best val receiver RRMSE =",
        f"{best_val_row['val_receiver_rrmse']:.6f}",
    )

    if (
        best_train_row[
            "train_full_rrmse"
        ]
        < 0.20
    ):

        print()
        print(
            "[PASS] high-mode FNO "
            "can fit the wavefield."
        )

    else:

        print()
        print(
            "[WARN] high-mode FNO "
            "did not reach "
            "train full RRMSE < 0.20."
        )

        print(
            "Do not scale training yet."
        )


if __name__ == "__main__":
    main()
