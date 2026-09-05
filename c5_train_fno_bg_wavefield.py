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


def numeric_key(p):
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else -1


def estimate_wave_scale(files):
    ss = 0.0
    n = 0

    for p in files:
        with np.load(p, allow_pickle=True) as z:
            w = z["wavefields"].astype(np.complex64)

        ss += float(
            np.sum(
                np.abs(w).astype(np.float64) ** 2
            )
        )
        n += int(w.size)

    return math.sqrt(
        ss / max(n, 1)
    )


class WavefieldBGDataset(Dataset):

    def __init__(
        self,
        root,
        split,
        background_path,
        wave_scale,
        max_images=0,
    ):
        self.root = Path(root)
        self.wave_scale = float(wave_scale)

        files = sorted(
            (self.root / split).glob("*.npz"),
            key=numeric_key,
        )

        if max_images > 0:
            files = files[:max_images]

        self.files = files

        bg = np.load(
            background_path,
            allow_pickle=True,
        )

        self.background = (
            bg["background_field"]
            .astype(np.complex64)
        )

        self.bg_src_indices = (
            bg["src_indices"]
            .astype(np.int64)
        )

        self.items = []

        for p in self.files:

            with np.load(
                p,
                allow_pickle=True,
            ) as z:

                ns = int(
                    z["wavefields"].shape[0]
                )

                src = (
                    z["src_indices"]
                    .astype(np.int64)
                )

            if not np.array_equal(
                src,
                self.bg_src_indices,
            ):
                raise RuntimeError(
                    f"Source geometry mismatch: {p}"
                )

            for s in range(ns):
                self.items.append(
                    (p, s)
                )

        print(
            f"[Dataset] split={split} "
            f"images={len(files)} "
            f"pairs={len(self.items)} "
            f"wave_scale={self.wave_scale:.8e}"
        )

    def __len__(self):
        return len(self.items)

    @lru_cache(maxsize=8)
    def _load(self, path_str):

        with np.load(
            path_str,
            allow_pickle=True,
        ) as z:

            return {
                "speed":
                    z["target_480"]
                    .astype(np.float32),

                "wavefields":
                    z["wavefields"]
                    .astype(np.complex64),

                "dobs":
                    z["dobs_complex"]
                    .astype(np.complex64),

                "rec":
                    z["rec_indices"]
                    .astype(np.int64),
            }

    def __getitem__(self, idx):

        p, s = self.items[idx]

        d = self._load(
            str(p)
        )

        speed = d["speed"]

        target = (
            d["wavefields"][s]
        )

        bg = (
            self.background[s]
        )

        dobs = (
            d["dobs"][s]
        )

        rec = d["rec"]

        # Current project normalization:
        # speed = norm * 102.5 + 1502.5
        speed_norm = (
            speed - 1502.5
        ) / 102.5

        bg_re = (
            bg.real
            / self.wave_scale
        )

        bg_im = (
            bg.imag
            / self.wave_scale
        )

        inp = np.stack(
            [
                speed_norm,
                bg_re,
                bg_im,
            ],
            axis=0,
        ).astype(np.float32)

        target_2ch = np.stack(
            [
                target.real,
                target.imag,
            ],
            axis=0,
        ).astype(np.float32)

        target_2ch /= self.wave_scale

        bg_2ch = np.stack(
            [
                bg.real,
                bg.imag,
            ],
            axis=0,
        ).astype(np.float32)

        bg_2ch /= self.wave_scale

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
                torch.from_numpy(target_2ch),

            "background":
                torch.from_numpy(bg_2ch),

            "dobs":
                torch.from_numpy(dobs_2ch),

            "rec":
                torch.from_numpy(rec),
        }


class SpectralConv2d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
    ):
        super().__init__()

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
            scale * torch.randn(
                *shape,
                dtype=torch.cfloat,
            )
        )

        self.weight_neg = nn.Parameter(
            scale * torch.randn(
                *shape,
                dtype=torch.cfloat,
            )
        )

    @staticmethod
    def compl_mul2d(x, w):
        return torch.einsum(
            "bixy,ioxy->boxy",
            x,
            w,
        )

    def forward(self, x):

        b, _, h, w = x.shape

        x_ft = torch.fft.rfft2(
            x,
            norm="ortho",
        )

        out_ft = torch.zeros(
            b,
            self.weight_pos.shape[1],
            h,
            w // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        mx = min(
            self.modes,
            h,
        )

        my = min(
            self.modes,
            w // 2 + 1,
        )

        out_ft[
            :, :, :mx, :my
        ] = self.compl_mul2d(
            x_ft[
                :, :, :mx, :my
            ],
            self.weight_pos[
                :, :, :mx, :my
            ],
        )

        out_ft[
            :, :, -mx:, :my
        ] = self.compl_mul2d(
            x_ft[
                :, :, -mx:, :my
            ],
            self.weight_neg[
                :, :, :mx, :my
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

        self.spectral = (
            SpectralConv2d(
                width,
                width,
                modes,
            )
        )

        # local pathway preserves
        # high-frequency input information
        self.local = nn.Conv2d(
            width,
            width,
            kernel_size=1,
        )

        self.norm = nn.GroupNorm(
            8,
            width,
        )

    def forward(self, x):

        y = (
            self.spectral(x)
            +
            self.local(x)
        )

        y = self.norm(y)

        return F.gelu(y)


class BackgroundFNO(nn.Module):

    def __init__(
        self,
        modes=25,
        width=32,
        depth=4,
    ):
        super().__init__()

        self.lift = nn.Conv2d(
            3,
            width,
            kernel_size=1,
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

        self.proj1 = nn.Conv2d(
            width,
            width,
            kernel_size=1,
        )

        self.proj2 = nn.Conv2d(
            width,
            2,
            kernel_size=1,
        )

        # Start exactly from homogeneous
        # background prediction.
        nn.init.zeros_(
            self.proj2.weight
        )

        nn.init.zeros_(
            self.proj2.bias
        )

    def forward(
        self,
        x,
        background,
    ):

        z = self.lift(x)

        for block in self.blocks:
            z = block(z)

        z = F.gelu(
            self.proj1(z)
        )

        delta = self.proj2(z)

        return (
            background
            +
            delta
        )


def sample_receivers(
    field,
    rec,
):

    b, c, h, w = (
        field.shape
    )

    flat = field.reshape(
        b,
        c,
        h * w,
    )

    idx = (
        rec[:, :, 0]
        *
        w
        +
        rec[:, :, 1]
    )

    idx = (
        idx
        .unsqueeze(1)
        .expand(
            -1,
            c,
            -1,
        )
    )

    return torch.gather(
        flat,
        2,
        idx,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    fn = 0.0
    fd = 0.0

    rn = 0.0
    rd = 0.0

    for batch in loader:

        x = batch["input"].to(
            device
        )

        y = batch["target"].to(
            device
        )

        bg = batch[
            "background"
        ].to(
            device
        )

        dobs = batch["dobs"].to(
            device
        )

        rec = batch["rec"].to(
            device
        )

        pred = model(
            x,
            bg,
        )

        fn += float(
            torch.sum(
                (pred-y)**2
            ).cpu()
        )

        fd += float(
            torch.sum(
                y**2
            ).cpu()
        )

        pred_rec = sample_receivers(
            pred,
            rec,
        )

        rn += float(
            torch.sum(
                (pred_rec-dobs)**2
            ).cpu()
        )

        rd += float(
            torch.sum(
                dobs**2
            ).cpu()
        )

    return {
        "full_rrmse":
            math.sqrt(
                fn/max(fd,1e-12)
            ),

        "receiver_rrmse":
            math.sqrt(
                rn/max(rd,1e-12)
            ),
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data_root",
        required=True,
    )

    ap.add_argument(
        "--background_path",
        required=True,
    )

    ap.add_argument(
        "--output_dir",
        required=True,
    )

    ap.add_argument(
        "--modes",
        type=int,
        default=25,
    )

    ap.add_argument(
        "--width",
        type=int,
        default=32,
    )

    ap.add_argument(
        "--depth",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=60,
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    ap.add_argument(
        "--receiver_weight",
        type=float,
        default=0.05,
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

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

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
        (root/"train").glob(
            "*.npz"
        ),
        key=numeric_key,
    )

    if args.max_train_images > 0:
        train_files = (
            train_files[
                :args.max_train_images
            ]
        )

    wave_scale = (
        estimate_wave_scale(
            train_files
        )
    )

    train_set = (
        WavefieldBGDataset(
            root,
            "train",
            args.background_path,
            wave_scale,
            max_images=
                args.max_train_images,
        )
    )

    val_set = (
        WavefieldBGDataset(
            root,
            "val",
            args.background_path,
            wave_scale,
            max_images=
                args.max_val_images,
        )
    )

    train_loader = DataLoader(
        train_set,
        batch_size=
            args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    train_eval_loader = (
        DataLoader(
            train_set,
            batch_size=1,
            shuffle=False,
        )
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
    )

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    model = BackgroundFNO(
        modes=args.modes,
        width=args.width,
        depth=args.depth,
    ).to(device)

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("="*100)
    print(
        "C5.1d PAPER-LIKE "
        "BACKGROUND FNO"
    )
    print("="*100)

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

    print(
        "wave_scale =",
        f"{wave_scale:.8e}",
    )

    print(
        "parameters =",
        f"{params:,}",
    )

    print()

    initial_train = evaluate(
        model,
        train_eval_loader,
        device,
    )

    initial_val = evaluate(
        model,
        val_loader,
        device,
    )

    print(
        "INITIAL train full="
        f"{initial_train['full_rrmse']:.6f} "
        "rec="
        f"{initial_train['receiver_rrmse']:.6f}"
    )

    print(
        "INITIAL val   full="
        f"{initial_val['full_rrmse']:.6f} "
        "rec="
        f"{initial_val['receiver_rrmse']:.6f}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-6,
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

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        loss_sum = 0.0

        for batch in train_loader:

            x = batch["input"].to(
                device
            )

            y = batch["target"].to(
                device
            )

            bg = batch[
                "background"
            ].to(
                device
            )

            dobs = batch[
                "dobs"
            ].to(
                device
            )

            rec = batch[
                "rec"
            ].to(
                device
            )

            pred = model(
                x,
                bg,
            )

            full_loss = (
                F.mse_loss(
                    pred,
                    y,
                )
            )

            pred_rec = (
                sample_receivers(
                    pred,
                    rec,
                )
            )

            rec_loss = (
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
                rec_loss
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

            loss_sum += float(
                loss.detach().cpu()
            )

        scheduler.step()

        tr = evaluate(
            model,
            train_eval_loader,
            device,
        )

        va = evaluate(
            model,
            val_loader,
            device,
        )

        row = {
            "epoch":
                epoch,

            "loss":
                loss_sum /
                len(train_loader),

            "train_full":
                tr["full_rrmse"],

            "train_receiver":
                tr[
                    "receiver_rrmse"
                ],

            "val_full":
                va["full_rrmse"],

            "val_receiver":
                va[
                    "receiver_rrmse"
                ],
        }

        history.append(row)

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"loss={row['loss']:.6e} | "
            f"train full="
            f"{row['train_full']:.4f} "
            f"rec="
            f"{row['train_receiver']:.4f} | "
            f"val full="
            f"{row['val_full']:.4f} "
            f"rec="
            f"{row['val_receiver']:.4f}"
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
            out/"last.pth",
        )

        if (
            row["train_full"]
            <
            best_train
        ):
            best_train = (
                row["train_full"]
            )

            torch.save(
                ckpt,
                out/"best_train.pth",
            )

        with open(
            out/"history.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                history,
                f,
                indent=2,
            )

    best = min(
        history,
        key=lambda x:
            x["train_full"],
    )

    print()
    print("="*100)
    print(
        "C5.1d MICRO-OVERFIT SUMMARY"
    )
    print("="*100)

    print(
        "initial train full =",
        f"{initial_train['full_rrmse']:.6f}",
    )

    print(
        "initial train rec  =",
        f"{initial_train['receiver_rrmse']:.6f}",
    )

    print(
        "best train full    =",
        f"{best['train_full']:.6f}",
        "@ epoch",
        best["epoch"],
    )

    print(
        "best train rec     =",
        f"{best['train_receiver']:.6f}",
    )

    print(
        "corresponding val  =",
        f"full {best['val_full']:.6f}, "
        f"rec {best['val_receiver']:.6f}",
    )

    if best["train_full"] < 0.20:
        print(
            "[PASS] FNO-BG "
            "micro-overfit."
        )

    elif best["train_full"] < 0.50:
        print(
            "[PARTIAL] FNO-BG "
            "beats simple baselines "
            "but is not yet sufficient."
        )

    else:
        print(
            "[FAIL] FNO-BG still "
            "cannot fit one-image "
            "wavefields."
        )


if __name__ == "__main__":
    main()
