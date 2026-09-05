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
    ss = 0.0
    n = 0

    for p in files:
        with np.load(p, allow_pickle=True) as z:
            w = z["wavefields"].astype(np.complex64)

        a = np.abs(w).astype(np.float64)

        ss += float(np.sum(a * a))
        n += int(a.size)

    return math.sqrt(
        ss / max(n, 1)
    )


def estimate_speed_stats(raw_root):
    files = sorted(
        (Path(raw_root) / "train").glob("*.npz"),
        key=numeric_key,
    )

    if not files:
        raise RuntimeError(
            f"No raw train npz under {raw_root}"
        )

    total = 0.0
    total2 = 0.0
    n = 0

    for p in files:
        with np.load(p, allow_pickle=True) as z:
            x = z["target_480"].astype(np.float64)

        total += float(x.sum())
        total2 += float((x * x).sum())
        n += int(x.size)

    mean = total / n

    var = (
        total2 / n
        -
        mean * mean
    )

    std = math.sqrt(
        max(var, 1e-12)
    )

    return mean, std, len(files)


# ============================================================
# Dataset
# ============================================================

class AdaMgNODataset(Dataset):

    def __init__(
        self,
        root,
        split,
        background_path,
        wave_scale,
        speed_mean,
        speed_std,
        max_images=0,
    ):
        self.root = Path(root)
        self.wave_scale = float(wave_scale)

        self.speed_mean = float(speed_mean)
        self.speed_std = float(speed_std)

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

        for p in files:

            with np.load(
                p,
                allow_pickle=True,
            ) as z:

                src = (
                    z["src_indices"]
                    .astype(np.int64)
                )

                ns = int(
                    z["wavefields"].shape[0]
                )

            if not np.array_equal(
                src,
                self.bg_src_indices,
            ):
                raise RuntimeError(
                    f"source geometry mismatch: {p}"
                )

            for s in range(ns):
                self.items.append(
                    (p, s)
                )

        print(
            f"[Dataset] split={split} "
            f"images={len(files)} "
            f"pairs={len(self.items)} "
            f"speed_mean={self.speed_mean:.6f} "
            f"speed_std={self.speed_std:.6f} "
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

        wave = d["wavefields"][s]

        bg = self.background[s]

        dobs = d["dobs"][s]

        rec = d["rec"]

        # ----------------------------------------------------
        # Paper uses mean-variance standardization.
        # Here statistics are recomputed on CURRENT 300 train
        # instead of blindly using paper's 1488.39 / 27.53.
        # ----------------------------------------------------

        speed_norm = (
            speed - self.speed_mean
        ) / self.speed_std

        speed_norm = (
            speed_norm[None]
            .astype(np.float32)
        )

        bg_2ch = np.stack(
            [
                bg.real,
                bg.imag,
            ],
            axis=0,
        ).astype(np.float32)

        bg_2ch /= self.wave_scale

        wave_2ch = np.stack(
            [
                wave.real,
                wave.imag,
            ],
            axis=0,
        ).astype(np.float32)

        wave_2ch /= self.wave_scale

        dobs_2ch = np.stack(
            [
                dobs.real,
                dobs.imag,
            ],
            axis=0,
        ).astype(np.float32)

        dobs_2ch /= self.wave_scale

        return {
            "speed":
                torch.from_numpy(
                    speed_norm
                ),

            "background":
                torch.from_numpy(
                    bg_2ch
                ),

            "target":
                torch.from_numpy(
                    wave_2ch
                ),

            "dobs":
                torch.from_numpy(
                    dobs_2ch
                ),

            "rec":
                torch.from_numpy(
                    rec
                ),
        }


# ============================================================
# AdaConv
# ============================================================

class AdaConv(nn.Module):
    """
    Diff-ANO Eq.(21)-style adaptive convolution:

        MLP(Filter_X * X)  *  (Filter_Y * Y)

    X: coefficient/sound-speed latent
    Y: wavefield/residual latent
    """

    def __init__(
        self,
        channels,
    ):
        super().__init__()

        self.filter_x = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
            bias=True,
        )

        self.filter_y = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
            bias=False,
        )

        # two hidden MLP layers + output projection
        self.mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
            ),
        )

        # Stable initialization:
        # start adaptive multiplier near 1.
        last = self.mlp[-1]

        nn.init.zeros_(
            last.weight
        )

        nn.init.ones_(
            last.bias
        )

    def forward(
        self,
        coeff,
        primary,
    ):
        adaptive = self.mlp(
            self.filter_x(coeff)
        )

        field = self.filter_y(
            primary
        )

        return (
            adaptive
            *
            field
        )


# ============================================================
# PDE-informed smoothing level
# ============================================================

class AdaMgLevel(nn.Module):

    def __init__(
        self,
        channels,
    ):
        super().__init__()

        # K_h(X, u)
        self.K = AdaConv(
            channels
        )

        # S_h(X, residual)
        self.S = AdaConv(
            channels
        )

    def residual(
        self,
        coeff,
        u,
        f,
    ):
        return (
            f
            -
            self.K(
                coeff,
                u,
            )
        )

    def smooth(
        self,
        coeff,
        u,
        f,
    ):
        r = self.residual(
            coeff,
            u,
            f,
        )

        u = (
            u
            +
            self.S(
                coeff,
                r,
            )
        )

        return u


# ============================================================
# Seven-level V-cycle
# ============================================================

class AdaVcycle(nn.Module):

    def __init__(
        self,
        channels,
        levels=7,
    ):
        super().__init__()

        self.channels = channels
        self.levels = levels

        self.level_ops = nn.ModuleList(
            [
                AdaMgLevel(
                    channels
                )
                for _ in range(
                    levels
                )
            ]
        )

        # 3x3, stride 2, NO padding:
        #
        # 480 -> 239 -> 119 -> 59
        #     -> 29 -> 14 -> 6
        #
        # matching Diff-ANO's stated hierarchy.
        self.u_restrict = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    bias=False,
                )
                for _ in range(
                    levels - 1
                )
            ]
        )

        self.r_restrict = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    bias=False,
                )
                for _ in range(
                    levels - 1
                )
            ]
        )

        self.prolong = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    channels,
                    channels,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    bias=False,
                )
                for _ in range(
                    levels - 1
                )
            ]
        )

    def build_coeff_pyramid(
        self,
        coeff,
    ):
        pyr = [coeff]

        x = coeff

        for _ in range(
            self.levels - 1
        ):
            x = F.avg_pool2d(
                x,
                kernel_size=3,
                stride=2,
                padding=0,
            )

            pyr.append(x)

        return pyr

    def forward(
        self,
        u0,
        f0,
        coeff0,
    ):

        coeffs = (
            self.build_coeff_pyramid(
                coeff0
            )
        )

        u_list = []
        f_list = []

        u = u0
        f = f0

        # ----------------------------------------------------
        # downward pass
        # ----------------------------------------------------

        for level in range(
            self.levels - 1
        ):

            coeff = coeffs[level]

            # pre-smoothing
            u = (
                self.level_ops[level]
                .smooth(
                    coeff,
                    u,
                    f,
                )
            )

            u_list.append(u)
            f_list.append(f)

            r = (
                self.level_ops[level]
                .residual(
                    coeff,
                    u,
                    f,
                )
            )

            u = self.u_restrict[
                level
            ](u)

            f = self.r_restrict[
                level
            ](r)

        # coarsest level
        u = (
            self.level_ops[-1]
            .smooth(
                coeffs[-1],
                u,
                f,
            )
        )

        # ----------------------------------------------------
        # upward pass
        # ----------------------------------------------------

        for level in range(
            self.levels - 2,
            -1,
            -1,
        ):

            u_fine = u_list[level]
            f_fine = f_list[level]

            target_size = (
                u_fine.shape
            )

            correction = (
                self.prolong[level](
                    u,
                    output_size=
                        target_size,
                )
            )

            u = (
                u_fine
                +
                correction
            )

            # post-smoothing
            u = (
                self.level_ops[level]
                .smooth(
                    coeffs[level],
                    u,
                    f_fine,
                )
            )

        return u


# ============================================================
# Diff-ANO-style recurrent MgNO
# ============================================================

class AdaMgNOWavefield(nn.Module):

    def __init__(
        self,
        channels=12,
        repeats=4,
        levels=7,
    ):
        super().__init__()

        self.channels = channels
        self.repeats = repeats

        # Three latent states:
        # coefficient X, current solution u, forcing/source-like f
        self.coeff_lift = nn.Conv2d(
            1,
            channels,
            kernel_size=1,
        )

        self.u_lift = nn.Conv2d(
            2,
            channels,
            kernel_size=1,
        )

        self.f_lift = nn.Conv2d(
            2,
            channels,
            kernel_size=1,
        )

        # ONE shared V-cycle core.
        # Recurrently applied `repeats` times.
        self.core = AdaVcycle(
            channels=channels,
            levels=levels,
        )

        self.proj1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

        self.proj2 = nn.Conv2d(
            channels,
            2,
            kernel_size=1,
        )

        # Start exactly from background prediction.
        nn.init.zeros_(
            self.proj2.weight
        )

        nn.init.zeros_(
            self.proj2.bias
        )

    def forward(
        self,
        speed,
        background,
    ):

        coeff = self.coeff_lift(
            speed
        )

        u = self.u_lift(
            background
        )

        f = self.f_lift(
            background
        )

        for _ in range(
            self.repeats
        ):
            u = self.core(
                u,
                f,
                coeff,
            )

        delta = self.proj2(
            F.gelu(
                self.proj1(u)
            )
        )

        return (
            background
            +
            delta
        )


# ============================================================
# Receiver extraction / evaluation
# ============================================================

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

        speed = batch[
            "speed"
        ].to(device)

        bg = batch[
            "background"
        ].to(device)

        target = batch[
            "target"
        ].to(device)

        dobs = batch[
            "dobs"
        ].to(device)

        rec = batch[
            "rec"
        ].to(device)

        pred = model(
            speed,
            bg,
        )

        fn += float(
            torch.sum(
                (pred-target) ** 2
            ).cpu()
        )

        fd += float(
            torch.sum(
                target ** 2
            ).cpu()
        )

        pred_rec = (
            sample_receivers(
                pred,
                rec,
            )
        )

        rn += float(
            torch.sum(
                (pred_rec-dobs) ** 2
            ).cpu()
        )

        rd += float(
            torch.sum(
                dobs ** 2
            ).cpu()
        )

    return {
        "full_rrmse":
            math.sqrt(
                fn /
                max(fd, 1e-12)
            ),

        "receiver_rrmse":
            math.sqrt(
                rn /
                max(rd, 1e-12)
            ),
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
        "--raw_root",
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
        "--channels",
        type=int,
        default=12,
    )

    ap.add_argument(
        "--repeats",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--levels",
        type=int,
        default=7,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=5e-4,
    )

    ap.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
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

    # --------------------------------------------------------
    # Current-data normalization
    # --------------------------------------------------------

    speed_mean, speed_std, n_stats = (
        estimate_speed_stats(
            args.raw_root
        )
    )

    train_files = sorted(
        (root/"train").glob("*.npz"),
        key=numeric_key,
    )

    if args.max_train_images > 0:
        train_files = train_files[
            :args.max_train_images
        ]

    wave_scale = (
        estimate_wave_scale(
            train_files
        )
    )

    train_set = AdaMgNODataset(
        root=root,
        split="train",
        background_path=
            args.background_path,
        wave_scale=wave_scale,
        speed_mean=speed_mean,
        speed_std=speed_std,
        max_images=
            args.max_train_images,
    )

    val_set = AdaMgNODataset(
        root=root,
        split="val",
        background_path=
            args.background_path,
        wave_scale=wave_scale,
        speed_mean=speed_mean,
        speed_std=speed_std,
        max_images=
            args.max_val_images,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=
            args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    train_eval_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=False,
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

    model = AdaMgNOWavefield(
        channels=
            args.channels,
        repeats=
            args.repeats,
        levels=
            args.levels,
    ).to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("="*100)
    print(
        "C5.1f DIFF-ANO-STYLE "
        "AdaMgNO MICRO-OVERFIT"
    )
    print("="*100)

    print(
        "channels        =",
        args.channels,
    )

    print(
        "shared repeats  =",
        args.repeats,
    )

    print(
        "levels          =",
        args.levels,
    )

    print(
        "hierarchy       =",
        "480 -> 239 -> 119 -> "
        "59 -> 29 -> 14 -> 6",
    )

    print(
        "speed stats N   =",
        n_stats,
    )

    print(
        "speed mean/std  =",
        f"{speed_mean:.6f} / "
        f"{speed_std:.6f}",
    )

    print(
        "wave scale      =",
        f"{wave_scale:.8e}",
    )

    print(
        "parameters      =",
        f"{n_params:,}",
    )

    print(
        "loss            =",
        "full-wavefield MSE only",
    )

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

    print()

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
        weight_decay=
            args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=
                len(train_loader),
            pct_start=0.30,
            anneal_strategy="cos",
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

        loss_sum = 0.0

        for batch in train_loader:

            speed = batch[
                "speed"
            ].to(device)

            bg = batch[
                "background"
            ].to(device)

            target = batch[
                "target"
            ].to(device)

            pred = model(
                speed,
                bg,
            )

            # Diff-ANO Eq.(35):
            # pure full-wavefield L2 supervision
            loss = F.mse_loss(
                pred,
                target,
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
            scheduler.step()

            loss_sum += float(
                loss.detach().cpu()
            )

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
                max(
                    len(train_loader),
                    1,
                ),

            "train_full":
                tr["full_rrmse"],

            "train_receiver":
                tr["receiver_rrmse"],

            "val_full":
                va["full_rrmse"],

            "val_receiver":
                va["receiver_rrmse"],

            "lr":
                scheduler
                .get_last_lr()[0],
        }

        history.append(row)

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"loss={row['loss']:.6e} "
            f"lr={row['lr']:.3e} | "
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

            "speed_mean":
                speed_mean,

            "speed_std":
                speed_std,

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

        if (
            row["val_full"]
            <
            best_val
        ):
            best_val = (
                row["val_full"]
            )

            torch.save(
                ckpt,
                out/"best_val.pth",
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
        key=lambda r:
            r["train_full"],
    )

    bestv = min(
        history,
        key=lambda r:
            r["val_full"],
    )

    print()
    print("="*100)
    print(
        "C5.1f AdaMgNO MICRO-OVERFIT SUMMARY"
    )
    print("="*100)

    print(
        "initial train full =",
        f"{initial_train['full_rrmse']:.6f}",
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

    print(
        "best val full      =",
        f"{bestv['val_full']:.6f}",
        "@ epoch",
        bestv["epoch"],
    )

    if best["train_full"] < 0.10:

        print(
            "[STRONG PASS] AdaMgNO "
            "micro-overfit confirmed."
        )

    elif best["train_full"] < 0.20:

        print(
            "[PASS] AdaMgNO "
            "micro-overfit confirmed."
        )

    elif best["train_full"] < 0.30:

        print(
            "[PARTIAL] AdaMgNO improved "
            "but capacity remains limited."
        )

    else:

        print(
            "[FAIL] AdaMgNO-I still "
            "cannot fit one-image wavefields."
        )


if __name__ == "__main__":
    main()
