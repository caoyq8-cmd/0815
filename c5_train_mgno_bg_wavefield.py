import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from c5_train_fno_bg_wavefield import (
    WavefieldBGDataset,
    estimate_wave_scale,
    numeric_key,
    sample_receivers,
)


# ============================================================
# MgNO core
# ============================================================

def conv3(ch, stride=1, bias=False):
    return nn.Conv2d(
        ch,
        ch,
        kernel_size=3,
        stride=stride,
        padding=1,
        padding_mode="reflect",
        bias=bias,
    )


class MgIte(nn.Module):
    """
    Learned smoothing:
        u <- u + S(f - A(u))
    """

    def __init__(self, channels):
        super().__init__()

        self.A = conv3(
            channels,
            bias=False,
        )

        self.S = conv3(
            channels,
            bias=True,
        )

    def forward(self, out):
        u, f = out

        residual = (
            f
            -
            self.A(u)
        )

        u = (
            u
            +
            self.S(residual)
        )

        return u, f


class MgIteInit(nn.Module):
    """
    First smoothing operation:
        u = S(f)
    """

    def __init__(self, channels):
        super().__init__()

        self.S = conv3(
            channels,
            bias=True,
        )

    def forward(self, f):

        u = self.S(f)

        return u, f


class Restrict(nn.Module):
    """
    Fine -> coarse using learned residual restriction.

        residual = f - A(u)
        f_c = R(residual)
        u_c = Pi(u)
    """

    def __init__(
        self,
        channels,
        use_res=True,
    ):
        super().__init__()

        self.use_res = use_res

        self.A = conv3(
            channels,
            bias=False,
        )

        self.Pi = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode="reflect",
            bias=False,
        )

        self.R = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode="reflect",
            bias=False,
        )

    def forward(self, out):

        u, f = out

        if self.use_res:
            f_coarse = self.R(
                f - self.A(u)
            )
        else:
            f_coarse = self.R(f)

        u_coarse = self.Pi(u)

        return (
            u_coarse,
            f_coarse,
        )


class MgLevelPre(nn.Module):

    def __init__(
        self,
        channels,
        n_iters,
        is_first=False,
    ):
        super().__init__()

        layers = []

        if is_first:
            layers.append(
                MgIteInit(channels)
            )

            n_remaining = max(
                n_iters - 1,
                0,
            )

        else:
            n_remaining = n_iters

        for _ in range(
            n_remaining
        ):
            layers.append(
                MgIte(channels)
            )

        self.layers = nn.ModuleList(
            layers
        )

        self.is_first = is_first

    def forward(self, out):

        for i, layer in enumerate(
            self.layers
        ):

            if (
                self.is_first
                and
                i == 0
            ):
                out = layer(out)
            else:
                out = layer(out)

        return out


class MgLevelPost(nn.Module):

    def __init__(
        self,
        channels,
        n_iters,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                MgIte(channels)
                for _ in range(
                    n_iters
                )
            ]
        )

    def forward(self, out):

        for layer in self.layers:
            out = layer(out)

        return out


class MgConv480(nn.Module):
    """
    Resolution-agnostic V-cycle.

    Designed for:
        480 -> 240 -> 120 -> 60 -> 30

    This follows the MgNO idea but removes
    fixed 64x64 / 101x101 assumptions.
    """

    def __init__(
        self,
        channels=12,
        levels=5,
        pre_iters=1,
        post_iters=1,
        use_res=True,
    ):
        super().__init__()

        self.channels = channels
        self.levels = levels

        self.pre = nn.ModuleList()

        for level in range(
            levels
        ):
            self.pre.append(
                MgLevelPre(
                    channels,
                    n_iters=pre_iters,
                    is_first=(
                        level == 0
                    ),
                )
            )

        self.restrict = nn.ModuleList(
            [
                Restrict(
                    channels,
                    use_res=use_res,
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
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
                for _ in range(
                    levels - 1
                )
            ]
        )

        self.post = nn.ModuleList(
            [
                MgLevelPost(
                    channels,
                    n_iters=post_iters,
                )
                for _ in range(
                    levels - 1
                )
            ]
        )

        # Dynamic normalization:
        # no hard-coded spatial shape.
        groups = 1

        for g in [
            8, 6, 4, 3, 2
        ]:
            if channels % g == 0:
                groups = g
                break

        self.norms = nn.ModuleList(
            [
                nn.GroupNorm(
                    groups,
                    channels,
                )
                for _ in range(
                    levels - 1
                )
            ]
        )

    def forward(self, f):

        out_list = []

        # --------------------------------------------
        # Downward V-cycle
        # --------------------------------------------

        out = self.pre[0](f)

        out_list.append(out)

        for level in range(
            1,
            self.levels
        ):

            out = self.restrict[
                level - 1
            ](out)

            out = self.pre[
                level
            ](out)

            out_list.append(out)

        # --------------------------------------------
        # Upward V-cycle
        # --------------------------------------------

        for level in range(
            self.levels - 2,
            -1,
            -1,
        ):

            u_fine, f_fine = (
                out_list[level]
            )

            u_coarse = (
                out_list[
                    level + 1
                ][0]
            )

            correction = (
                self.prolong[
                    level
                ](
                    u_coarse
                )
            )

            # Safety for odd dimensions.
            if (
                correction.shape[-2:]
                !=
                u_fine.shape[-2:]
            ):
                correction = (
                    F.interpolate(
                        correction,
                        size=
                            u_fine.shape[
                                -2:
                            ],
                        mode="bilinear",
                        align_corners=False,
                    )
                )

            u_fine = (
                u_fine
                +
                correction
            )

            u_fine = self.norms[
                level
            ](
                u_fine
            )

            out_list[
                level
            ] = self.post[
                level
            ](
                (
                    u_fine,
                    f_fine,
                )
            )

        return (
            out_list[0][0]
        )


class MgNOWavefieldBG(nn.Module):
    """
    MgNO-I-like USCT wavefield model.

    Input:
        speed_norm,
        Re(u_bg),
        Im(u_bg)

    Output:
        u_bg + learned correction
    """

    def __init__(
        self,
        channels=12,
        num_vcycles=4,
        levels=5,
        pre_iters=1,
        post_iters=1,
    ):
        super().__init__()

        self.lift = nn.Conv2d(
            3,
            channels,
            kernel_size=1,
        )

        self.cycles = nn.ModuleList(
            [
                MgConv480(
                    channels=channels,
                    levels=levels,
                    pre_iters=pre_iters,
                    post_iters=post_iters,
                    use_res=True,
                )
                for _ in range(
                    num_vcycles
                )
            ]
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

        # Start exactly from u_bg.
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

        for cycle in self.cycles:
            z = F.gelu(
                cycle(z)
            )

        z = F.gelu(
            self.proj1(z)
        )

        delta = self.proj2(z)

        return (
            background
            +
            delta
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

        x = batch[
            "input"
        ].to(device)

        target = batch[
            "target"
        ].to(device)

        bg = batch[
            "background"
        ].to(device)

        dobs = batch[
            "dobs"
        ].to(device)

        rec = batch[
            "rec"
        ].to(device)

        pred = model(
            x,
            bg,
        )

        full_num += float(
            torch.sum(
                (
                    pred
                    -
                    target
                ) ** 2
            ).cpu()
        )

        full_den += float(
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

        rec_num += float(
            torch.sum(
                (
                    pred_rec
                    -
                    dobs
                ) ** 2
            ).cpu()
        )

        rec_den += float(
            torch.sum(
                dobs ** 2
            ).cpu()
        )

    return {
        "full_rrmse":
            float(
                np.sqrt(
                    full_num
                    /
                    max(
                        full_den,
                        1e-12,
                    )
                )
            ),

        "receiver_rrmse":
            float(
                np.sqrt(
                    rec_num
                    /
                    max(
                        rec_den,
                        1e-12,
                    )
                )
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
        "--channels",
        type=int,
        default=12,
    )

    ap.add_argument(
        "--num_vcycles",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--levels",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--pre_iters",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--post_iters",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=80,
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
        (
            root / "train"
        ).glob("*.npz"),
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    train_eval_loader = (
        DataLoader(
            train_set,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
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

    model = MgNOWavefieldBG(
        channels=args.channels,
        num_vcycles=
            args.num_vcycles,
        levels=args.levels,
        pre_iters=
            args.pre_iters,
        post_iters=
            args.post_iters,
    ).to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("=" * 100)
    print(
        "C5.1e MgNO-I-LIKE "
        "BACKGROUND WAVEFIELD"
    )
    print("=" * 100)

    print(
        "channels       =",
        args.channels,
    )

    print(
        "num_vcycles    =",
        args.num_vcycles,
    )

    print(
        "levels         =",
        args.levels,
    )

    print(
        "grid hierarchy =",
        "480 -> 240 -> 120 -> 60 -> 30",
    )

    print(
        "pre/post       =",
        args.pre_iters,
        "/",
        args.post_iters,
    )

    print(
        "wave_scale     =",
        f"{wave_scale:.8e}",
    )

    print(
        "parameters     =",
        f"{n_params:,}",
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

            x = batch[
                "input"
            ].to(device)

            target = batch[
                "target"
            ].to(device)

            bg = batch[
                "background"
            ].to(device)

            dobs = batch[
                "dobs"
            ].to(device)

            rec = batch[
                "rec"
            ].to(device)

            pred = model(
                x,
                bg,
            )

            full_loss = (
                F.mse_loss(
                    pred,
                    target,
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
                loss_sum /
                max(
                    len(train_loader),
                    1,
                ),

            "train_full":
                train_metrics[
                    "full_rrmse"
                ],

            "train_receiver":
                train_metrics[
                    "receiver_rrmse"
                ],

            "val_full":
                val_metrics[
                    "full_rrmse"
                ],

            "val_receiver":
                val_metrics[
                    "receiver_rrmse"
                ],

            "lr":
                scheduler
                .get_last_lr()[0],
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
            out / "last.pth",
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
                out /
                "best_train.pth",
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

    best = min(
        history,
        key=lambda r:
            r["train_full"],
    )

    print()
    print("=" * 100)
    print(
        "C5.1e MgNO MICRO-OVERFIT SUMMARY"
    )
    print("=" * 100)

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

    if best["train_full"] < 0.10:

        print(
            "[STRONG PASS] MgNO-like "
            "full-field capacity confirmed."
        )

    elif best["train_full"] < 0.20:

        print(
            "[PASS] MgNO-like "
            "micro-overfit confirmed."
        )

    elif best["train_full"] < 0.30:

        print(
            "[PARTIAL] roughly comparable "
            "to FNO-BG; do not scale yet."
        )

    else:

        print(
            "[FAIL] MgNO-like did not "
            "improve full-field capacity."
        )


if __name__ == "__main__":
    main()
