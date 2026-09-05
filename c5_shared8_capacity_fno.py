import os
import re
import csv
import json
import math
import random
import argparse
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================================
# Utilities
# ======================================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numeric_key(path):
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


def complex_rrmse_np(pred, target, eps=1e-12):
    pred = np.asarray(pred)
    target = np.asarray(target)

    num = np.sqrt(np.mean(np.abs(pred - target) ** 2))
    den = np.sqrt(np.mean(np.abs(target) ** 2)) + eps
    return float(num / den)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ======================================================================================
# FNO
#
# This architecture intentionally matches the C5.1g parameter count:
#
# modes=25, width=32, depth=4
# -> 5,125,730 parameters
#
# Input:
#   [speed_norm, bg_real_norm, bg_imag_norm]
#
# Output:
#   [wave_real_norm, wave_imag_norm]
# ======================================================================================

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / math.sqrt(in_channels * out_channels)

        self.weight_pos = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes,
                modes,
                dtype=torch.cfloat,
            )
        )

        self.weight_neg = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes,
                modes,
                dtype=torch.cfloat,
            )
        )

    @staticmethod
    def compl_mul2d(x, weights):
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
            dtype=torch.cfloat,
            device=x.device,
        )

        mx = min(self.modes, h)
        my = min(self.modes, w // 2 + 1)

        out_ft[:, :, :mx, :my] = self.compl_mul2d(
            x_ft[:, :, :mx, :my],
            self.weight_pos[:, :, :mx, :my],
        )

        out_ft[:, :, -mx:, :my] = self.compl_mul2d(
            x_ft[:, :, -mx:, :my],
            self.weight_neg[:, :, :mx, :my],
        )

        return torch.fft.irfft2(
            out_ft,
            s=(h, w),
            norm="ortho",
        )


class FNOBlock(nn.Module):
    def __init__(self, width, modes):
        super().__init__()

        self.spectral = SpectralConv2d(
            width,
            width,
            modes,
        )

        self.pointwise = nn.Conv2d(
            width,
            width,
            kernel_size=1,
        )

        # width=32 -> exactly the same normalization family used previously
        self.norm = nn.GroupNorm(
            num_groups=8,
            num_channels=width,
        )

    def forward(self, x):
        y = (
            self.spectral(x)
            + self.pointwise(x)
        )

        y = self.norm(y)
        y = F.gelu(y)

        return x + y


class FNOBackgroundWavefield(nn.Module):
    def __init__(
        self,
        modes=25,
        width=32,
        depth=4,
    ):
        super().__init__()

        # 3 channels:
        # speed + background real + background imag
        self.lift = nn.Conv2d(
            3,
            width,
            kernel_size=1,
        )

        self.blocks = nn.ModuleList(
            [
                FNOBlock(
                    width=width,
                    modes=modes,
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


# ======================================================================================
# Background loading
# ======================================================================================

def _to_complex_wavefield_array(arr, h, w):
    """
    Convert possible background formats into complex [Nsrc,H,W].
    """

    a = np.asarray(arr)

    # complex [N,H,W]
    if np.iscomplexobj(a):
        if a.ndim == 3 and a.shape[-2:] == (h, w):
            return a.astype(np.complex64)

        if (
            a.ndim == 4
            and a.shape[0] == 1
            and a.shape[-2:] == (h, w)
        ):
            return a[0].astype(np.complex64)

    # real [N,2,H,W]
    if (
        not np.iscomplexobj(a)
        and a.ndim == 4
        and a.shape[1] == 2
        and a.shape[-2:] == (h, w)
    ):
        return (
            a[:, 0]
            + 1j * a[:, 1]
        ).astype(np.complex64)

    # real [2,N,H,W]
    if (
        not np.iscomplexobj(a)
        and a.ndim == 4
        and a.shape[0] == 2
        and a.shape[-2:] == (h, w)
    ):
        return (
            a[0]
            + 1j * a[1]
        ).astype(np.complex64)

    return None


def load_background_wavefields(
    path,
    target_src_indices,
    h,
    w,
):
    z = np.load(
        path,
        allow_pickle=True,
    )

    print("[Background] keys =", list(z.keys()))

    preferred = [
        # IMPORTANT:
        # Diff-ANO uses the homogeneous-medium background Helmholtz
        # wavefield as neural-operator input.
        #
        # In c5_wavefield_baselines.npz:
        #   mean_field       = empirical mean of TRAIN wavefields
        #   background_field = homogeneous-background wavefield
        #
        # NEVER silently prefer mean_field here.
        "background_field",
        "background_wavefields",
        "background_wavefield",
        "wavefields_background",
        "wavefield_background",
        "u_background",
        "u_bg",
        "u_homo",
        "background",
        "wavefields",
    ]

    candidate = None
    candidate_key = None

    # Prefer semantically meaningful keys.
    for key in preferred:
        if key in z:
            value = _to_complex_wavefield_array(
                z[key],
                h,
                w,
            )
            if value is not None:
                candidate = value
                candidate_key = key
                break

    # Otherwise automatically inspect every array.
    if candidate is None:
        for key in z.keys():
            value = _to_complex_wavefield_array(
                z[key],
                h,
                w,
            )
            if value is not None:
                candidate = value
                candidate_key = key
                break

    if candidate is None:
        raise RuntimeError(
            "Could not find a background wavefield array "
            f"with spatial size {(h, w)} in {path}. "
            f"Available keys: {list(z.keys())}"
        )

    print(
        "[Background] selected key =",
        candidate_key,
        "shape =",
        candidate.shape,
    )

    if candidate_key == "mean_field":
        raise RuntimeError(
            "FATAL: mean_field is the empirical TRAIN-wavefield mean, "
            "not the homogeneous background wavefield. "
            "Use background_field."
        )

    target_src_indices = np.asarray(
        target_src_indices,
        dtype=np.int64,
    )

    nsrc = len(target_src_indices)

    # Best case: baseline file also contains source coordinates.
    baseline_src = None

    for key in (
        "src_indices",
        "source_indices",
        "sources",
    ):
        if key in z:
            arr = np.asarray(z[key])

            if (
                arr.ndim == 2
                and arr.shape[1] == 2
            ):
                baseline_src = arr.astype(np.int64)
                print(
                    "[Background] source-index key =",
                    key,
                    "shape =",
                    baseline_src.shape,
                )
                break

    if baseline_src is not None:
        selected = []

        for src in target_src_indices:
            matches = np.where(
                np.all(
                    baseline_src == src[None],
                    axis=1,
                )
            )[0]

            if len(matches) != 1:
                raise RuntimeError(
                    "Unable to uniquely match source "
                    f"{src.tolist()} against baseline source list."
                )

            selected.append(
                candidate[int(matches[0])]
            )

        candidate = np.stack(
            selected,
            axis=0,
        )

    else:
        # Smoke dataset and baseline are expected to use the same ordered 8 sources.
        if candidate.shape[0] < nsrc:
            raise RuntimeError(
                f"Background contains only {candidate.shape[0]} sources, "
                f"but dataset requests {nsrc}."
            )

        candidate = candidate[:nsrc]

        print(
            "[Background] no source-index array found; "
            "using the first",
            nsrc,
            "background wavefields in stored order.",
        )

    if candidate.shape != (
        nsrc,
        h,
        w,
    ):
        raise RuntimeError(
            "Unexpected final background shape: "
            f"{candidate.shape}, expected {(nsrc, h, w)}"
        )

    return candidate.astype(
        np.complex64
    )


# ======================================================================================
# Dataset preparation
# ======================================================================================

def find_train_files(data_root):
    train_dir = Path(data_root) / "train"

    files = sorted(
        train_dir.glob("train_*.npz"),
        key=numeric_key,
    )

    if not files:
        raise FileNotFoundError(
            f"No train_*.npz found under {train_dir}"
        )

    return files


def compute_wave_scale(train_files):
    """
    C5 wavefield scale:
        sqrt(E |Y|^2)

    This is the natural complex RMS normalization.
    """

    sum_sq = 0.0
    count = 0

    for p in train_files:
        with np.load(
            p,
            allow_pickle=True,
        ) as z:
            w = z["wavefields"]

            sum_sq += float(
                np.sum(
                    np.abs(w) ** 2,
                    dtype=np.float64,
                )
            )

            count += int(
                np.prod(w.shape)
            )

    return float(
        np.sqrt(
            sum_sq / max(count, 1)
        )
    )


def prepare_one_image_all_sources(
    data_root,
    background_path,
    speed_mean,
    speed_std,
    forced_wave_scale,
):
    train_files = find_train_files(
        data_root
    )

    # Important:
    # scale is computed using ALL train files,
    # while capacity fitting itself uses exactly ONE image.
    estimated_wave_scale = compute_wave_scale(
        train_files
    )

    wave_scale = (
        float(forced_wave_scale)
        if forced_wave_scale > 0
        else estimated_wave_scale
    )

    first_path = train_files[0]

    with np.load(
        first_path,
        allow_pickle=True,
    ) as z:

        required = [
            "target_480",
            "wavefields",
            "src_indices",
            "rec_indices",
        ]

        missing = [
            key
            for key in required
            if key not in z
        ]

        if missing:
            raise KeyError(
                f"{first_path} missing keys: {missing}; "
                f"available={list(z.keys())}"
            )

        speed = z[
            "target_480"
        ].astype(
            np.float32
        )

        wave = z[
            "wavefields"
        ].astype(
            np.complex64
        )

        src_indices = z[
            "src_indices"
        ].astype(
            np.int64
        )

        rec_indices = z[
            "rec_indices"
        ].astype(
            np.int64
        )

        dobs = (
            z["dobs_complex"].astype(
                np.complex64
            )
            if "dobs_complex" in z
            else None
        )

    if wave.ndim != 3:
        raise RuntimeError(
            f"Expected wavefields [Nsrc,H,W], got {wave.shape}"
        )

    nsrc, h, w = wave.shape

    if nsrc != 8:
        raise RuntimeError(
            f"C5.1h-B is locked to 8 sources, "
            f"but first train sample contains {nsrc}."
        )

    if speed.shape != (
        h,
        w,
    ):
        raise RuntimeError(
            f"speed shape={speed.shape}, wavefield spatial={(h,w)}"
        )

    if src_indices.shape[0] != nsrc:
        raise RuntimeError(
            f"src_indices={src_indices.shape}, nsrc={nsrc}"
        )

    background = load_background_wavefields(
        background_path,
        src_indices,
        h,
        w,
    )

    # ------------------------------------------------------------
    # Speed normalization.
    #
    # Diff-ANO paper reports mu=1488.39, sigma=27.53.
    # Keep explicitly configurable from CLI.
    # ------------------------------------------------------------
    speed_norm = (
        speed - float(speed_mean)
    ) / float(speed_std)

    # Same speed map repeated for every source.
    speed_batch = np.repeat(
        speed_norm[
            None,
            None,
            :,
            :
        ],
        nsrc,
        axis=0,
    )

    bg_2ch = np.stack(
        [
            background.real,
            background.imag,
        ],
        axis=1,
    ).astype(
        np.float32
    )

    bg_2ch /= wave_scale

    target_2ch = np.stack(
        [
            wave.real,
            wave.imag,
        ],
        axis=1,
    ).astype(
        np.float32
    )

    target_2ch /= wave_scale

    x = np.concatenate(
        [
            speed_batch.astype(
                np.float32
            ),
            bg_2ch,
        ],
        axis=1,
    )

    y = target_2ch

    print(
        f"[Dataset] split=train images=1 "
        f"pairs={nsrc} "
        f"wave_scale={wave_scale:.8e}"
    )

    print(
        "[Dataset] first file =",
        first_path,
    )

    print(
        "[Dataset] speed shape =",
        speed.shape,
    )

    print(
        "[Dataset] input shape =",
        x.shape,
    )

    print(
        "[Dataset] target shape =",
        y.shape,
    )

    print(
        "[Dataset] wavefields shape =",
        wave.shape,
    )

    print(
        "[Dataset] src_indices shape =",
        src_indices.shape,
    )

    print(
        "[Dataset] rec_indices shape =",
        rec_indices.shape,
    )

    print(
        "[Dataset] estimated wave_scale =",
        f"{estimated_wave_scale:.8e}",
    )

    print(
        "[Dataset] used wave_scale      =",
        f"{wave_scale:.8e}",
    )

    if dobs is not None:
        print(
            "[Dataset] dobs shape =",
            dobs.shape,
        )

    return {
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
        "speed": speed,
        "wave": wave,
        "background": background,
        "src_indices": src_indices,
        "rec_indices": rec_indices,
        "dobs": dobs,
        "wave_scale": wave_scale,
        "estimated_wave_scale": estimated_wave_scale,
        "file": str(first_path),
    }


# ======================================================================================
# Evaluation
# ======================================================================================

@torch.no_grad()
def evaluate_shared(
    model,
    x,
    target_wave,
    rec_indices,
    wave_scale,
    device,
    source_batch_size,
):
    model.eval()

    preds = []

    for start in range(
        0,
        x.shape[0],
        source_batch_size,
    ):
        end = min(
            start + source_batch_size,
            x.shape[0],
        )

        xb = x[
            start:end
        ].to(
            device
        )

        pred = model(
            xb
        )

        pred_np = (
            pred.detach()
            .cpu()
            .numpy()
        )

        pred_complex = (
            pred_np[:, 0]
            + 1j * pred_np[:, 1]
        ) * wave_scale

        preds.append(
            pred_complex.astype(
                np.complex64
            )
        )

    pred_wave = np.concatenate(
        preds,
        axis=0,
    )

    full = []
    rec = []

    rr = rec_indices[:, 0]
    cc = rec_indices[:, 1]

    for s in range(
        pred_wave.shape[0]
    ):
        full_rr = complex_rrmse_np(
            pred_wave[s],
            target_wave[s],
        )

        pred_rec = pred_wave[
            s,
            rr,
            cc,
        ]

        true_rec = target_wave[
            s,
            rr,
            cc,
        ]

        rec_rr = complex_rrmse_np(
            pred_rec,
            true_rec,
        )

        full.append(
            full_rr
        )

        rec.append(
            rec_rr
        )

    return {
        "pred_wave": pred_wave,
        "full": np.asarray(
            full,
            dtype=np.float64,
        ),
        "rec": np.asarray(
            rec,
            dtype=np.float64,
        ),
    }


# ======================================================================================
# Main
# ======================================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data_root",
        required=True,
    )

    parser.add_argument(
        "--background_path",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    parser.add_argument(
        "--modes",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260904,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    # Gradient accumulation over source mini-batches.
    # Optimizer update is still based on all eight sources.
    parser.add_argument(
        "--source_batch_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--speed_mean",
        type=float,
        default=1488.39,
    )

    parser.add_argument(
        "--speed_std",
        type=float,
        default=27.53,
    )

    # Set >0 to lock exactly to C5.1g scale.
    parser.add_argument(
        "--wave_scale",
        type=float,
        default=3.72290883e-02,
    )

    args = parser.parse_args()

    out = Path(
        args.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(
        args.seed
    )

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False"
        )

    device = torch.device(
        args.device
    )

    data = prepare_one_image_all_sources(
        data_root=args.data_root,
        background_path=args.background_path,
        speed_mean=args.speed_mean,
        speed_std=args.speed_std,
        forced_wave_scale=args.wave_scale,
    )

    x = data[
        "x"
    ].float()

    y = data[
        "y"
    ].float()

    target_wave = data[
        "wave"
    ]

    rec_indices = data[
        "rec_indices"
    ]

    wave_scale = data[
        "wave_scale"
    ]

    nsrc = x.shape[0]

    model = FNOBackgroundWavefield(
        modes=args.modes,
        width=args.width,
        depth=args.depth,
    ).to(
        device
    )

    nparams = count_parameters(
        model
    )

    print("=" * 100)
    print("C5.1h-B SHARED 8-SOURCE FNO-BG CAPACITY AUDIT")
    print("=" * 100)

    print(
        "num_sources       =",
        nsrc,
    )

    print(
        "modes             =",
        args.modes,
    )

    print(
        "width             =",
        args.width,
    )

    print(
        "depth             =",
        args.depth,
    )

    print(
        "parameters        =",
        f"{nparams:,}",
    )

    print(
        "wave_scale        =",
        f"{wave_scale:.8e}",
    )

    print(
        "speed_mean        =",
        args.speed_mean,
    )

    print(
        "speed_std         =",
        args.speed_std,
    )

    print(
        "steps             =",
        args.steps,
    )

    print(
        "lr                =",
        args.lr,
    )

    print(
        "min_lr            =",
        args.min_lr,
    )

    print(
        "source_batch_size =",
        args.source_batch_size,
    )

    # Hard architecture guard.
    if (
        args.modes == 25
        and args.width == 32
        and args.depth == 4
        and nparams != 5_125_730
    ):
        raise RuntimeError(
            "Architecture guard failed: "
            f"expected 5,125,730 params, got {nparams:,}"
        )

    initial = evaluate_shared(
        model=model,
        x=x,
        target_wave=target_wave,
        rec_indices=rec_indices,
        wave_scale=wave_scale,
        device=device,
        source_batch_size=args.source_batch_size,
    )

    print()
    print(
        "INITIAL "
        f"full_mean={initial['full'].mean():.6f} "
        f"full_max={initial['full'].max():.6f} "
        f"rec_mean={initial['rec'].mean():.6f} "
        f"rec_max={initial['rec'].max():.6f}"
    )

    print()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=args.min_lr,
    )

    history = []

    best_mean_full = float(
        "inf"
    )

    best_step = -1
    best_metrics = None

    source_indices = list(
        range(nsrc)
    )

    for step in range(
        1,
        args.steps + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        # Shuffle source order while preserving exact all-source mean loss.
        random.shuffle(
            source_indices
        )

        accumulated_loss = 0.0

        for start in range(
            0,
            nsrc,
            args.source_batch_size,
        ):
            ids = source_indices[
                start:
                start + args.source_batch_size
            ]

            xb = x[
                ids
            ].to(
                device
            )

            yb = y[
                ids
            ].to(
                device
            )

            pred = model(
                xb
            )

            chunk_loss = F.mse_loss(
                pred,
                yb,
                reduction="mean",
            )

            # Weight chunk according to number of sources so that
            # total gradient = mean loss over all 8 source pairs.
            weight = len(ids) / float(
                nsrc
            )

            loss = (
                chunk_loss
                * weight
            )

            loss.backward()

            accumulated_loss += (
                float(
                    chunk_loss.detach().cpu()
                )
                * weight
            )

        optimizer.step()
        scheduler.step()

        do_log = (
            step == 1
            or step % args.log_every == 0
            or step == args.steps
        )

        if do_log:
            metrics = evaluate_shared(
                model=model,
                x=x,
                target_wave=target_wave,
                rec_indices=rec_indices,
                wave_scale=wave_scale,
                device=device,
                source_batch_size=args.source_batch_size,
            )

            full = metrics[
                "full"
            ]

            rec = metrics[
                "rec"
            ]

            current_lr = scheduler.get_last_lr()[
                0
            ]

            row = {
                "step": step,
                "loss": accumulated_loss,
                "lr": current_lr,
                "full_mean": float(
                    full.mean()
                ),
                "full_std": float(
                    full.std()
                ),
                "full_max": float(
                    full.max()
                ),
                "rec_mean": float(
                    rec.mean()
                ),
                "rec_std": float(
                    rec.std()
                ),
                "rec_max": float(
                    rec.max()
                ),
            }

            history.append(
                row
            )

            print(
                f"[{step:04d}/{args.steps}] "
                f"loss={accumulated_loss:.6e} "
                f"lr={current_lr:.3e} | "
                f"full_mean={row['full_mean']:.6f} "
                f"full_max={row['full_max']:.6f} | "
                f"rec_mean={row['rec_mean']:.6f} "
                f"rec_max={row['rec_max']:.6f}"
            )

            if (
                row["full_mean"]
                < best_mean_full
            ):
                best_mean_full = row[
                    "full_mean"
                ]

                best_step = step

                best_metrics = {
                    "full": full.copy(),
                    "rec": rec.copy(),
                }

                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "step": step,
                        "args": vars(args),
                        "wave_scale": wave_scale,
                        "src_indices": data[
                            "src_indices"
                        ],
                        "rec_indices": data[
                            "rec_indices"
                        ],
                        "full": full,
                        "rec": rec,
                    },
                    out / "best.pt",
                )

    # ------------------------------------------------------------------
    # Reload best model before final source-level audit.
    # ------------------------------------------------------------------
    best_ckpt = torch.load(
        out / "best.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        best_ckpt[
            "model_state"
        ]
    )

    final = evaluate_shared(
        model=model,
        x=x,
        target_wave=target_wave,
        rec_indices=rec_indices,
        wave_scale=wave_scale,
        device=device,
        source_batch_size=args.source_batch_size,
    )

    full = final[
        "full"
    ]

    rec = final[
        "rec"
    ]

    np.savez_compressed(
        out / "best_predictions.npz",
        pred_wavefields=final[
            "pred_wave"
        ],
        target_wavefields=target_wave,
        background_wavefields=data[
            "background"
        ],
        src_indices=data[
            "src_indices"
        ],
        rec_indices=data[
            "rec_indices"
        ],
        speed_480=data[
            "speed"
        ],
        wave_scale=np.asarray(
            [wave_scale],
            dtype=np.float64,
        ),
    )

    # ------------------------------------------------------------------
    # Save history
    # ------------------------------------------------------------------
    history_csv = out / "history.csv"

    with history_csv.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss",
                "lr",
                "full_mean",
                "full_std",
                "full_max",
                "rec_mean",
                "rec_std",
                "rec_max",
            ],
        )

        writer.writeheader()
        writer.writerows(
            history
        )

    with (
        out
        / "history.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Per-source summary
    # ------------------------------------------------------------------
    rows = []

    for s in range(
        nsrc
    ):
        rows.append(
            {
                "source": s,
                "src_x": int(
                    data[
                        "src_indices"
                    ][s, 0]
                ),
                "src_y": int(
                    data[
                        "src_indices"
                    ][s, 1]
                ),
                "initial_full": float(
                    initial[
                        "full"
                    ][s]
                ),
                "initial_rec": float(
                    initial[
                        "rec"
                    ][s]
                ),
                "best_full": float(
                    full[s]
                ),
                "best_rec": float(
                    rec[s]
                ),
                "best_step": int(
                    best_step
                ),
            }
        )

    summary_csv = (
        out
        / "c5_1h_shared8_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print("=" * 100)
    print("C5.1h-B SHARED 8-SOURCE CAPACITY SUMMARY")
    print("=" * 100)

    for s in range(
        nsrc
    ):
        print(
            f"source {s:02d} | "
            f"full={full[s]:.6f} | "
            f"rec={rec[s]:.6f}"
        )

    print()
    print(
        "best step =",
        best_step,
    )

    print()
    print(
        "full mean =",
        f"{full.mean():.6f}",
    )

    print(
        "full std  =",
        f"{full.std():.6f}",
    )

    print(
        "full max  =",
        f"{full.max():.6f}",
    )

    print()
    print(
        "rec mean  =",
        f"{rec.mean():.6f}",
    )

    print(
        "rec std   =",
        f"{rec.std():.6f}",
    )

    print(
        "rec max   =",
        f"{rec.max():.6f}",
    )

    print()

    # ------------------------------------------------------------------
    # Frozen capacity decision
    # ------------------------------------------------------------------
    if (
        full.mean() < 0.03
        and full.max() < 0.05
        and rec.mean() < 0.01
    ):
        print(
            "[STRONG PASS] shared 8-source "
            "capacity confirmed."
        )

    elif (
        full.mean() < 0.06
        and full.max() < 0.10
    ):
        print(
            "[PASS] shared source-conditioned "
            "mapping is learnable."
        )

    else:
        print(
            "[WARNING] shared-source capacity "
            "is not yet sufficient."
        )

    print()
    print(
        "summary_csv =",
        summary_csv,
    )

    print(
        "best_ckpt   =",
        out / "best.pt",
    )


if __name__ == "__main__":
    main()
