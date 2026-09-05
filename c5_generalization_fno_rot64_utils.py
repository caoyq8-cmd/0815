import os
import csv
import json
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# IMPORTANT:
# Reuse exactly the FNO implementation that passed C5.1h-B.
from c5_shared8_capacity_fno import (
    FNOBackgroundWavefield,
    set_seed,
    count_parameters,
    numeric_key,
    compute_wave_scale,
    load_background_wavefields,
    complex_rrmse_np,
)


# ======================================================================================
# Data
# ======================================================================================

def find_split_files(data_root, split):
    split_dir = Path(data_root) / split

    files = sorted(
        split_dir.glob(f"{split}_*.npz"),
        key=numeric_key,
    )

    # fallback
    if not files:
        files = sorted(
            split_dir.glob("*.npz"),
            key=numeric_key,
        )

    if not files:
        raise FileNotFoundError(
            f"No npz files found in {split_dir}"
        )

    return files


class WavefieldSplit:
    """
    Preload one split into CPU memory.

    Stored:
      speed_norm : [N,1,H,W]
      wave_norm  : [N,S,2,H,W]
      wave_raw   : [N,S,H,W] complex64

    Background is shared across images:
      bg_norm    : [S,2,H,W]
    """

    def __init__(
        self,
        files,
        background_path,
        wave_scale,
        speed_mean,
        speed_std,
        reference_src_indices=None,
        reference_rec_indices=None,
    ):
        self.files = files
        self.wave_scale = float(wave_scale)
        self.speed_mean = float(speed_mean)
        self.speed_std = float(speed_std)

        speed_list = []
        wave_norm_list = []
        wave_raw_list = []

        src_ref = None
        rec_ref = None

        for i, path in enumerate(files):
            with np.load(path, allow_pickle=True) as z:
                required = [
                    "target_480",
                    "wavefields",
                    "src_indices",
                    "rec_indices",
                ]

                missing = [
                    k for k in required
                    if k not in z
                ]

                if missing:
                    raise KeyError(
                        f"{path} missing {missing}; "
                        f"available={list(z.keys())}"
                    )

                speed = z["target_480"].astype(
                    np.float32
                )

                wave = z["wavefields"].astype(
                    np.complex64
                )

                src = z["src_indices"].astype(
                    np.int64
                )

                rec = z["rec_indices"].astype(
                    np.int64
                )

            if wave.ndim != 3:
                raise RuntimeError(
                    f"{path}: wavefields must be [S,H,W], "
                    f"got {wave.shape}"
                )

            nsrc, h, w = wave.shape

            if nsrc != 64:
                raise RuntimeError(
                    f"{path}: expected 64 sources, got {nsrc}"
                )

            if speed.shape != (h, w):
                raise RuntimeError(
                    f"{path}: speed={speed.shape}, "
                    f"wave spatial={(h,w)}"
                )

            if i == 0:
                src_ref = src.copy()
                rec_ref = rec.copy()

            else:
                if not np.array_equal(
                    src,
                    src_ref,
                ):
                    raise RuntimeError(
                        f"Source geometry differs in {path}"
                    )

                if not np.array_equal(
                    rec,
                    rec_ref,
                ):
                    raise RuntimeError(
                        f"Receiver geometry differs in {path}"
                    )

            speed_norm = (
                speed - self.speed_mean
            ) / self.speed_std

            target_2ch = np.stack(
                [
                    wave.real,
                    wave.imag,
                ],
                axis=1,
            ).astype(np.float32)

            target_2ch /= self.wave_scale

            speed_list.append(
                speed_norm[None]
            )

            wave_norm_list.append(
                target_2ch
            )

            wave_raw_list.append(
                wave
            )

        if reference_src_indices is not None:
            if not np.array_equal(
                src_ref,
                reference_src_indices,
            ):
                raise RuntimeError(
                    "Train/val source geometry mismatch."
                )

        if reference_rec_indices is not None:
            if not np.array_equal(
                rec_ref,
                reference_rec_indices,
            ):
                raise RuntimeError(
                    "Train/val receiver geometry mismatch."
                )

        self.speed_norm = np.stack(
            speed_list,
            axis=0,
        ).astype(np.float32)

        self.wave_norm = np.stack(
            wave_norm_list,
            axis=0,
        ).astype(np.float32)

        self.wave_raw = np.stack(
            wave_raw_list,
            axis=0,
        ).astype(np.complex64)

        self.src_indices = src_ref
        self.rec_indices = rec_ref

        self.num_images = self.speed_norm.shape[0]
        self.num_sources = self.wave_norm.shape[1]
        self.h = self.speed_norm.shape[-2]
        self.w = self.speed_norm.shape[-1]

        background = load_background_wavefields(
            background_path,
            self.src_indices,
            self.h,
            self.w,
        )

        bg_norm = np.stack(
            [
                background.real,
                background.imag,
            ],
            axis=1,
        ).astype(np.float32)

        bg_norm /= self.wave_scale

        self.background_raw = background
        self.background_norm = bg_norm

        print(
            f"[Split] images={self.num_images} "
            f"sources={self.num_sources} "
            f"pairs={self.num_images*self.num_sources}"
        )

        print(
            "[Split] speed_norm =",
            self.speed_norm.shape,
        )

        print(
            "[Split] wave_norm  =",
            self.wave_norm.shape,
        )

    @property
    def num_pairs(self):
        return self.num_images * self.num_sources

    def decode_pair_index(self, pair_idx):
        image_idx = pair_idx // self.num_sources
        source_idx = pair_idx % self.num_sources
        return image_idx, source_idx

    def make_pair(self, pair_idx):
        image_idx, source_idx = self.decode_pair_index(
            pair_idx
        )

        speed = self.speed_norm[
            image_idx
        ]

        bg = self.background_norm[
            source_idx
        ]

        x = np.concatenate(
            [
                speed,
                bg,
            ],
            axis=0,
        ).astype(np.float32)

        y = self.wave_norm[
            image_idx,
            source_idx,
        ]

        return x, y


def make_micro_batch(dataset, pair_ids):
    xs = []
    ys = []

    for idx in pair_ids:
        x, y = dataset.make_pair(
            int(idx)
        )
        xs.append(x)
        ys.append(y)

    x = torch.from_numpy(
        np.stack(xs, axis=0)
    ).float()

    y = torch.from_numpy(
        np.stack(ys, axis=0)
    ).float()

    return x, y


# ======================================================================================
# Evaluation
# ======================================================================================

@torch.no_grad()
def evaluate_split(
    model,
    dataset,
    device,
    micro_batch_size=1,
    save_predictions=False,
):
    model.eval()

    full_matrix = np.zeros(
        (
            dataset.num_images,
            dataset.num_sources,
        ),
        dtype=np.float64,
    )

    rec_matrix = np.zeros_like(
        full_matrix
    )

    predictions = (
        np.zeros_like(
            dataset.wave_raw
        )
        if save_predictions
        else None
    )

    rr = dataset.rec_indices[:, 0]
    cc = dataset.rec_indices[:, 1]

    pair_ids = np.arange(
        dataset.num_pairs
    )

    for start in range(
        0,
        dataset.num_pairs,
        micro_batch_size,
    ):
        ids = pair_ids[
            start:
            start + micro_batch_size
        ]

        xb, _ = make_micro_batch(
            dataset,
            ids,
        )

        xb = xb.to(
            device,
            non_blocking=True,
        )

        pred = model(
            xb
        )

        pred = (
            pred.detach()
            .cpu()
            .numpy()
        )

        pred_complex = (
            pred[:, 0]
            + 1j * pred[:, 1]
        ) * dataset.wave_scale

        for j, pair_idx in enumerate(ids):
            image_idx, source_idx = (
                dataset.decode_pair_index(
                    int(pair_idx)
                )
            )

            true_wave = dataset.wave_raw[
                image_idx,
                source_idx,
            ]

            pred_wave = pred_complex[j]

            full_rr = complex_rrmse_np(
                pred_wave,
                true_wave,
            )

            rec_rr = complex_rrmse_np(
                pred_wave[rr, cc],
                true_wave[rr, cc],
            )

            full_matrix[
                image_idx,
                source_idx,
            ] = full_rr

            rec_matrix[
                image_idx,
                source_idx,
            ] = rec_rr

            if predictions is not None:
                predictions[
                    image_idx,
                    source_idx,
                ] = pred_wave.astype(
                    np.complex64
                )

    return {
        "full": full_matrix,
        "rec": rec_matrix,
        "predictions": predictions,
    }


def metric_summary(matrix):
    return {
        "mean": float(
            matrix.mean()
        ),
        "std": float(
            matrix.std()
        ),
        "median": float(
            np.median(matrix)
        ),
        "min": float(
            matrix.min()
        ),
        "max": float(
            matrix.max()
        ),
    }


def print_metric_summary(name, matrix):
    s = metric_summary(
        matrix
    )

    print(
        f"{name:18s} "
        f"mean={s['mean']:.6f} "
        f"std={s['std']:.6f} "
        f"median={s['median']:.6f} "
        f"min={s['min']:.6f} "
        f"max={s['max']:.6f}"
    )


# ======================================================================================
# CSV audits
# ======================================================================================

def save_pair_csv(
    path,
    dataset,
    full,
    rec,
):
    with open(
        path,
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "image_idx",
                "file",
                "source_idx",
                "src_x",
                "src_y",
                "full_rrmse",
                "rec_rrmse",
            ]
        )

        for i in range(
            dataset.num_images
        ):
            for s in range(
                dataset.num_sources
            ):
                writer.writerow(
                    [
                        i,
                        str(
                            dataset.files[i]
                        ),
                        s,
                        int(
                            dataset.src_indices[
                                s, 0
                            ]
                        ),
                        int(
                            dataset.src_indices[
                                s, 1
                            ]
                        ),
                        float(
                            full[i, s]
                        ),
                        float(
                            rec[i, s]
                        ),
                    ]
                )


def save_source_csv(
    path,
    full,
    rec,
):
    with open(
        path,
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "source_idx",
                "full_mean",
                "full_std",
                "full_max",
                "rec_mean",
                "rec_std",
                "rec_max",
            ]
        )

        for s in range(
            full.shape[1]
        ):
            writer.writerow(
                [
                    s,
                    float(
                        full[:, s].mean()
                    ),
                    float(
                        full[:, s].std()
                    ),
                    float(
                        full[:, s].max()
                    ),
                    float(
                        rec[:, s].mean()
                    ),
                    float(
                        rec[:, s].std()
                    ),
                    float(
                        rec[:, s].max()
                    ),
                ]
            )


def save_image_csv(
    path,
    dataset,
    full,
    rec,
):
    with open(
        path,
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "image_idx",
                "file",
                "full_mean",
                "full_std",
                "full_max",
                "rec_mean",
                "rec_std",
                "rec_max",
            ]
        )

        for i in range(
            full.shape[0]
        ):
            writer.writerow(
                [
                    i,
                    str(
                        dataset.files[i]
                    ),
                    float(
                        full[i].mean()
                    ),
                    float(
                        full[i].std()
                    ),
                    float(
                        full[i].max()
                    ),
                    float(
                        rec[i].mean()
                    ),
                    float(
                        rec[i].std()
                    ),
                    float(
                        rec[i].max()
                    ),
                ]
            )


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
        "--epochs",
        type=int,
        default=50,
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
        "--micro_batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--accum_pairs",
        type=int,
        default=4,
        help=(
            "Number of image-source pairs accumulated "
            "before one optimizer update."
        ),
    )

    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
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
            "CUDA requested but unavailable."
        )

    device = torch.device(
        args.device
    )

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    train_files = find_split_files(
        args.data_root,
        "train",
    )

    val_files = find_split_files(
        args.data_root,
        "val",
    )

    print("=" * 100)
    print("C5.1i FNO-BG OPERATOR GENERALIZATION")
    print("=" * 100)

    print(
        "train images =",
        len(train_files),
    )

    print(
        "val images   =",
        len(val_files),
    )

    if len(train_files) != 20:
        print(
            f"[WARNING] expected 20 train images, "
            f"found {len(train_files)}"
        )

    if len(val_files) != 5:
        print(
            f"[WARNING] expected 5 val images, "
            f"found {len(val_files)}"
        )

    estimated_scale = compute_wave_scale(
        train_files
    )

    print(
        "estimated train wave_scale =",
        f"{estimated_scale:.8e}",
    )

    print(
        "locked wave_scale          =",
        f"{args.wave_scale:.8e}",
    )

    rel_scale_diff = abs(
        estimated_scale
        - args.wave_scale
    ) / (
        abs(args.wave_scale)
        + 1e-12
    )

    print(
        "relative scale difference  =",
        f"{rel_scale_diff:.6e}",
    )

    if rel_scale_diff > 1e-3:
        print(
            "[WARNING] locked wave_scale differs "
            "noticeably from current train-set estimate."
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    train_set = WavefieldSplit(
        files=train_files,
        background_path=args.background_path,
        wave_scale=args.wave_scale,
        speed_mean=args.speed_mean,
        speed_std=args.speed_std,
    )

    val_set = WavefieldSplit(
        files=val_files,
        background_path=args.background_path,
        wave_scale=args.wave_scale,
        speed_mean=args.speed_mean,
        speed_std=args.speed_std,
        reference_src_indices=train_set.src_indices,
        reference_rec_indices=train_set.rec_indices,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

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

    print()
    print(
        "modes       =",
        args.modes,
    )

    print(
        "width       =",
        args.width,
    )

    print(
        "depth       =",
        args.depth,
    )

    print(
        "parameters  =",
        f"{nparams:,}",
    )

    print(
        "epochs      =",
        args.epochs,
    )

    print(
        "lr          =",
        args.lr,
    )

    print(
        "accum_pairs =",
        args.accum_pairs,
    )

    if (
        args.modes == 25
        and args.width == 32
        and args.depth == 4
        and nparams != 5_125_730
    ):
        raise RuntimeError(
            f"Architecture mismatch: "
            f"expected 5,125,730 params, got {nparams:,}"
        )

    # Number of optimizer updates per epoch.
    updates_per_epoch = math.ceil(
        train_set.num_pairs
        / args.accum_pairs
    )

    total_updates = (
        updates_per_epoch
        * args.epochs
    )

    print(
        "train pairs       =",
        train_set.num_pairs,
    )

    print(
        "val pairs         =",
        val_set.num_pairs,
    )

    print(
        "updates / epoch   =",
        updates_per_epoch,
    )

    print(
        "total updates     =",
        total_updates,
    )

    # With 20*8=160, accum=4, epochs=50:
    # 40 updates/epoch * 50 = exactly 2000.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_updates,
        eta_min=args.min_lr,
    )

    # ------------------------------------------------------------------
    # Initial val
    # ------------------------------------------------------------------

    initial_val = evaluate_split(
        model,
        val_set,
        device,
        micro_batch_size=args.micro_batch_size,
        save_predictions=False,
    )

    print()
    print("INITIAL VALIDATION")
    print_metric_summary(
        "val full",
        initial_val["full"],
    )

    print_metric_summary(
        "val receiver",
        initial_val["rec"],
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    best_val_full = float("inf")
    best_epoch = -1
    best_update = -1

    global_update = 0

    history = []

    rng = np.random.default_rng(
        args.seed
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        permutation = rng.permutation(
            train_set.num_pairs
        )

        epoch_loss_sum = 0.0
        epoch_pair_count = 0

        for group_start in range(
            0,
            train_set.num_pairs,
            args.accum_pairs,
        ):
            group_ids = permutation[
                group_start:
                group_start
                + args.accum_pairs
            ]

            optimizer.zero_grad(
                set_to_none=True
            )

            group_loss = 0.0

            # micro-batches inside one accumulated optimizer update
            for micro_start in range(
                0,
                len(group_ids),
                args.micro_batch_size,
            ):
                micro_ids = group_ids[
                    micro_start:
                    micro_start
                    + args.micro_batch_size
                ]

                xb, yb = make_micro_batch(
                    train_set,
                    micro_ids,
                )

                xb = xb.to(
                    device,
                    non_blocking=True,
                )

                yb = yb.to(
                    device,
                    non_blocking=True,
                )

                pred = model(
                    xb
                )

                micro_loss = F.mse_loss(
                    pred,
                    yb,
                    reduction="mean",
                )

                weight = (
                    len(micro_ids)
                    / float(
                        len(group_ids)
                    )
                )

                loss = (
                    micro_loss
                    * weight
                )

                loss.backward()

                group_loss += (
                    float(
                        micro_loss
                        .detach()
                        .cpu()
                    )
                    * weight
                )

            optimizer.step()
            scheduler.step()

            global_update += 1

            epoch_loss_sum += (
                group_loss
                * len(group_ids)
            )

            epoch_pair_count += (
                len(group_ids)
            )

        epoch_loss = (
            epoch_loss_sum
            / max(
                epoch_pair_count,
                1,
            )
        )

        do_eval = (
            epoch == 1
            or epoch % args.eval_every == 0
            or epoch == args.epochs
        )

        if do_eval:
            val_metrics = evaluate_split(
                model,
                val_set,
                device,
                micro_batch_size=args.micro_batch_size,
                save_predictions=False,
            )

            val_full = val_metrics[
                "full"
            ]

            val_rec = val_metrics[
                "rec"
            ]

            lr_now = scheduler.get_last_lr()[
                0
            ]

            row = {
                "epoch": epoch,
                "update": global_update,
                "train_loss": epoch_loss,
                "lr": lr_now,
                "val_full_mean": float(
                    val_full.mean()
                ),
                "val_full_std": float(
                    val_full.std()
                ),
                "val_full_max": float(
                    val_full.max()
                ),
                "val_rec_mean": float(
                    val_rec.mean()
                ),
                "val_rec_std": float(
                    val_rec.std()
                ),
                "val_rec_max": float(
                    val_rec.max()
                ),
            }

            history.append(
                row
            )

            print(
                f"[epoch {epoch:03d}/{args.epochs}] "
                f"update={global_update:04d}/{total_updates} "
                f"loss={epoch_loss:.6e} "
                f"lr={lr_now:.3e} | "
                f"val_full={row['val_full_mean']:.6f} "
                f"(max={row['val_full_max']:.6f}) | "
                f"val_rec={row['val_rec_mean']:.6f} "
                f"(max={row['val_rec_max']:.6f})"
            )

            if (
                row["val_full_mean"]
                < best_val_full
            ):
                best_val_full = row[
                    "val_full_mean"
                ]

                best_epoch = epoch
                best_update = global_update

                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch,
                        "update": global_update,
                        "args": vars(args),
                        "wave_scale": args.wave_scale,
                        "speed_mean": args.speed_mean,
                        "speed_std": args.speed_std,
                        "src_indices": train_set.src_indices,
                        "rec_indices": train_set.rec_indices,
                        "val_full_mean": row[
                            "val_full_mean"
                        ],
                        "val_rec_mean": row[
                            "val_rec_mean"
                        ],
                    },
                    out / "best.pt",
                )

    # ------------------------------------------------------------------
    # Save training history
    # ------------------------------------------------------------------

    with (
        out / "history.csv"
    ).open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                history[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            history
        )

    with (
        out / "history.json"
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
    # Reload BEST checkpoint
    # ------------------------------------------------------------------

    ckpt = torch.load(
        out / "best.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        ckpt["model_state"]
    )

    print()
    print("=" * 100)
    print("BEST CHECKPOINT FULL AUDIT")
    print("=" * 100)

    print(
        "best epoch  =",
        best_epoch,
    )

    print(
        "best update =",
        best_update,
    )

    # Full train audit
    train_best = evaluate_split(
        model,
        train_set,
        device,
        micro_batch_size=args.micro_batch_size,
        save_predictions=False,
    )

    # Full val audit + predictions
    val_best = evaluate_split(
        model,
        val_set,
        device,
        micro_batch_size=args.micro_batch_size,
        save_predictions=True,
    )

    print()
    print("TRAIN")
    print_metric_summary(
        "train full",
        train_best["full"],
    )

    print_metric_summary(
        "train receiver",
        train_best["rec"],
    )

    print()
    print("VALIDATION")
    print_metric_summary(
        "val full",
        val_best["full"],
    )

    print_metric_summary(
        "val receiver",
        val_best["rec"],
    )

    # ------------------------------------------------------------------
    # Per-source audit
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("VALIDATION PER-SOURCE AUDIT")
    print("=" * 100)

    for s in range(
        val_set.num_sources
    ):
        vf = val_best[
            "full"
        ][:, s]

        vr = val_best[
            "rec"
        ][:, s]

        print(
            f"source {s:02d} | "
            f"full={vf.mean():.6f}±{vf.std():.6f} "
            f"max={vf.max():.6f} | "
            f"rec={vr.mean():.6f}±{vr.std():.6f} "
            f"max={vr.max():.6f}"
        )

    # ------------------------------------------------------------------
    # Per-image audit
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("VALIDATION PER-IMAGE AUDIT")
    print("=" * 100)

    for i in range(
        val_set.num_images
    ):
        vf = val_best[
            "full"
        ][i]

        vr = val_best[
            "rec"
        ][i]

        print(
            f"val image {i:02d} | "
            f"full={vf.mean():.6f}±{vf.std():.6f} "
            f"max={vf.max():.6f} | "
            f"rec={vr.mean():.6f}±{vr.std():.6f} "
            f"max={vr.max():.6f}"
        )

    # ------------------------------------------------------------------
    # Save audits
    # ------------------------------------------------------------------

    save_pair_csv(
        out / "train_per_pair_best.csv",
        train_set,
        train_best["full"],
        train_best["rec"],
    )

    save_pair_csv(
        out / "val_per_pair_best.csv",
        val_set,
        val_best["full"],
        val_best["rec"],
    )

    save_source_csv(
        out / "val_per_source_best.csv",
        val_best["full"],
        val_best["rec"],
    )

    save_image_csv(
        out / "val_per_image_best.csv",
        val_set,
        val_best["full"],
        val_best["rec"],
    )

    np.savez_compressed(
        out / "val_best_predictions.npz",
        pred_wavefields=val_best[
            "predictions"
        ],
        target_wavefields=val_set.wave_raw,
        background_wavefields=val_set.background_raw,
        src_indices=val_set.src_indices,
        rec_indices=val_set.rec_indices,
        wave_scale=np.asarray(
            [args.wave_scale],
            dtype=np.float64,
        ),
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    summary = {
        "best_epoch": best_epoch,
        "best_update": best_update,
        "parameters": nparams,

        "train_full": metric_summary(
            train_best["full"]
        ),

        "train_rec": metric_summary(
            train_best["rec"]
        ),

        "val_full": metric_summary(
            val_best["full"]
        ),

        "val_rec": metric_summary(
            val_best["rec"]
        ),
    }

    with (
        out / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    vf = summary[
        "val_full"
    ]["mean"]

    vr = summary[
        "val_rec"
    ]["mean"]

    print()
    print("=" * 100)
    print("C5.1i FINAL SUMMARY")
    print("=" * 100)

    print(
        f"best epoch       = {best_epoch}"
    )

    print(
        f"best update      = {best_update}"
    )

    print(
        f"train full mean  = "
        f"{summary['train_full']['mean']:.6f}"
    )

    print(
        f"train rec mean   = "
        f"{summary['train_rec']['mean']:.6f}"
    )

    print(
        f"val full mean    = "
        f"{summary['val_full']['mean']:.6f}"
    )

    print(
        f"val full std     = "
        f"{summary['val_full']['std']:.6f}"
    )

    print(
        f"val full max     = "
        f"{summary['val_full']['max']:.6f}"
    )

    print(
        f"val rec mean     = "
        f"{summary['val_rec']['mean']:.6f}"
    )

    print(
        f"val rec std      = "
        f"{summary['val_rec']['std']:.6f}"
    )

    print(
        f"val rec max      = "
        f"{summary['val_rec']['max']:.6f}"
    )

    print()

    if (
        vf < 0.10
        and vr < 0.05
    ):
        print(
            "[STRONG PASS] FNO operator "
            "generalization confirmed."
        )

    elif (
        vf < 0.15
        and vr < 0.08
    ):
        print(
            "[PASS] FNO generalization is usable; "
            "MgNO comparison is now warranted."
        )

    elif vf < 0.25:
        print(
            "[WARNING] noticeable operator "
            "generalization gap."
        )

    else:
        print(
            "[FAIL] severe unseen-image "
            "generalization failure."
        )

    print()
    print(
        "best checkpoint =",
        out / "best.pt",
    )

    print(
        "summary         =",
        out / "summary.json",
    )

    print(
        "val pair audit  =",
        out / "val_per_pair_best.csv",
    )

    print(
        "val source audit=",
        out / "val_per_source_best.csv",
    )

    print(
        "val image audit =",
        out / "val_per_image_best.csv",
    )


if __name__ == "__main__":
    main()
