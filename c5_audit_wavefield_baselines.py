import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from cbs_model import ConvergentBornSeries_Batch


def numeric_key(p):
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else -1


def rrmse_from_sums(num, den):
    return float(np.sqrt(num / max(den, 1e-30)))


def accumulate_rrmse(pred, target):
    diff = pred - target

    num = float(
        np.sum(
            np.abs(diff).astype(np.float64) ** 2
        )
    )

    den = float(
        np.sum(
            np.abs(target).astype(np.float64) ** 2
        )
    )

    return num, den


def receiver_extract(field, rec):
    """
    field: [S,H,W] complex
    rec:   [R,2]
    return [S,R]
    """

    return field[
        :,
        rec[:, 0],
        rec[:, 1],
    ]


@torch.no_grad()
def make_background_wavefield(
    src_indices,
    frequency,
    cbs_iters,
    boundary_width,
    boundary_strength,
    boundary_type,
    background_speed,
    device,
):
    sos = torch.full(
        (1, 1, 480, 480),
        float(background_speed),
        dtype=torch.float32,
        device=device,
    )

    model = ConvergentBornSeries_Batch(
        f=float(frequency),
        sos=sos,
        boundary_width=[
            int(boundary_width),
            int(boundary_width),
        ],
        boundary_strength=float(
            boundary_strength
        ),
        boundary_type=str(
            boundary_type
        ),
        src_loc_set=src_indices,
        device=str(device),
    )

    u = model(
        max_iters=int(cbs_iters)
    )

    return (
        u[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.complex64)
    )


def load_files(root, split):
    return sorted(
        (root / split).glob("*.npz"),
        key=numeric_key,
    )


def evaluate_baseline(
    files,
    predictor,
):
    full_num = 0.0
    full_den = 0.0

    rec_num = 0.0
    rec_den = 0.0

    sample_full = []
    sample_rec = []

    for p in files:

        with np.load(
            p,
            allow_pickle=True,
        ) as z:

            target = (
                z["wavefields"]
                .astype(np.complex64)
            )

            dobs = (
                z["dobs_complex"]
                .astype(np.complex64)
            )

            rec = (
                z["rec_indices"]
                .astype(np.int64)
            )

        pred = predictor(
            p,
            target,
        )

        pred_rec = receiver_extract(
            pred,
            rec,
        )

        n, d = accumulate_rrmse(
            pred,
            target,
        )

        rn, rd = accumulate_rrmse(
            pred_rec,
            dobs,
        )

        full_num += n
        full_den += d

        rec_num += rn
        rec_den += rd

        sample_full.append(
            rrmse_from_sums(n, d)
        )

        sample_rec.append(
            rrmse_from_sums(rn, rd)
        )

    return {
        "full_rrmse":
            rrmse_from_sums(
                full_num,
                full_den,
            ),

        "receiver_rrmse":
            rrmse_from_sums(
                rec_num,
                rec_den,
            ),

        "sample_full_mean":
            float(
                np.mean(sample_full)
            ),

        "sample_full_std":
            float(
                np.std(sample_full)
            ),

        "sample_rec_mean":
            float(
                np.mean(sample_rec)
            ),

        "sample_rec_std":
            float(
                np.std(sample_rec)
            ),
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        required=True,
    )

    ap.add_argument(
        "--background_speed",
        type=float,
        default=1500.0,
    )

    ap.add_argument(
        "--device",
        default="cuda:0",
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    args = ap.parse_args()

    root = Path(args.root)

    train_files = load_files(
        root,
        "train",
    )

    val_files = load_files(
        root,
        "val",
    )

    if not train_files:
        raise RuntimeError(
            "No training wavefields."
        )

    print("=" * 100)
    print(
        "C5 WAVEFIELD BASELINE AUDIT"
    )
    print("=" * 100)

    print(
        "train images =",
        len(train_files),
    )

    print(
        "val images   =",
        len(val_files),
    )

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    with np.load(
        train_files[0],
        allow_pickle=True,
    ) as z:

        src_indices = (
            z["src_indices"]
            .astype(np.int64)
        )

        rec_indices = (
            z["rec_indices"]
            .astype(np.int64)
        )

        frequency = float(
            z["frequency"][0]
        )

        cbs_iters = int(
            z["cbs_iters"][0]
        )

        boundary_width = int(
            z["boundary_width"][0]
        )

        boundary_strength = float(
            z["boundary_strength"][0]
        )

        boundary_type = str(
            z["boundary_type"][0]
        )

    print()
    print(
        "sources =",
        src_indices.shape[0],
    )

    print(
        "receivers =",
        rec_indices.shape[0],
    )

    # --------------------------------------------------
    # Mean train wavefield
    # --------------------------------------------------

    print()
    print(
        "[1] computing training "
        "mean wavefield ..."
    )

    mean_field = None

    for p in train_files:

        with np.load(
            p,
            allow_pickle=True,
        ) as z:

            w = (
                z["wavefields"]
                .astype(np.complex64)
            )

        if mean_field is None:

            mean_field = (
                w.astype(np.complex128)
            )

        else:

            mean_field += w

    mean_field /= len(
        train_files
    )

    mean_field = (
        mean_field
        .astype(np.complex64)
    )

    # --------------------------------------------------
    # Homogeneous CBS background
    # --------------------------------------------------

    print()
    print(
        "[2] generating homogeneous "
        f"{args.background_speed:.1f} m/s "
        "background wavefields ..."
    )

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    bg_field = (
        make_background_wavefield(
            src_indices=
                src_indices,

            frequency=
                frequency,

            cbs_iters=
                cbs_iters,

            boundary_width=
                boundary_width,

            boundary_strength=
                boundary_strength,

            boundary_type=
                boundary_type,

            background_speed=
                args.background_speed,

            device=
                device,
        )
    )

    # --------------------------------------------------
    # Baselines
    # --------------------------------------------------

    zero_field = np.zeros_like(
        mean_field
    )

    predictors = {
        "zero":
            lambda p, t:
                zero_field,

        "homogeneous_1500":
            lambda p, t:
                bg_field,

        "train_mean":
            lambda p, t:
                mean_field,
    }

    results = {}

    print()
    print("=" * 100)
    print(
        "BASELINE RESULTS"
    )
    print("=" * 100)

    for name, pred_fn in (
        predictors.items()
    ):

        print()
        print(
            "---",
            name,
            "---",
        )

        train_metrics = (
            evaluate_baseline(
                train_files,
                pred_fn,
            )
        )

        val_metrics = (
            evaluate_baseline(
                val_files,
                pred_fn,
            )
        )

        results[name] = {
            "train":
                train_metrics,

            "val":
                val_metrics,
        }

        print(
            "TRAIN "
            f"full={train_metrics['full_rrmse']:.6f} "
            f"receiver={train_metrics['receiver_rrmse']:.6f}"
        )

        print(
            "VAL   "
            f"full={val_metrics['full_rrmse']:.6f} "
            f"receiver={val_metrics['receiver_rrmse']:.6f}"
        )

    # --------------------------------------------------
    # Scattered field ratio
    # --------------------------------------------------

    scat_full_num = 0.0
    total_full_den = 0.0

    for p in train_files:

        with np.load(
            p,
            allow_pickle=True,
        ) as z:

            target = (
                z["wavefields"]
                .astype(np.complex64)
            )

        diff = (
            target
            -
            bg_field
        )

        scat_full_num += float(
            np.sum(
                np.abs(diff)
                .astype(np.float64) ** 2
            )
        )

        total_full_den += float(
            np.sum(
                np.abs(target)
                .astype(np.float64) ** 2
            )
        )

    scattered_ratio = (
        rrmse_from_sums(
            scat_full_num,
            total_full_den,
        )
    )

    print()
    print("=" * 100)
    print(
        "SCATTERED-FIELD ENERGY"
    )
    print("=" * 100)

    print(
        "||u-u_bg|| / ||u|| =",
        f"{scattered_ratio:.6f}",
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    payload = {
        "background_speed":
            args.background_speed,

        "train_images":
            len(train_files),

        "val_images":
            len(val_files),

        "results":
            results,

        "scattered_relative_norm_train":
            scattered_ratio,
    }

    out = Path(
        args.out
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        out,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
        )

    np.savez_compressed(
        out.with_suffix(".npz"),

        mean_field=
            mean_field,

        background_field=
            bg_field,

        src_indices=
            src_indices,

        rec_indices=
            rec_indices,
    )

    print()
    print(
        "[PASS] baseline audit complete"
    )

    print(
        "summary =",
        out,
    )


if __name__ == "__main__":
    main()
