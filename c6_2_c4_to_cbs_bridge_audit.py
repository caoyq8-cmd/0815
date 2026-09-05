import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


CENTER = 1502.5
SCALE = 102.5

SPEED_MIN = 1400.0
SPEED_MAX = 1605.0


def as_bhw(x):
    x = np.asarray(x)

    if x.ndim == 4:
        if x.shape[1] != 1:
            raise RuntimeError(
                f"Expected [B,1,H,W], got {x.shape}"
            )
        x = x[:, 0]

    if x.ndim != 3:
        raise RuntimeError(
            f"Expected [B,H,W], got {x.shape}"
        )

    return x.astype(np.float32)


def norm_to_speed(x):
    return (
        np.asarray(x, dtype=np.float32)
        * SCALE
        + CENTER
    )


def resize_480(x):
    t = torch.from_numpy(
        np.asarray(x, dtype=np.float32)
    )[None, None]

    y = F.interpolate(
        t,
        size=(480, 480),
        mode="bilinear",
        align_corners=False,
    )

    return y[0, 0].cpu().numpy()


def mse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def mae(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def rrmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    return float(
        np.sqrt(np.mean((a - b) ** 2))
        /
        (
            np.sqrt(np.mean(b ** 2))
            + 1e-12
        )
    )


def maxabs(a, b):
    return float(
        np.max(
            np.abs(
                np.asarray(a, dtype=np.float64)
                -
                np.asarray(b, dtype=np.float64)
            )
        )
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--pred_norm",
        required=True,
    )

    ap.add_argument(
        "--gt_norm",
        required=True,
    )

    ap.add_argument(
        "--hint_norm",
        required=True,
    )

    ap.add_argument(
        "--c5_val_dir",
        required=True,
    )

    ap.add_argument(
        "--output_dir",
        required=True,
    )

    ap.add_argument(
        "--max_samples",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--max_gt256_abs_tol",
        type=float,
        default=1e-2,
    )

    ap.add_argument(
        "--max_gt256_rrmse_tol",
        type=float,
        default=1e-6,
    )

    args = ap.parse_args()

    pred_norm = as_bhw(
        np.load(args.pred_norm)
    )

    gt_norm = as_bhw(
        np.load(args.gt_norm)
    )

    hint_norm = as_bhw(
        np.load(args.hint_norm)
    )

    if not (
        pred_norm.shape
        == gt_norm.shape
        == hint_norm.shape
    ):
        raise RuntimeError(
            "C4 array shape mismatch:\n"
            f"pred={pred_norm.shape}\n"
            f"gt={gt_norm.shape}\n"
            f"hint={hint_norm.shape}"
        )

    pred_speed = norm_to_speed(pred_norm)
    gt_speed = norm_to_speed(gt_norm)
    hint_speed = norm_to_speed(hint_norm)

    # No global clip here: the candidate clip is applied per-sample
    # AFTER transposing into CBS coordinates (see loop below), so
    # clip_fraction measures clipping in CBS space.

    out = Path(args.output_dir)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    c5_dir = Path(args.c5_val_dir)

    n = min(
        args.max_samples,
        len(pred_speed),
    )

    rows = []

    print("=" * 120)
    print("C6.2 REAL C4 -> CBS PHYSICAL BRIDGE AUDIT")
    print("=" * 120)

    print("pred_norm =", args.pred_norm)
    print("gt_norm   =", args.gt_norm)
    print("hint_norm =", args.hint_norm)
    print("C5 val    =", args.c5_val_dir)
    print("samples   =", n)

    print()
    print(
        "C4 prediction normalized range =",
        float(pred_norm[:n].min()),
        float(pred_norm[:n].max()),
    )

    print(
        "C4 prediction speed range      =",
        float(pred_speed[:n].min()),
        float(pred_speed[:n].max()),
    )

    for i in range(1, n + 1):

        path = c5_dir / f"test_{i}.npz"

        if not path.exists():
            raise FileNotFoundError(path)

        z = np.load(
            path,
            allow_pickle=True,
        )

        target256 = (
            z["target_256"]
            .astype(np.float32)
        )

        target480 = (
            z["target_480"]
            .astype(np.float32)
        )

        # C4 array is zero-indexed;
        # C5 filenames are test_1, test_2, ...
        idx = i - 1

        # C4/condition-cache arrays use image coordinates.
        # CBS wavefield data use physical coordinates.
        # Therefore transpose 256x256 arrays before CBS bridging.
        gt256 = gt_speed[idx].T.copy()
        hint256 = hint_speed[idx].T.copy()
        # Candidate MUST enter CBS coordinates (transpose) BEFORE
        # the physical-range clip, so clip_fraction is measured on
        # the CBS-space candidate.
        raw_cand256 = pred_speed[idx].T.copy()
        cand256 = np.clip(
            raw_cand256,
            SPEED_MIN,
            SPEED_MAX,
        )
        gt480_from_c4 = resize_480(gt256)
        hint480 = resize_480(hint256)
        cand480 = resize_480(cand256)

        gt256_abs = maxabs(
            gt256,
            target256,
        )

        gt256_rr = rrmse(
            gt256,
            target256,
        )

        gt480_rr = rrmse(
            gt480_from_c4,
            target480,
        )

        hint_mse = mse(
            hint256,
            target256,
        )

        pred_mse = mse(
            cand256,
            target256,
        )

        hint_mae = mae(
            hint256,
            target256,
        )

        pred_mae = mae(
            cand256,
            target256,
        )

        mse_gain = (
            (hint_mse - pred_mse)
            / (hint_mse + 1e-12)
        )

        clip_frac = float(
            np.mean(
                cand256 != raw_cand256
            )
        )

        source_file = ""

        if "source_file" in z.files:
            source_file = str(
                np.asarray(
                    z["source_file"]
                ).reshape(-1)[0]
            )

        row = {
            "test_id": i,

            "gt256_max_abs":
                gt256_abs,

            "gt256_rrmse":
                gt256_rr,

            "gt480_bilinear_rrmse":
                gt480_rr,

            "hint_mse256":
                hint_mse,

            "pred_mse256":
                pred_mse,

            "mse_gain_frac":
                mse_gain,

            "hint_mae256":
                hint_mae,

            "pred_mae256":
                pred_mae,

            "pred_clip_fraction":
                clip_frac,

            "source_file":
                source_file,
        }

        rows.append(row)

        # Carry everything required by the
        # next true-CBS / ANO experiment.
        payload = {
            "candidate_256":
                cand256.astype(np.float32),

            "candidate_480":
                cand480.astype(np.float32),

            "candidate_256_raw":
                raw_cand256.astype(
                    np.float32
                ),

            "hint_256":
                hint256.astype(np.float32),

            "hint_480":
                hint480.astype(np.float32),

            "c4_gt_256":
                gt256.astype(np.float32),

            "c4_gt_480_bilinear":
                gt480_from_c4.astype(
                    np.float32
                ),

            "target_256":
                target256,

            "target_480":
                target480,
        }

        copy_keys = [
            "source_positions",
            "src_indices",
            "rec_indices",
            "dobs_complex",
            "dobs_saved",
            "wavefields",
            "source_file",
            "measurement_mode",
            "frequency",
            "cbs_iters",
            "boundary_width",
            "boundary_strength",
            "boundary_type",
            "alignment_rrmse",
        ]

        for key in copy_keys:
            if key in z.files:
                payload[key] = z[key]

        np.savez_compressed(
            out / f"test_{i}.npz",
            **payload,
        )

        print(
            f"test_{i:02d} | "
            f"GT256 maxabs={gt256_abs:.6e} "
            f"RR={gt256_rr:.6e} | "
            f"GT480 interp RR={gt480_rr:.6e} | "
            f"hint MSE={hint_mse:.5f} "
            f"C4 MSE={pred_mse:.5f} "
            f"gain={100*mse_gain:+.3f}% | "
            f"clip={100*clip_frac:.3f}%"
        )

    csv_path = out / "alignment.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    max_abs_256 = max(
        r["gt256_max_abs"]
        for r in rows
    )

    max_rr_256 = max(
        r["gt256_rrmse"]
        for r in rows
    )

    mean_interp480_rr = float(
        np.mean(
            [
                r["gt480_bilinear_rrmse"]
                for r in rows
            ]
        )
    )

    hint_mse_mean = float(
        np.mean(
            [
                r["hint_mse256"]
                for r in rows
            ]
        )
    )

    pred_mse_mean = float(
        np.mean(
            [
                r["pred_mse256"]
                for r in rows
            ]
        )
    )

    alignment_pass = (
        max_abs_256
        <= args.max_gt256_abs_tol
        and
        max_rr_256
        <= args.max_gt256_rrmse_tol
    )

    summary = {
        "num_samples":
            n,

        "pred_norm_path":
            str(args.pred_norm),

        "max_gt256_abs":
            max_abs_256,

        "max_gt256_rrmse":
            max_rr_256,

        "mean_gt480_bilinear_rrmse":
            mean_interp480_rr,

        "hint_mse256_mean":
            hint_mse_mean,

        "c4_pred_mse256_mean":
            pred_mse_mean,

        "c4_pred_mse_gain_frac":
            (
                (hint_mse_mean - pred_mse_mean)
                /
                (hint_mse_mean + 1e-12)
            ),

        "alignment_pass":
            bool(alignment_pass),

        "gt256_abs_tol":
            args.max_gt256_abs_tol,

        "gt256_rrmse_tol":
            args.max_gt256_rrmse_tol,
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

    print()
    print("=" * 120)
    print("C6.2 SUMMARY")
    print("=" * 120)

    for k, v in summary.items():
        print(
            f"{k:32s} = {v}"
        )

    print()

    if alignment_pass:
        print(
            "[STRONG PASS] "
            "C4 VAL index / normalization / "
            "C5 target_256 identity verified."
        )

        print(
            "Ready for C6.3 true-CBS "
            "physics correction on real "
            "Conditional-CM samples."
        )

    else:
        print(
            "[FAIL] C4 and C5 sample identity "
            "is not sufficiently aligned."
        )

        print(
            "DO NOT start C6.3 before "
            "resolving this mismatch."
        )

        raise SystemExit(2)


if __name__ == "__main__":
    main()
