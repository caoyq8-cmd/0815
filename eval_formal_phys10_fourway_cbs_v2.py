import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cbs_model import ConvergentBornSeries_Batch


CENTER = 1502.5
SCALE = 102.5


def upsample_256_to_480_phys(speed_img):
    """
    Formal arrays use image coordinates.
    CBS uses physical coordinates.

    image [256,256]
       -> transpose
       -> bilinear [480,480]
    """

    x = speed_img.astype(
        np.float32
    )

    t = torch.from_numpy(
        np.ascontiguousarray(x)
    )[None, None].float()

    y = F.interpolate(
        t,
        size=(480,480),
        mode="bilinear",
        align_corners=False,
    )

    return (
        y[0,0]
        .cpu()
        .numpy()
        .astype(np.float32)
    )


@torch.no_grad()
def cbs_measurement(
    speed480,
    src_indices,
    rec_indices,
    frequency,
    cbs_iters,
    boundary_width,
    boundary_strength,
    boundary_type,
    device,
):

    sos = torch.from_numpy(
        speed480
    ).view(
        1,1,480,480
    ).to(
        device=device,
        dtype=torch.float32,
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

    rec = torch.from_numpy(
        rec_indices.astype(np.int64)
    ).to(
        device=device,
        dtype=torch.long,
    )

    dobs = (
        u[
            0,
            :,
            rec[:,0],
            rec[:,1],
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.complex64)
    )

    del u
    del model
    del sos
    del rec

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dobs


def physics_metrics(pred, obs):
    resid = pred - obs

    num2 = float(
        np.sum(
            np.abs(resid) ** 2
        )
    )

    den2 = float(
        np.sum(
            np.abs(obs) ** 2
        )
    )

    rrmse = float(
        np.sqrt(
            num2
            / max(
                den2,
                1e-30,
            )
        )
    )

    objective = (
        0.5 * num2
    )

    normalized_objective = (
        num2
        / max(
            den2,
            1e-30,
        )
    )

    return {
        "cbs_objective":
            objective,

        "measurement_rrmse":
            rrmse,

        "normalized_objective":
            normalized_objective,
    }


def image_metrics(pred_norm, gt_norm):

    err_mps = (
        SCALE
        * (
            pred_norm
            - gt_norm
        )
    )

    return {
        "image_mse":
            float(
                np.mean(
                    err_mps ** 2
                )
            ),

        "image_mae":
            float(
                np.mean(
                    np.abs(
                        err_mps
                    )
                )
            ),
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--formal_root",
        required=True,
    )

    ap.add_argument(
        "--raw_pred",
        required=True,
    )

    ap.add_argument(
        "--cbs_root",
        required=True,
    )

    ap.add_argument(
        "--output_dir",
        required=True,
    )

    ap.add_argument(
        "--start_id",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--end_id",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--device",
        default="cuda:0",
    )

    args = ap.parse_args()

    formal_root = Path(
        args.formal_root
    )

    cbs_root = Path(
        args.cbs_root
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gt = np.load(
        formal_root
        / "val_gt_norm.npy"
    ).astype(np.float32)

    hint = np.load(
        formal_root
        / "val_cond_norm.npy"
    ).astype(np.float32)

    raw = np.load(
        args.raw_pred
    ).astype(np.float32)

    assert (
        gt.shape
        == hint.shape
        == raw.shape
    )

    delta = raw - hint

    delta_mps = (
        SCALE * delta
    )

    hard3 = (
        np.abs(
            delta_mps
        )
        >= 3.0
    ).astype(np.float32)

    # AGR-Stable from the original AGR sweep:
    # mode=soft, tau=3 m/s, upper=inf, alpha=0.125.
    soft3_mps = (
        np.sign(delta_mps)
        * np.maximum(
            np.abs(delta_mps) - 3.0,
            0.0
        )
    )

    soft3_norm = (
        soft3_mps
        / SCALE
    )

    candidates = {
        "InversionNet":
            hint,

        "HP-C4":
            hint
            + 0.20
            * delta,

        "AGR-MSE":
            hint
            + 0.175
            * hard3
            * delta,

        "AGR-Stable":
            hint
            + 0.125
            * soft3_norm,
    }

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device =",
        device,
    )

    rows = []

    for sid in range(
        args.start_id,
        args.end_id + 1,
    ):

        idx = sid - 1

        p = (
            cbs_root
            / f"test_{sid}.npz"
        )

        z = np.load(
            p,
            allow_pickle=True,
        )

        obs = (
            z["dobs_complex"]
            .astype(np.complex64)
        )

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
        print("=" * 108)
        print(
            f"SAMPLE {sid:02d} | "
            f"original_index="
            f"{int(z['original_index'][0])}"
        )
        print("=" * 108)

        for method, arr in candidates.items():

            pred_norm = (
                arr[idx]
                .astype(np.float32)
            )

            # NO clipping here.
            speed256_img = (
                CENTER
                + SCALE
                * pred_norm
            ).astype(np.float32)

            speed480_phys = (
                upsample_256_to_480_phys(
                    speed256_img
                )
            )

            pred_dobs = (
                cbs_measurement(
                    speed480=speed480_phys,
                    src_indices=src_indices,
                    rec_indices=rec_indices,
                    frequency=frequency,
                    cbs_iters=cbs_iters,
                    boundary_width=boundary_width,
                    boundary_strength=boundary_strength,
                    boundary_type=boundary_type,
                    device=device,
                )
            )

            im = image_metrics(
                pred_norm,
                gt[idx],
            )

            ph = physics_metrics(
                pred_dobs,
                obs,
            )

            row = {
                "formal_val_id":
                    sid,

                "original_index":
                    int(
                        z[
                            "original_index"
                        ][0]
                    ),

                "method":
                    method,

                "image_mse":
                    im["image_mse"],

                "image_mae":
                    im["image_mae"],

                "cbs_objective":
                    ph[
                        "cbs_objective"
                    ],

                "measurement_rrmse":
                    ph[
                        "measurement_rrmse"
                    ],

                "normalized_objective":
                    ph[
                        "normalized_objective"
                    ],

                "speed256_min":
                    float(
                        speed256_img.min()
                    ),

                "speed256_max":
                    float(
                        speed256_img.max()
                    ),
            }

            rows.append(
                row
            )

            print(
                f"{method:14s} | "
                f"imgMSE="
                f"{row['image_mse']:9.4f} | "
                f"MAE="
                f"{row['image_mae']:7.4f} | "
                f"CBS-RRMSE="
                f"{row['measurement_rrmse']:.8f} | "
                f"Jnorm="
                f"{row['normalized_objective']:.8e} | "
                f"range=["
                f"{row['speed256_min']:.2f},"
                f"{row['speed256_max']:.2f}]"
            )

    # --------------------------------------------------
    # Save detailed table
    # --------------------------------------------------

    csv_path = (
        output_dir
        / "fourway_per_sample.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            )
        )

        w.writeheader()
        w.writerows(rows)

    # --------------------------------------------------
    # Aggregate
    # --------------------------------------------------

    names = [
        "InversionNet",
        "HP-C4",
        "AGR-MSE",
        "AGR-Stable",
    ]

    summary = {}

    base_rr = {
        r["formal_val_id"]:
            r["measurement_rrmse"]

        for r in rows
        if r["method"]
        == "InversionNet"
    }

    base_im = {
        r["formal_val_id"]:
            r["image_mse"]

        for r in rows
        if r["method"]
        == "InversionNet"
    }

    for name in names:

        rr = [
            r
            for r in rows
            if r["method"] == name
        ]

        physics_wins = sum(
            r["measurement_rrmse"]
            <
            base_rr[
                r["formal_val_id"]
            ]

            for r in rr
        )

        image_wins = sum(
            r["image_mse"]
            <
            base_im[
                r["formal_val_id"]
            ]

            for r in rr
        )

        mean_rr = float(
            np.mean([
                r["measurement_rrmse"]
                for r in rr
            ])
        )

        mean_img = float(
            np.mean([
                r["image_mse"]
                for r in rr
            ])
        )

        if name == "InversionNet":
            physics_gain = 0.0
            image_gain = 0.0

        else:
            base_mean_rr = float(
                np.mean(
                    list(
                        base_rr.values()
                    )
                )
            )

            base_mean_img = float(
                np.mean(
                    list(
                        base_im.values()
                    )
                )
            )

            physics_gain = (
                (
                    base_mean_rr
                    - mean_rr
                )
                / base_mean_rr
            )

            image_gain = (
                (
                    base_mean_img
                    - mean_img
                )
                / base_mean_img
            )

        summary[name] = {
            "mean_image_mse":
                mean_img,

            "mean_measurement_rrmse":
                mean_rr,

            "image_wins_vs_inv":
                int(
                    image_wins
                ),

            "physics_wins_vs_inv":
                int(
                    physics_wins
                ),

            "image_gain_fraction":
                float(
                    image_gain
                ),

            "physics_gain_fraction":
                float(
                    physics_gain
                ),
        }

    summary_path = (
        output_dir
        / "fourway_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 108)
    print("DEV SUMMARY")
    print("=" * 108)

    for name in names:

        s = summary[name]

        print(
            f"{name:14s} | "
            f"Image MSE="
            f"{s['mean_image_mse']:.6f} | "
            f"ImgWin="
            f"{s['image_wins_vs_inv']:2d}/"
            f"{len(base_im)} | "
            f"CBS-RRMSE="
            f"{s['mean_measurement_rrmse']:.8f} | "
            f"PhysWin="
            f"{s['physics_wins_vs_inv']:2d}/"
            f"{len(base_rr)} | "
            f"PhysGain="
            f"{100*s['physics_gain_fraction']:+.3f}%"
        )

    print()
    print(
        "saved =",
        csv_path,
    )

    print(
        "saved =",
        summary_path,
    )


if __name__ == "__main__":
    main()
