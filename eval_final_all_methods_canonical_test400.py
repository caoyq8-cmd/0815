from pathlib import Path
import csv
import json
import math

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# Canonical metric definition
# =============================================================================

V_MIN = 1400.0
V_MAX = 1605.0
DATA_RANGE = V_MAX - V_MIN


def gaussian_window(
    window_size=11,
    sigma=1.5,
    dtype=torch.float32,
):
    coords = (
        torch.arange(
            window_size,
            dtype=dtype,
        )
        - window_size // 2
    )

    g = torch.exp(
        -(coords ** 2)
        / (2 * sigma ** 2)
    )

    g = g / g.sum()

    return torch.outer(g, g)[None, None]


def ssim_np(pred, target):

    pred_t = torch.from_numpy(
        pred.astype(np.float32)
    )[None, None]

    target_t = torch.from_numpy(
        target.astype(np.float32)
    )[None, None]

    pred_t = (
        (pred_t - V_MIN)
        / DATA_RANGE
    ).clamp(0, 1)

    target_t = (
        (target_t - V_MIN)
        / DATA_RANGE
    ).clamp(0, 1)

    w = gaussian_window(
        11,
        1.5,
        pred_t.dtype,
    )

    mu1 = F.conv2d(
        pred_t,
        w,
        padding=5,
    )

    mu2 = F.conv2d(
        target_t,
        w,
        padding=5,
    )

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2

    sig1 = (
        F.conv2d(
            pred_t * pred_t,
            w,
            padding=5,
        )
        - mu1_sq
    )

    sig2 = (
        F.conv2d(
            target_t * target_t,
            w,
            padding=5,
        )
        - mu2_sq
    )

    sig12 = (
        F.conv2d(
            pred_t * target_t,
            w,
            padding=5,
        )
        - mu12
    )

    sig1 = torch.clamp(
        sig1,
        min=0.0,
    )

    sig2 = torch.clamp(
        sig2,
        min=0.0,
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim = (
        (2 * mu12 + c1)
        * (2 * sig12 + c2)
        /
        (
            (mu1_sq + mu2_sq + c1)
            * (sig1 + sig2 + c2)
            + 1e-12
        )
    )

    return float(
        ssim.mean().item()
    )


def metrics(pred, gt):

    d = (
        pred.astype(np.float64)
        - gt.astype(np.float64)
    )

    mse = float(
        np.mean(d ** 2)
    )

    mae = float(
        np.mean(np.abs(d))
    )

    rmse = float(
        np.sqrt(mse)
    )

    if mse <= 1e-12:
        psnr = 99.0
    else:
        psnr = float(
            20
            * math.log10(
                DATA_RANGE / rmse
            )
        )

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": ssim_np(
            pred,
            gt,
        ),
    }


def squeeze400(x):

    x = np.asarray(x)

    if x.ndim == 4:
        if x.shape[1] == 1:
            x = x[:, 0]
        elif x.shape[-1] == 1:
            x = x[..., 0]

    if x.shape != (
        400,
        256,
        256,
    ):
        raise RuntimeError(
            f"unexpected prediction shape: {x.shape}"
        )

    return x.astype(
        np.float32
    )


# =============================================================================
# Paths
# =============================================================================

C4ROOT = Path(
    "/home/featurize/work/USCT_repro/"
    "paper_reproduction/cosign_usct/"
    "metrics/c4_frozen_test400_v1"
)

CACHE = Path(
    "/home/featurize/work/USCT_repro/"
    "USCT_download/condition_cache/"
    "inversionnet_formal_3600_400_400_e89/"
    "test"
)

NAIVE = Path(
    "/home/featurize/work/USCT_repro/"
    "USCT_download/"
    "formal_diffano_truecbs_test400"
)

PGCR = Path(
    "/home/featurize/work/USCT_repro/"
    "USCT_download/"
    "pgcr_frozen_test400"
)

OUT = Path(
    "/home/featurize/work/USCT_repro/"
    "USCT_download/"
    "final_thesis_metrics"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Load C4 predictions
# =============================================================================

preds = {
    "InversionNet":
        squeeze400(
            np.load(
                C4ROOT
                / "inversionnet_pred_speed.npy"
            )
        ),

    "Raw-C4":
        squeeze400(
            np.load(
                C4ROOT
                / "c4_raw_pred_speed.npy"
            )
        ),

    "HP-C4":
        squeeze400(
            np.load(
                C4ROOT
                / "hp_c4_pred_speed.npy"
            )
        ),

    "AGR-MSE":
        squeeze400(
            np.load(
                C4ROOT
                / "agr_mse_pred_speed.npy"
            )
        ),

    "AGR-Stable":
        squeeze400(
            np.load(
                C4ROOT
                / "agr_stable_pred_speed.npy"
            )
        ),
}


# =============================================================================
# Load canonical GT and verify InversionNet identity
# =============================================================================

targets = []

for sid in range(1, 401):

    z = np.load(
        CACHE
        / f"test_{sid}.npz"
    )

    gt = z[
        "target_speed"
    ][0].astype(
        np.float32
    )

    cond = z[
        "condition_speed"
    ][0].astype(
        np.float32
    )

    targets.append(gt)

    err = float(
        np.max(
            np.abs(
                cond
                - preds["InversionNet"][
                    sid - 1
                ]
            )
        )
    )

    if err > 1e-5:
        raise RuntimeError(
            f"InversionNet cache mismatch "
            f"sid={sid}, maxabs={err}"
        )

targets = np.stack(
    targets,
    axis=0,
)


# =============================================================================
# Add physics methods
# =============================================================================

naive_all = []
pgcr_all = []

for sid in range(1, 401):

    n = np.load(
        NAIVE
        / f"test_{sid}"
        / "final_result.npz"
    )

    p = np.load(
        PGCR
        / f"test_{sid}"
        / "final_result.npz"
    )

    naive = n[
        "corrected_256_phys"
    ].astype(np.float32)

    pgcr = p[
        "corrected_256_phys"
    ].astype(np.float32)

    gt_n = n[
        "target_256_phys"
    ].astype(np.float32)

    gt_p = p[
        "target_256_phys"
    ].astype(np.float32)

    gt = targets[
        sid - 1
    ]

    if (
        np.max(
            np.abs(gt_n - gt)
        )
        > 1e-5
    ):
        raise RuntimeError(
            f"Naive target mismatch {sid}"
        )

    if (
        np.max(
            np.abs(gt_p - gt)
        )
        > 1e-5
    ):
        raise RuntimeError(
            f"PGCR target mismatch {sid}"
        )

    naive_all.append(
        naive
    )

    pgcr_all.append(
        pgcr
    )

preds["Naive-CBS8"] = np.stack(
    naive_all
)

preds["PGCR"] = np.stack(
    pgcr_all
)


# =============================================================================
# Evaluation
# =============================================================================

rows = []

method_names = list(
    preds.keys()
)

for sid in range(1, 401):

    gt = targets[
        sid - 1
    ]

    row = {
        "sid": sid
    }

    for method in method_names:

        m = metrics(
            preds[method][sid - 1],
            gt,
        )

        prefix = (
            method
            .lower()
            .replace("-", "_")
        )

        for k, v in m.items():
            row[
                f"{prefix}_{k}"
            ] = v

    rows.append(row)


# =============================================================================
# Aggregate
# =============================================================================

def a(method, metric):

    prefix = (
        method
        .lower()
        .replace("-", "_")
    )

    key = (
        f"{prefix}_{metric}"
    )

    return np.asarray(
        [
            r[key]
            for r in rows
        ],
        dtype=np.float64,
    )


summary = {}

print("=" * 125)
print("FINAL CANONICAL TEST400 — ALL METHODS")
print("=" * 125)

print(
    f"{'Method':16s}"
    f"{'MSE':>12s}"
    f"{'MAE':>12s}"
    f"{'RMSE':>12s}"
    f"{'PSNR':>12s}"
    f"{'SSIM':>12s}"
)

print("-" * 125)

for method in method_names:

    s = {}

    for metric in [
        "mse",
        "mae",
        "rmse",
        "psnr",
        "ssim",
    ]:
        s[metric] = float(
            a(
                method,
                metric,
            ).mean()
        )

    summary[method] = s

    print(
        f"{method:16s}"
        f"{s['mse']:12.6f}"
        f"{s['mae']:12.6f}"
        f"{s['rmse']:12.6f}"
        f"{s['psnr']:12.6f}"
        f"{s['ssim']:12.6f}"
    )

print("=" * 125)


# =============================================================================
# Wins against InversionNet
# =============================================================================

base = "InversionNet"

wins = {}

print()
print("WINS VS INVERSIONNET / 400")
print("-" * 110)

for method in method_names:

    if method == base:
        continue

    w = {
        "MSE":
            int(
                np.sum(
                    a(method, "mse")
                    <
                    a(base, "mse")
                )
            ),

        "MAE":
            int(
                np.sum(
                    a(method, "mae")
                    <
                    a(base, "mae")
                )
            ),

        "PSNR":
            int(
                np.sum(
                    a(method, "psnr")
                    >
                    a(base, "psnr")
                )
            ),

        "SSIM":
            int(
                np.sum(
                    a(method, "ssim")
                    >
                    a(base, "ssim")
                )
            ),
    }

    wins[method] = w

    print(
        f"{method:16s} | "
        f"MSE {w['MSE']:3d} | "
        f"MAE {w['MAE']:3d} | "
        f"PSNR {w['PSNR']:3d} | "
        f"SSIM {w['SSIM']:3d}"
    )

summary["wins_vs_inversionnet"] = wins


# =============================================================================
# Save
# =============================================================================

csv_path = (
    OUT
    / "canonical_test400_all_methods.csv"
)

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(rows)

json_path = (
    OUT
    / "canonical_test400_all_methods.json"
)

json_path.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)

print()
print(
    "saved CSV  =",
    csv_path,
)

print(
    "saved JSON =",
    json_path,
)
