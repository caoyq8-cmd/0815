from pathlib import Path
import json
import csv
import math

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# Canonical image metrics
# =============================================================================

V_MIN = 1400.0
V_MAX = 1605.0
DATA_RANGE = V_MAX - V_MIN


def gaussian_window(
    window_size=11,
    sigma=1.5,
    device="cpu",
    dtype=torch.float32,
):
    coords = (
        torch.arange(
            window_size,
            dtype=dtype,
            device=device,
        )
        - window_size // 2
    )

    g = torch.exp(
        -(coords ** 2) / (2 * sigma ** 2)
    )
    g = g / g.sum()

    return torch.outer(g, g)[None, None]


def ssim_np(
    pred,
    target,
    window_size=11,
    sigma=1.5,
):

    pred_t = torch.from_numpy(
        pred.astype(np.float32)
    )[None, None]

    target_t = torch.from_numpy(
        target.astype(np.float32)
    )[None, None]

    pred_t = (
        (pred_t - V_MIN) / DATA_RANGE
    ).clamp(0.0, 1.0)

    target_t = (
        (target_t - V_MIN) / DATA_RANGE
    ).clamp(0.0, 1.0)

    w = gaussian_window(
        window_size,
        sigma,
        pred_t.device,
        pred_t.dtype,
    )

    mu1 = F.conv2d(
        pred_t,
        w,
        padding=window_size // 2,
    )

    mu2 = F.conv2d(
        target_t,
        w,
        padding=window_size // 2,
    )

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2

    s1 = (
        F.conv2d(
            pred_t * pred_t,
            w,
            padding=window_size // 2,
        )
        - mu1_sq
    )

    s2 = (
        F.conv2d(
            target_t * target_t,
            w,
            padding=window_size // 2,
        )
        - mu2_sq
    )

    s12 = (
        F.conv2d(
            pred_t * target_t,
            w,
            padding=window_size // 2,
        )
        - mu12
    )

    s1 = torch.clamp(s1, min=0.0)
    s2 = torch.clamp(s2, min=0.0)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (
        (2 * mu12 + c1)
        * (2 * s12 + c2)
    )

    denominator = (
        (mu1_sq + mu2_sq + c1)
        * (s1 + s2 + c2)
        + 1e-12
    )

    return float(
        (numerator / denominator)
        .mean()
        .item()
    )


def metrics(pred, target):

    diff = (
        pred.astype(np.float64)
        - target.astype(np.float64)
    )

    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))

    if mse <= 1e-12:
        psnr = 99.0
    else:
        psnr = float(
            20.0
            * math.log10(
                DATA_RANGE / rmse
            )
        )

    ssim = ssim_np(pred, target)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": ssim,
    }


# =============================================================================
# Paths
# =============================================================================

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
# Evaluate
# =============================================================================

rows = []

for sid in range(1, 401):

    pn = (
        NAIVE
        / f"test_{sid}"
        / "final_result.npz"
    )

    pp = (
        PGCR
        / f"test_{sid}"
        / "final_result.npz"
    )

    if not pn.exists():
        raise FileNotFoundError(pn)

    if not pp.exists():
        raise FileNotFoundError(pp)

    n = np.load(pn)
    p = np.load(pp)

    # ---------------------------------------------------------
    # hard identity checks
    # ---------------------------------------------------------

    target_n = n["target_256_phys"].astype(
        np.float32
    )

    target_p = p["target_256_phys"].astype(
        np.float32
    )

    init_n = n["condition_256_phys"].astype(
        np.float32
    )

    init_p = p["condition_256_phys"].astype(
        np.float32
    )

    if np.max(
        np.abs(target_n - target_p)
    ) > 1e-6:
        raise RuntimeError(
            f"target mismatch sid={sid}"
        )

    if np.max(
        np.abs(init_n - init_p)
    ) > 1e-6:
        raise RuntimeError(
            f"condition mismatch sid={sid}"
        )

    naive = n[
        "corrected_256_phys"
    ].astype(np.float32)

    pgcr = p[
        "corrected_256_phys"
    ].astype(np.float32)

    m_init = metrics(
        init_p,
        target_p,
    )

    m_naive = metrics(
        naive,
        target_p,
    )

    m_pgcr = metrics(
        pgcr,
        target_p,
    )

    row = {
        "sid": sid,
    }

    for prefix, mm in [
        ("inv", m_init),
        ("naive", m_naive),
        ("pgcr", m_pgcr),
    ]:
        for k, v in mm.items():
            row[
                f"{prefix}_{k}"
            ] = v

    rows.append(row)


# =============================================================================
# Aggregate
# =============================================================================

def arr(key):
    return np.asarray(
        [r[key] for r in rows],
        dtype=np.float64,
    )


methods = [
    ("InversionNet", "inv"),
    ("Naive-CBS8", "naive"),
    ("PGCR", "pgcr"),
]

summary = {}

print("=" * 125)
print("CANONICAL TEST400 IMAGE METRICS")
print("=" * 125)

print(
    f"{'method':16s}"
    f"{'MSE':>12s}"
    f"{'MAE':>12s}"
    f"{'RMSE':>12s}"
    f"{'PSNR':>12s}"
    f"{'SSIM':>12s}"
)

print("-" * 125)

for name, prefix in methods:

    s = {
        k: float(
            arr(f"{prefix}_{k}").mean()
        )
        for k in [
            "mse",
            "mae",
            "rmse",
            "psnr",
            "ssim",
        ]
    }

    summary[name] = s

    print(
        f"{name:16s}"
        f"{s['mse']:12.6f}"
        f"{s['mae']:12.6f}"
        f"{s['rmse']:12.6f}"
        f"{s['psnr']:12.6f}"
        f"{s['ssim']:12.6f}"
    )

print("=" * 125)


# =============================================================================
# Pairwise wins and improvement
# =============================================================================

summary["pairwise"] = {

    "Naive_vs_Inv": {
        "mse_wins": int(
            np.sum(
                arr("naive_mse")
                <
                arr("inv_mse")
            )
        ),

        "mae_wins": int(
            np.sum(
                arr("naive_mae")
                <
                arr("inv_mae")
            )
        ),

        "psnr_wins": int(
            np.sum(
                arr("naive_psnr")
                >
                arr("inv_psnr")
            )
        ),

        "ssim_wins": int(
            np.sum(
                arr("naive_ssim")
                >
                arr("inv_ssim")
            )
        ),
    },

    "PGCR_vs_Inv": {
        "mse_wins": int(
            np.sum(
                arr("pgcr_mse")
                <
                arr("inv_mse")
            )
        ),

        "mae_wins": int(
            np.sum(
                arr("pgcr_mae")
                <
                arr("inv_mae")
            )
        ),

        "psnr_wins": int(
            np.sum(
                arr("pgcr_psnr")
                >
                arr("inv_psnr")
            )
        ),

        "ssim_wins": int(
            np.sum(
                arr("pgcr_ssim")
                >
                arr("inv_ssim")
            )
        ),
    },

    "PGCR_vs_Naive": {
        "mse_wins": int(
            np.sum(
                arr("pgcr_mse")
                <
                arr("naive_mse")
            )
        ),

        "mae_wins": int(
            np.sum(
                arr("pgcr_mae")
                <
                arr("naive_mae")
            )
        ),

        "psnr_wins": int(
            np.sum(
                arr("pgcr_psnr")
                >
                arr("naive_psnr")
            )
        ),

        "ssim_wins": int(
            np.sum(
                arr("pgcr_ssim")
                >
                arr("naive_ssim")
            )
        ),
    },
}


inv_mse = summary["InversionNet"]["mse"]
naive_mse = summary["Naive-CBS8"]["mse"]
pgcr_mse = summary["PGCR"]["mse"]

summary["MSE_improvement_percent"] = {
    "Naive_vs_Inv":
        100.0
        * (inv_mse - naive_mse)
        / inv_mse,

    "PGCR_vs_Inv":
        100.0
        * (inv_mse - pgcr_mse)
        / inv_mse,

    "PGCR_vs_Naive":
        100.0
        * (naive_mse - pgcr_mse)
        / naive_mse,
}


print()
print("PAIRWISE WINS / 400")
print("-" * 125)

for name, s in summary[
    "pairwise"
].items():
    print(
        f"{name:20s} | "
        f"MSE {s['mse_wins']:3d} | "
        f"MAE {s['mae_wins']:3d} | "
        f"PSNR {s['psnr_wins']:3d} | "
        f"SSIM {s['ssim_wins']:3d}"
    )

print()
print(
    "MSE improvement PGCR vs InversionNet = "
    f"{summary['MSE_improvement_percent']['PGCR_vs_Inv']:.4f}%"
)

print(
    "MSE improvement PGCR vs Naive-CBS8 = "
    f"{summary['MSE_improvement_percent']['PGCR_vs_Naive']:.4f}%"
)


# =============================================================================
# Save per-sample CSV
# =============================================================================

csv_path = OUT / (
    "canonical_test400_"
    "inversionnet_naive_pgcr.csv"
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


json_path = OUT / (
    "canonical_test400_"
    "inversionnet_naive_pgcr.json"
)

json_path.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)

print()
print("saved CSV  =", csv_path)
print("saved JSON =", json_path)
