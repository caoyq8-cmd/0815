
#!/usr/bin/env python3
"""
C6.7 Structural Trade-off and Error Redistribution Audit

Purpose
-------
Explain the C6.6 observation:
    measurement J decreases,
    MSE often decreases,
but
    MAE and SSIM can worsen.

This is a pure post-processing audit. It does NOT:
- retrain any model,
- rerun CBS/MgNO/CM,
- tune any method parameter,
- use GT for physics acceptance.

Expected C6.6 trajectory files:
    test_<id>_terminal_trajectory.npz
with keys:
    hint_norm
    gt_norm
    c4_final_norm
    terminal_exact_norm
    terminal_gra_norm

Main analyses
-------------
1) Absolute-error quantiles: p50/p75/p90/p95/p99/max
2) Spatial redistribution:
   - full image
   - inner crop32 (192x192)
   - border32 band
3) Scale-space metrics:
   - raw
   - box smooth5
   - box smooth9
4) Pixelwise redistribution:
   - fraction of pixels improved/worsened vs C4-only
   - top-10% C4-error region vs remaining 90%
5) Terminal-update spectrum:
   - low / mid / high radial-frequency energy
   - update RMS and total variation
6) Paired statistics:
   - delta MSE / MAE / SSIM vs C4-only
   - bootstrap 95% CI of mean delta
   - Wilcoxon signed-rank p-value (if scipy.stats available)

No claim should be based on one metric alone.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from scipy.ndimage import uniform_filter
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

from skimage.metrics import structural_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CENTER = 1502.5
SCALE = 102.5
SPEED_MIN = 1400.0
SPEED_MAX = 1605.0
DATA_RANGE = SPEED_MAX - SPEED_MIN

METHOD_KEYS = {
    "C4-only": "c4_final_norm",
    "C4+TExact": "terminal_exact_norm",
    "C4+TGRA": "terminal_gra_norm",
}

METHOD_ORDER = [
    "C4-only",
    "C4+TExact",
    "C4+TGRA",
]

ERR_QS = [50, 75, 90, 95, 99, 100]


def norm_to_speed(x):
    return np.asarray(x, dtype=np.float64) * SCALE + CENTER


def mse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def mae(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def psnr_from_mse(v):
    return float(
        10.0 * math.log10(
            DATA_RANGE ** 2 / max(float(v), 1e-30)
        )
    )


def ssim(a, b):
    return float(
        structural_similarity(
            np.asarray(a, dtype=np.float64),
            np.asarray(b, dtype=np.float64),
            data_range=DATA_RANGE,
        )
    )


def stat(vals):
    x = np.asarray(
        [v for v in vals if np.isfinite(v)],
        dtype=np.float64,
    )
    if x.size == 0:
        return None
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def bootstrap_mean_ci(vals, seed=20260906, n_boot=10000):
    x = np.asarray(
        [v for v in vals if np.isfinite(v)],
        dtype=np.float64,
    )
    if x.size == 0:
        return None
    if x.size == 1:
        return {
            "mean": float(x[0]),
            "low95": float(x[0]),
            "high95": float(x[0]),
            "n": 1,
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(
        0,
        x.size,
        size=(n_boot, x.size),
    )
    means = x[idx].mean(axis=1)

    return {
        "mean": float(x.mean()),
        "low95": float(np.percentile(means, 2.5)),
        "high95": float(np.percentile(means, 97.5)),
        "n": int(x.size),
    }


def wilcoxon_p(vals):
    if wilcoxon is None:
        return None

    x = np.asarray(
        [v for v in vals if np.isfinite(v)],
        dtype=np.float64,
    )

    if x.size == 0:
        return None

    if np.all(np.abs(x) < 1e-30):
        return 1.0

    try:
        res = wilcoxon(
            x,
            zero_method="wilcox",
            alternative="two-sided",
            mode="auto",
        )
        return float(res.pvalue)
    except Exception:
        return None


def write_csv(path, rows):
    path = Path(path)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()

    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="raise",
        )
        w.writeheader()
        w.writerows(rows)


def region_masks(shape=(256, 256), border=32):
    h, w = shape

    inner = np.zeros(shape, dtype=bool)
    inner[
        border:h-border,
        border:w-border,
    ] = True

    border_mask = ~inner

    return {
        "full": np.ones(shape, dtype=bool),
        f"inner_crop{border}": inner,
        f"border{border}": border_mask,
    }


def region_metrics(pred, gt, mask):
    e = np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64)
    v = e[mask]
    return {
        "mse": float(np.mean(v ** 2)),
        "mae": float(np.mean(np.abs(v))),
    }


def smooth_metrics(pred, gt, size):
    pred_s = uniform_filter(
        np.asarray(pred, dtype=np.float64),
        size=size,
        mode="nearest",
    )
    gt_s = uniform_filter(
        np.asarray(gt, dtype=np.float64),
        size=size,
        mode="nearest",
    )

    m = mse(pred_s, gt_s)

    return {
        "mse": m,
        "mae": mae(pred_s, gt_s),
        "psnr": psnr_from_mse(m),
        "ssim": ssim(pred_s, gt_s),
    }


def abs_error_quantiles(pred, gt):
    ae = np.abs(
        np.asarray(pred, dtype=np.float64)
        - np.asarray(gt, dtype=np.float64)
    ).ravel()

    out = {}

    for q in ERR_QS:
        key = "max" if q == 100 else f"p{q}"
        out[key] = float(np.percentile(ae, q))

    return out


def total_variation(x):
    x = np.asarray(x, dtype=np.float64)
    dx = np.diff(x, axis=1)
    dy = np.diff(x, axis=0)

    return float(
        np.mean(np.abs(dx))
        +
        np.mean(np.abs(dy))
    )


def update_spectrum(delta):
    """
    Radial-frequency energy split using radius normalized by
    the maximum 2-D FFT radius sqrt(0.5^2+0.5^2).

    low  : r_norm <= 0.25
    mid  : 0.25 < r_norm <= 0.50
    high : r_norm > 0.50
    """
    d = np.asarray(delta, dtype=np.float64)
    h, w = d.shape

    F = np.fft.fft2(d)
    power = np.abs(F) ** 2

    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)

    yy, xx = np.meshgrid(
        fy,
        fx,
        indexing="ij",
    )

    r = np.sqrt(xx ** 2 + yy ** 2)
    rmax = math.sqrt(0.5 ** 2 + 0.5 ** 2)
    rn = r / rmax

    low = rn <= 0.25
    mid = (rn > 0.25) & (rn <= 0.50)
    high = rn > 0.50

    total = float(power.sum()) + 1e-30

    return {
        "low_energy_frac": float(power[low].sum() / total),
        "mid_energy_frac": float(power[mid].sum() / total),
        "high_energy_frac": float(power[high].sum() / total),
        "update_rms": float(np.sqrt(np.mean(d ** 2))),
        "update_mae": float(np.mean(np.abs(d))),
        "update_maxabs": float(np.max(np.abs(d))),
        "update_tv": total_variation(d),
    }


def parse_sample_id(path):
    stem = path.stem
    # test_11_terminal_trajectory
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(path)
    return int(parts[1])


def infer_split_label(run_dir):
    summary = Path(run_dir) / "summary.json"

    if summary.exists():
        try:
            d = json.loads(
                summary.read_text(
                    encoding="utf-8"
                )
            )
            return str(
                d.get(
                    "split",
                    Path(run_dir).name,
                )
            )
        except Exception:
            pass

    return Path(run_dir).name


def load_run(run_dir):
    run_dir = Path(run_dir)
    files = sorted(
        run_dir.glob(
            "test_*_terminal_trajectory.npz"
        ),
        key=parse_sample_id,
    )

    if not files:
        raise FileNotFoundError(
            f"No test_*_terminal_trajectory.npz under {run_dir}"
        )

    return {
        "run_dir": run_dir,
        "split": infer_split_label(run_dir),
        "files": files,
    }


def make_boxplot(path, rows, metric, title, ylabel):
    groups = []
    labels = []

    for method in [
        "C4+TExact",
        "C4+TGRA",
    ]:
        vals = [
            r[metric]
            for r in rows
            if r["method"] == method
        ]
        groups.append(vals)
        labels.append(method)

    plt.figure(figsize=(6.5, 4.5))
    plt.boxplot(
        groups,
        labels=labels,
        showmeans=True,
    )
    plt.axhline(
        0.0,
        linewidth=1.0,
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def make_quantile_plot(path, summary_rows):
    qs = [50, 75, 90, 95, 99, 100]
    keys = [
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
    ]

    plt.figure(figsize=(7.0, 4.8))

    for method in METHOD_ORDER:
        vals = []
        for key in keys:
            matched = [
                r["value_mean"]
                for r in summary_rows
                if (
                    r["method"] == method
                    and
                    r["quantile"] == key
                )
            ]
            vals.append(
                matched[0]
                if matched
                else np.nan
            )

        plt.plot(
            qs,
            vals,
            marker="o",
            label=method,
        )

    plt.xlabel("Absolute-error percentile")
    plt.ylabel("|error| (m/s)")
    plt.title("Absolute-error quantile profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def make_frequency_plot(path, spectral_rows):
    labels = [
        "low",
        "mid",
        "high",
    ]

    plt.figure(figsize=(7.0, 4.8))

    for method in [
        "C4+TExact",
        "C4+TGRA",
    ]:
        vals = []
        for band in labels:
            key = f"{band}_energy_frac"
            x = [
                r[key]
                for r in spectral_rows
                if r["method"] == method
            ]
            vals.append(
                float(np.mean(x))
            )

        plt.plot(
            labels,
            vals,
            marker="o",
            label=method,
        )

    plt.ylabel("Mean fraction of update spectral energy")
    plt.title("Terminal-update frequency distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def analyze_group(records, group_name):
    """
    records:
      list of dicts containing per-sample method metrics.
    """
    summary = {
        "group": group_name,
        "num_samples": len(
            set(
                (
                    r["split"],
                    r["sample"],
                )
                for r in records
            )
        ),
        "methods": {},
        "paired_vs_c4": {},
    }

    for method in METHOD_ORDER:
        sub = [
            r for r in records
            if r["method"] == method
        ]

        summary["methods"][method] = {
            "mse": stat([
                r["raw_mse"]
                for r in sub
            ]),
            "mae": stat([
                r["raw_mae"]
                for r in sub
            ]),
            "ssim": stat([
                r["raw_ssim"]
                for r in sub
            ]),
            "smooth5_mse": stat([
                r["smooth5_mse"]
                for r in sub
            ]),
            "smooth5_ssim": stat([
                r["smooth5_ssim"]
                for r in sub
            ]),
            "smooth9_mse": stat([
                r["smooth9_mse"]
                for r in sub
            ]),
            "smooth9_ssim": stat([
                r["smooth9_ssim"]
                for r in sub
            ]),
            "inner32_mse": stat([
                r["inner_crop32_mse"]
                for r in sub
            ]),
            "border32_mse": stat([
                r["border32_mse"]
                for r in sub
            ]),
            "p95_abs_error": stat([
                r["p95"]
                for r in sub
            ]),
            "p99_abs_error": stat([
                r["p99"]
                for r in sub
            ]),
            "max_abs_error": stat([
                r["max"]
                for r in sub
            ]),
        }

    for method in [
        "C4+TExact",
        "C4+TGRA",
    ]:
        sub = [
            r for r in records
            if r["method"] == method
        ]

        deltas = {}

        for metric in [
            "delta_mse_vs_c4",
            "delta_mae_vs_c4",
            "delta_ssim_vs_c4",
            "delta_smooth5_mse_vs_c4",
            "delta_smooth5_ssim_vs_c4",
            "delta_smooth9_mse_vs_c4",
            "delta_smooth9_ssim_vs_c4",
            "delta_inner32_mse_vs_c4",
            "delta_border32_mse_vs_c4",
            "delta_p95_vs_c4",
            "delta_p99_vs_c4",
            "delta_max_vs_c4",
        ]:
            vals = [
                r[metric]
                for r in sub
            ]

            deltas[metric] = {
                "stat": stat(vals),
                "bootstrap_mean_ci95":
                    bootstrap_mean_ci(vals),
                "wilcoxon_p_two_sided":
                    wilcoxon_p(vals),
                "win_count": int(
                    sum(v < 0 for v in vals)
                    if (
                        "mse" in metric
                        or
                        "mae" in metric
                        or
                        "p95" in metric
                        or
                        "p99" in metric
                        or
                        "max" in metric
                    )
                    else
                    sum(v > 0 for v in vals)
                ),
                "num_samples": len(vals),
            }

        summary[
            "paired_vs_c4"
        ][method] = {
            "metric_deltas": deltas,
            "improved_pixel_fraction": stat([
                r["improved_pixel_fraction"]
                for r in sub
            ]),
            "worsened_pixel_fraction": stat([
                r["worsened_pixel_fraction"]
                for r in sub
            ]),
            "top10_c4_error_mse_gain_frac": stat([
                r["top10_c4_error_mse_gain_frac"]
                for r in sub
            ]),
            "bottom90_c4_error_mae_gain_frac": stat([
                r["bottom90_c4_error_mae_gain_frac"]
                for r in sub
            ]),
        }

    return summary


def main():
    ap = argparse.ArgumentParser(
        description=(
            "C6.7 structural trade-off and error redistribution audit."
        )
    )

    ap.add_argument(
        "--run_dir",
        nargs="+",
        required=True,
        help=(
            "One or more frozen C6.6 run directories. "
            "Can pass DEV10 and secondary-transfer10 together."
        ),
    )

    ap.add_argument(
        "--output_dir",
        required=True,
    )

    ap.add_argument(
        "--border",
        type=int,
        default=32,
    )

    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs = [
        load_run(p)
        for p in args.run_dir
    ]

    print("=" * 150)
    print("C6.7 STRUCTURAL TRADE-OFF AND ERROR REDISTRIBUTION AUDIT")
    print("=" * 150)
    print("runs:")
    for r in runs:
        print(
            f"  {r['split']}: "
            f"{r['run_dir']} "
            f"({len(r['files'])} samples)"
        )
    print()

    metric_rows = []
    spectral_rows = []
    redistribution_rows = []

    masks = region_masks(
        (256, 256),
        args.border,
    )

    for run in runs:
        split = run["split"]

        for p in run["files"]:
            sid = parse_sample_id(p)

            z = np.load(
                p,
                allow_pickle=True,
            )

            required = [
                "gt_norm",
                "c4_final_norm",
                "terminal_exact_norm",
                "terminal_gra_norm",
            ]

            missing = [
                k for k in required
                if k not in z.files
            ]

            if missing:
                raise RuntimeError(
                    f"{p}: missing keys {missing}"
                )

            gt = norm_to_speed(
                z["gt_norm"]
            )

            pred = {
                method:
                    norm_to_speed(
                        z[key]
                    )
                for method, key
                in METHOD_KEYS.items()
            }

            c4 = pred["C4-only"]

            c4_abs = np.abs(
                c4 - gt
            )

            c4_top10_thr = float(
                np.percentile(
                    c4_abs,
                    90,
                )
            )

            top10 = c4_abs >= c4_top10_thr
            bottom90 = ~top10

            # -----------------------------------------------------
            # Per-method metrics.
            # -----------------------------------------------------
            per_method = {}

            for method in METHOD_ORDER:
                x = pred[method]

                raw_mse = mse(x, gt)
                raw_mae = mae(x, gt)
                raw_ssim = ssim(x, gt)

                sm5 = smooth_metrics(
                    x,
                    gt,
                    5,
                )
                sm9 = smooth_metrics(
                    x,
                    gt,
                    9,
                )

                q = abs_error_quantiles(
                    x,
                    gt,
                )

                row = {
                    "split": split,
                    "sample": sid,
                    "method": method,
                    "raw_mse": raw_mse,
                    "raw_mae": raw_mae,
                    "raw_psnr":
                        psnr_from_mse(
                            raw_mse
                        ),
                    "raw_ssim": raw_ssim,
                    "smooth5_mse":
                        sm5["mse"],
                    "smooth5_mae":
                        sm5["mae"],
                    "smooth5_psnr":
                        sm5["psnr"],
                    "smooth5_ssim":
                        sm5["ssim"],
                    "smooth9_mse":
                        sm9["mse"],
                    "smooth9_mae":
                        sm9["mae"],
                    "smooth9_psnr":
                        sm9["psnr"],
                    "smooth9_ssim":
                        sm9["ssim"],
                    **q,
                }

                for region_name, mask in masks.items():
                    rm = region_metrics(
                        x,
                        gt,
                        mask,
                    )
                    row[
                        f"{region_name}_mse"
                    ] = rm["mse"]
                    row[
                        f"{region_name}_mae"
                    ] = rm["mae"]

                per_method[method] = row
                metric_rows.append(row)

            # -----------------------------------------------------
            # Paired deltas vs C4-only.
            # -----------------------------------------------------
            c4row = per_method["C4-only"]

            for method in [
                "C4+TExact",
                "C4+TGRA",
            ]:
                row = per_method[method]
                x = pred[method]

                fields_minimize = [
                    "raw_mse",
                    "raw_mae",
                    "smooth5_mse",
                    "smooth9_mse",
                    f"inner_crop{args.border}_mse",
                    f"border{args.border}_mse",
                    "p95",
                    "p99",
                    "max",
                ]

                for key in fields_minimize:
                    clean_key = (
                        key
                        .replace(
                            "raw_mse",
                            "mse"
                        )
                        .replace(
                            "raw_mae",
                            "mae"
                        )
                        .replace(
                            f"inner_crop{args.border}",
                            "inner32"
                        )
                        .replace(
                            f"border{args.border}",
                            "border32"
                        )
                    )

                    row[
                        f"delta_{clean_key}_vs_c4"
                    ] = (
                        row[key]
                        - c4row[key]
                    )

                row[
                    "delta_ssim_vs_c4"
                ] = (
                    row["raw_ssim"]
                    - c4row["raw_ssim"]
                )

                row[
                    "delta_smooth5_ssim_vs_c4"
                ] = (
                    row["smooth5_ssim"]
                    - c4row["smooth5_ssim"]
                )

                row[
                    "delta_smooth9_ssim_vs_c4"
                ] = (
                    row["smooth9_ssim"]
                    - c4row["smooth9_ssim"]
                )

                ae0 = np.abs(
                    c4 - gt
                )
                ae1 = np.abs(
                    x - gt
                )

                improved = ae1 < ae0
                worsened = ae1 > ae0

                top10_mse0 = float(
                    np.mean(
                        (c4[top10] - gt[top10]) ** 2
                    )
                )
                top10_mse1 = float(
                    np.mean(
                        (x[top10] - gt[top10]) ** 2
                    )
                )

                bottom90_mae0 = float(
                    np.mean(
                        np.abs(
                            c4[bottom90]
                            - gt[bottom90]
                        )
                    )
                )
                bottom90_mae1 = float(
                    np.mean(
                        np.abs(
                            x[bottom90]
                            - gt[bottom90]
                        )
                    )
                )

                row[
                    "improved_pixel_fraction"
                ] = float(
                    np.mean(improved)
                )

                row[
                    "worsened_pixel_fraction"
                ] = float(
                    np.mean(worsened)
                )

                row[
                    "unchanged_pixel_fraction"
                ] = float(
                    1.0
                    - np.mean(improved)
                    - np.mean(worsened)
                )

                row[
                    "top10_c4_error_mse_before"
                ] = top10_mse0
                row[
                    "top10_c4_error_mse_after"
                ] = top10_mse1
                row[
                    "top10_c4_error_mse_gain_frac"
                ] = (
                    top10_mse0
                    - top10_mse1
                ) / max(
                    top10_mse0,
                    1e-30,
                )

                row[
                    "bottom90_c4_error_mae_before"
                ] = bottom90_mae0
                row[
                    "bottom90_c4_error_mae_after"
                ] = bottom90_mae1
                row[
                    "bottom90_c4_error_mae_gain_frac"
                ] = (
                    bottom90_mae0
                    - bottom90_mae1
                ) / max(
                    bottom90_mae0,
                    1e-30,
                )

                # Copy paired-only fields back to the already-stored row.
                # metric_rows contains the same dict object.
                redistribution_rows.append({
                    "split": split,
                    "sample": sid,
                    "method": method,
                    "improved_pixel_fraction":
                        row[
                            "improved_pixel_fraction"
                        ],
                    "worsened_pixel_fraction":
                        row[
                            "worsened_pixel_fraction"
                        ],
                    "unchanged_pixel_fraction":
                        row[
                            "unchanged_pixel_fraction"
                        ],
                    "top10_c4_error_mse_before":
                        top10_mse0,
                    "top10_c4_error_mse_after":
                        top10_mse1,
                    "top10_c4_error_mse_gain_frac":
                        row[
                            "top10_c4_error_mse_gain_frac"
                        ],
                    "bottom90_c4_error_mae_before":
                        bottom90_mae0,
                    "bottom90_c4_error_mae_after":
                        bottom90_mae1,
                    "bottom90_c4_error_mae_gain_frac":
                        row[
                            "bottom90_c4_error_mae_gain_frac"
                        ],
                })

                delta = x - c4

                spec = update_spectrum(
                    delta
                )

                spectral_rows.append({
                    "split": split,
                    "sample": sid,
                    "method": method,
                    **spec,
                })

    # -----------------------------------------------------------------
    # Defensive paired-delta schema audit.
    # -----------------------------------------------------------------
    required_delta_keys = [
        "delta_mse_vs_c4",
        "delta_mae_vs_c4",
        "delta_ssim_vs_c4",
        "delta_smooth5_mse_vs_c4",
        "delta_smooth5_ssim_vs_c4",
        "delta_smooth9_mse_vs_c4",
        "delta_smooth9_ssim_vs_c4",
        "delta_inner32_mse_vs_c4",
        "delta_border32_mse_vs_c4",
        "delta_p95_vs_c4",
        "delta_p99_vs_c4",
        "delta_max_vs_c4",
    ]

    for row in metric_rows:
        if row["method"] not in {
            "C4+TExact",
            "C4+TGRA",
        }:
            continue

        missing = [
            k
            for k in required_delta_keys
            if k not in row
        ]

        if missing:
            raise RuntimeError(
                "Paired-delta schema audit failed for "
                f"{row['split']} sample={row['sample']} "
                f"method={row['method']}: missing={missing}"
            )

    # -----------------------------------------------------------------
    # Aggregate summaries.
    # -----------------------------------------------------------------
    groups = {}

    split_names = [
        r["split"]
        for r in runs
    ]

    for split in split_names:
        sub = [
            r for r in metric_rows
            if r["split"] == split
        ]
        groups[split] = analyze_group(
            sub,
            split,
        )

    groups["combined"] = analyze_group(
        metric_rows,
        "combined",
    )

    # -----------------------------------------------------------------
    # Absolute-error quantile summary rows.
    # -----------------------------------------------------------------
    quantile_summary_rows = []

    for group_name in [
        *split_names,
        "combined",
    ]:
        if group_name == "combined":
            sub = metric_rows
        else:
            sub = [
                r for r in metric_rows
                if r["split"] == group_name
            ]

        for method in METHOD_ORDER:
            ss = [
                r for r in sub
                if r["method"] == method
            ]

            for key in [
                "p50",
                "p75",
                "p90",
                "p95",
                "p99",
                "max",
            ]:
                vals = [
                    r[key]
                    for r in ss
                ]

                quantile_summary_rows.append({
                    "group": group_name,
                    "method": method,
                    "quantile": key,
                    "value_mean":
                        float(
                            np.mean(vals)
                        ),
                    "value_median":
                        float(
                            np.median(vals)
                        ),
                    "value_std":
                        float(
                            np.std(vals)
                        ),
                })

    # -----------------------------------------------------------------
    # Spectrum aggregate.
    # -----------------------------------------------------------------
    spectrum_summary = {}

    for group_name in [
        *split_names,
        "combined",
    ]:
        if group_name == "combined":
            sub = spectral_rows
        else:
            sub = [
                r for r in spectral_rows
                if r["split"] == group_name
            ]

        spectrum_summary[
            group_name
        ] = {}

        for method in [
            "C4+TExact",
            "C4+TGRA",
        ]:
            ss = [
                r for r in sub
                if r["method"] == method
            ]

            spectrum_summary[
                group_name
            ][method] = {
                "low_energy_frac": stat([
                    r["low_energy_frac"]
                    for r in ss
                ]),
                "mid_energy_frac": stat([
                    r["mid_energy_frac"]
                    for r in ss
                ]),
                "high_energy_frac": stat([
                    r["high_energy_frac"]
                    for r in ss
                ]),
                "update_rms": stat([
                    r["update_rms"]
                    for r in ss
                ]),
                "update_tv": stat([
                    r["update_tv"]
                    for r in ss
                ]),
            }

    # -----------------------------------------------------------------
    # Save tables.
    # -----------------------------------------------------------------
    write_csv(
        out / "per_sample_structural_metrics.csv",
        metric_rows,
    )

    write_csv(
        out / "error_redistribution.csv",
        redistribution_rows,
    )

    write_csv(
        out / "terminal_update_spectrum.csv",
        spectral_rows,
    )

    write_csv(
        out / "abs_error_quantile_summary.csv",
        quantile_summary_rows,
    )

    summary = {
        "experiment": (
            "C6.7 Structural Trade-off and Error Redistribution Audit"
        ),
        "source_runs": [
            {
                "split": r["split"],
                "run_dir": str(
                    r["run_dir"]
                ),
                "num_samples":
                    len(r["files"]),
            }
            for r in runs
        ],
        "protocol": {
            "pure_postprocess": True,
            "no_parameter_tuning": True,
            "speed_center": CENTER,
            "speed_scale": SCALE,
            "data_range": DATA_RANGE,
            "border_width_px":
                args.border,
            "smooth5": (
                "5x5 box filtering of both prediction and GT"
            ),
            "smooth9": (
                "9x9 box filtering of both prediction and GT"
            ),
            "frequency_bands": {
                "low": (
                    "normalized radial FFT radius <= 0.25"
                ),
                "mid": (
                    "0.25 < normalized radial FFT radius <= 0.50"
                ),
                "high": (
                    "normalized radial FFT radius > 0.50"
                ),
            },
            "top10_error_region": (
                "pixels above each sample's 90th percentile "
                "of C4-only absolute error"
            ),
            "bootstrap_seed":
                20260906,
            "bootstrap_replicates":
                10000,
        },
        "groups": groups,
        "spectrum": spectrum_summary,
    }

    summary_path = (
        out /
        "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Figures from combined set.
    # -----------------------------------------------------------------
    combined_paired = [
        r for r in metric_rows
        if r["method"] in {
            "C4+TExact",
            "C4+TGRA",
        }
    ]

    make_boxplot(
        out / "fig_delta_mse_vs_c4.png",
        combined_paired,
        "delta_mse_vs_c4",
        "Terminal correction: MSE change vs C4-only",
        r"$\Delta$ MSE (m$^2$/s$^2$), negative is better",
    )

    make_boxplot(
        out / "fig_delta_mae_vs_c4.png",
        combined_paired,
        "delta_mae_vs_c4",
        "Terminal correction: MAE change vs C4-only",
        r"$\Delta$ MAE (m/s), negative is better",
    )

    make_boxplot(
        out / "fig_delta_ssim_vs_c4.png",
        combined_paired,
        "delta_ssim_vs_c4",
        "Terminal correction: SSIM change vs C4-only",
        r"$\Delta$ SSIM, positive is better",
    )

    combined_quantile_rows = [
        r for r in quantile_summary_rows
        if r["group"] == "combined"
    ]

    make_quantile_plot(
        out / "fig_abs_error_quantiles.png",
        combined_quantile_rows,
    )

    combined_spectral = spectral_rows

    make_frequency_plot(
        out / "fig_terminal_update_frequency.png",
        combined_spectral,
    )

    # -------------------------------------------------------------
    # Console summary focused on TGRA.
    # -------------------------------------------------------------
    print("=" * 150)
    print("C6.7 SUMMARY")
    print("=" * 150)

    for group_name in [
        *split_names,
        "combined",
    ]:
        g = groups[group_name]
        tg = g[
            "paired_vs_c4"
        ]["C4+TGRA"]

        dmse = tg[
            "metric_deltas"
        ]["delta_mse_vs_c4"]

        dmae = tg[
            "metric_deltas"
        ]["delta_mae_vs_c4"]

        dssim = tg[
            "metric_deltas"
        ]["delta_ssim_vs_c4"]

        dp95 = tg[
            "metric_deltas"
        ]["delta_p95_vs_c4"]

        dp99 = tg[
            "metric_deltas"
        ]["delta_p99_vs_c4"]

        print()
        print(group_name)
        print("-" * 150)

        print(
            "TGRA ΔMSE mean/median = "
            f"{dmse['stat']['mean']:+.6f} / "
            f"{dmse['stat']['median']:+.6f}"
        )

        print(
            "TGRA ΔMAE mean/median = "
            f"{dmae['stat']['mean']:+.6f} / "
            f"{dmae['stat']['median']:+.6f}"
        )

        print(
            "TGRA ΔSSIM mean/median= "
            f"{dssim['stat']['mean']:+.6f} / "
            f"{dssim['stat']['median']:+.6f}"
        )

        print(
            "TGRA Δp95 |err| mean  = "
            f"{dp95['stat']['mean']:+.6f} m/s"
        )

        print(
            "TGRA Δp99 |err| mean  = "
            f"{dp99['stat']['mean']:+.6f} m/s"
        )

        print(
            "pixel improved/worsened = "
            f"{100*tg['improved_pixel_fraction']['mean']:.2f}% / "
            f"{100*tg['worsened_pixel_fraction']['mean']:.2f}%"
        )

        print(
            "top10 C4-error MSE gain = "
            f"{100*tg['top10_c4_error_mse_gain_frac']['mean']:+.3f}%"
        )

        print(
            "bottom90 C4-error MAE gain = "
            f"{100*tg['bottom90_c4_error_mae_gain_frac']['mean']:+.3f}%"
        )

        sp = spectrum_summary[
            group_name
        ]["C4+TGRA"]

        print(
            "TGRA update spectrum low/mid/high = "
            f"{sp['low_energy_frac']['mean']:.4f} / "
            f"{sp['mid_energy_frac']['mean']:.4f} / "
            f"{sp['high_energy_frac']['mean']:.4f}"
        )

    print()
    print("saved summary =", summary_path)
    print("saved figures =", out)


if __name__ == "__main__":
    main()
