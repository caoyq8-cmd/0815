#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline thesis metrics for saved Stable-CBS correction results.

Expected input: a result root that recursively contains per-sample
``final_result.npz`` files produced by ``run_cbs_physics_correction.py``.
The script does NOT rerun CBS. It recomputes image metrics from saved arrays
and reads physical residual metrics from ``history.json`` / embedded history.

Outputs:
  per_sample_metrics.csv
  summary.json
  summary_table.tex
  scatter_*.png
  convergence_*.png (when histories are available)

Recommended thesis convention:
  PSNR/SSIM data_range = speed_max - speed_min = 205 m/s.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from skimage.metrics import structural_similarity
except Exception as exc:  # pragma: no cover
    structural_similarity = None
    _SKIMAGE_IMPORT_ERROR = exc

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

import matplotlib.pyplot as plt


def mse(pred, target):
    d = pred.astype(np.float64) - target.astype(np.float64)
    return float(np.mean(d * d))


def mae(pred, target):
    return float(np.mean(np.abs(pred.astype(np.float64) - target.astype(np.float64))))


def rmse(pred, target):
    return float(math.sqrt(mse(pred, target)))


def psnr(pred, target, data_range):
    v = mse(pred, target)
    if v <= 1e-15:
        return 99.0
    return float(20.0 * math.log10(float(data_range) / math.sqrt(v)))


def ssim(pred, target, data_range):
    if structural_similarity is None:
        raise RuntimeError(
            "SSIM requires scikit-image. Install with: pip install scikit-image\n"
            f"Original import error: {_SKIMAGE_IMPORT_ERROR}"
        )
    return float(
        structural_similarity(
            target.astype(np.float64),
            pred.astype(np.float64),
            data_range=float(data_range),
        )
    )


def _to_builtin(x):
    if isinstance(x, np.generic):
        return x.item()
    return x


def load_history(npz_path: Path) -> Optional[List[dict]]:
    json_path = npz_path.parent / "history.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    with np.load(npz_path, allow_pickle=True) as z:
        if "history" not in z.files:
            return None
        h = z["history"]
        rows = []
        for item in h.tolist():
            if isinstance(item, dict):
                rows.append({k: _to_builtin(v) for k, v in item.items()})
        return rows or None


def extract_arrays(npz_path: Path):
    with np.load(npz_path, allow_pickle=True) as z:
        required = [
            "condition_256_phys", "corrected_256_phys", "target_256_phys",
            "condition_480", "corrected_480", "target_480",
        ]
        missing = [k for k in required if k not in z.files]
        if missing:
            raise KeyError(f"{npz_path}: missing keys {missing}; available={z.files}")
        return {k: z[k].astype(np.float32) for k in required}


def image_metric_block(prefix: str, pred, target, data_range: float) -> Dict[str, float]:
    return {
        f"{prefix}_mse": mse(pred, target),
        f"{prefix}_mae": mae(pred, target),
        f"{prefix}_rmse": rmse(pred, target),
        f"{prefix}_psnr": psnr(pred, target, data_range),
        f"{prefix}_ssim": ssim(pred, target, data_range),
    }


def parse_sample_id(path: Path, root: Path) -> str:
    try:
        rel = path.parent.relative_to(root)
        return str(rel).replace("/", "__") or path.parent.name
    except Exception:
        return path.parent.name


def safe_improve(init, final, higher_better=False):
    if higher_better:
        return float((final - init) / (abs(init) + 1e-12))
    return float((init - final) / (abs(init) + 1e-12))


def wilcoxon_p(a, b):
    if wilcoxon is None or len(a) < 2:
        return None
    try:
        return float(wilcoxon(np.asarray(a), np.asarray(b), zero_method="wilcox").pvalue)
    except ValueError:
        return None


def summarize(rows: List[dict]) -> dict:
    keys = [k for k in rows[0] if k != "sample_id" and isinstance(rows[0][k], (int, float, np.number))]
    out = {"num_samples": len(rows)}
    for key in keys:
        vals = np.array([float(r[key]) for r in rows if r.get(key) is not None], dtype=np.float64)
        if vals.size:
            out[f"{key}_mean"] = float(vals.mean())
            out[f"{key}_std"] = float(vals.std(ddof=0))

    # Paired significance tests: lower is better except PSNR/SSIM.
    pairs = [
        ("init256_mse", "final256_mse", "mse256"),
        ("init256_mae", "final256_mae", "mae256"),
        ("init256_rmse", "final256_rmse", "rmse256"),
        ("init256_psnr", "final256_psnr", "psnr256"),
        ("init256_ssim", "final256_ssim", "ssim256"),
        ("init480_mse", "final480_mse", "mse480"),
        ("init480_mae", "final480_mae", "mae480"),
        ("init480_rmse", "final480_rmse", "rmse480"),
        ("init480_psnr", "final480_psnr", "psnr480"),
        ("init480_ssim", "final480_ssim", "ssim480"),
        ("init_dobs_rel_l1", "final_dobs_rel_l1", "dobs_rel_l1"),
        ("init_dobs_rel_l2", "final_dobs_rel_l2", "dobs_rel_l2"),
        ("init_dobs_abs_loss", "final_dobs_abs_loss", "dobs_abs_loss"),
    ]
    out["paired_wilcoxon"] = {}
    for a_key, b_key, name in pairs:
        paired = [(r.get(a_key), r.get(b_key)) for r in rows]
        paired = [(a, b) for a, b in paired if a is not None and b is not None]
        if len(paired) >= 2:
            a = [x[0] for x in paired]
            b = [x[1] for x in paired]
            out["paired_wilcoxon"][name] = {"n": len(paired), "p_value": wilcoxon_p(a, b)}
    return out


def write_csv(rows, path: Path):
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def mean_std(rows, key):
    vals = np.array([r[key] for r in rows if r.get(key) is not None], dtype=float)
    return float(vals.mean()), float(vals.std(ddof=0))


def write_latex_table(rows, path: Path):
    metrics = [
        ("MSE", "init256_mse", "final256_mse", False),
        ("MAE (m/s)", "init256_mae", "final256_mae", False),
        ("RMSE (m/s)", "init256_rmse", "final256_rmse", False),
        ("PSNR (dB)", "init256_psnr", "final256_psnr", True),
        ("SSIM", "init256_ssim", "final256_ssim", True),
        ("Physics Rel-$L_1$", "init_dobs_rel_l1", "final_dobs_rel_l1", False),
        ("Physics Rel-$L_2$", "init_dobs_rel_l2", "final_dobs_rel_l2", False),
        ("Physics Abs.", "init_dobs_abs_loss", "final_dobs_abs_loss", False),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{同一self-consistent测试集上的统一评价指标（脚本自动生成）}",
        r"\label{tab:unified_metrics_auto}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"指标 & InversionNet初值 & Stable CBS Correction \\",
        r"\midrule",
    ]
    for name, a, b, higher in metrics:
        av = [r[a] for r in rows if r.get(a) is not None]
        bv = [r[b] for r in rows if r.get(b) is not None]
        if not av or not bv:
            continue
        am, ast = float(np.mean(av)), float(np.std(av))
        bm, bst = float(np.mean(bv)), float(np.std(bv))
        fmt = ".4f" if max(abs(am), abs(bm)) >= 1e-3 else ".6g"
        lines.append(f"{name} & ${am:{fmt}}\\pm{ast:{fmt}}$ & ${bm:{fmt}}\\pm{bst:{fmt}}$ \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scatter(rows, xkey, ykey, xlabel, ylabel, out_path: Path):
    x = np.asarray([r[xkey] for r in rows if r.get(xkey) is not None and r.get(ykey) is not None], dtype=float)
    y = np.asarray([r[ykey] for r in rows if r.get(xkey) is not None and r.get(ykey) is not None], dtype=float)
    if not len(x):
        return
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.03 * max(hi - lo, 1e-12)
    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(x, y, s=24)
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def convergence_plot(histories, key, ylabel, out_path: Path):
    valid = [h for h in histories if h and all(key in row for row in h)]
    if not valid:
        return
    n = min(len(h) for h in valid)
    arr = np.asarray([[float(row[key]) for row in h[:n]] for h in valid], dtype=float)
    x = np.arange(n)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    plt.figure(figsize=(6.0, 4.2))
    plt.plot(x, mean, marker="o")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Physics correction iteration")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_root", required=True, help="Root containing per-sample final_result.npz")
    ap.add_argument("--output_dir", default="./thesis_eval_saved_cbs")
    ap.add_argument("--speed_min", type=float, default=1400.0)
    ap.add_argument("--speed_max", type=float, default=1605.0)
    ap.add_argument("--max_samples", type=int, default=-1)
    args = ap.parse_args()

    root = Path(args.result_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_range = args.speed_max - args.speed_min

    files = sorted(root.rglob("final_result.npz"))
    if args.max_samples > 0:
        files = files[: args.max_samples]
    if not files:
        raise RuntimeError(f"No final_result.npz found under {root}")

    rows = []
    histories = []
    for idx, p in enumerate(files, 1):
        arr = extract_arrays(p)
        h = load_history(p)
        histories.append(h)
        row = {"sample_id": parse_sample_id(p, root)}
        row.update(image_metric_block("init256", arr["condition_256_phys"], arr["target_256_phys"], data_range))
        row.update(image_metric_block("final256", arr["corrected_256_phys"], arr["target_256_phys"], data_range))
        row.update(image_metric_block("init480", arr["condition_480"], arr["target_480"], data_range))
        row.update(image_metric_block("final480", arr["corrected_480"], arr["target_480"], data_range))

        row["mse256_improve_ratio"] = safe_improve(row["init256_mse"], row["final256_mse"])
        row["mae256_improve_ratio"] = safe_improve(row["init256_mae"], row["final256_mae"])
        row["psnr256_relative_gain"] = safe_improve(row["init256_psnr"], row["final256_psnr"], higher_better=True)
        row["ssim256_relative_gain"] = safe_improve(row["init256_ssim"], row["final256_ssim"], higher_better=True)

        for key in ["dobs_rel_l1", "dobs_rel_l2", "dobs_abs_loss"]:
            row[f"init_{key}"] = None
            row[f"final_{key}"] = None
        if h:
            first, last = h[0], h[-1]
            for key in ["dobs_rel_l1", "dobs_rel_l2", "dobs_abs_loss"]:
                if key in first and key in last:
                    row[f"init_{key}"] = float(first[key])
                    row[f"final_{key}"] = float(last[key])
            if row["init_dobs_abs_loss"] is not None:
                row["dobs_abs_improve_ratio"] = safe_improve(row["init_dobs_abs_loss"], row["final_dobs_abs_loss"])
            else:
                row["dobs_abs_improve_ratio"] = None
        else:
            row["dobs_abs_improve_ratio"] = None
        rows.append(row)
        print(f"[{idx:03d}/{len(files):03d}] {row['sample_id']}")

    write_csv(rows, out / "per_sample_metrics.csv")
    summary = summarize(rows)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_latex_table(rows, out / "summary_table.tex")

    scatter(rows, "init256_mse", "final256_mse", "Initial MSE (256)", "Final MSE (256)", out / "scatter_mse256.png")
    scatter(rows, "init480_mse", "final480_mse", "Initial MSE (480)", "Final MSE (480)", out / "scatter_mse480.png")
    scatter(rows, "init256_psnr", "final256_psnr", "Initial PSNR (dB)", "Final PSNR (dB)", out / "scatter_psnr256.png")
    scatter(rows, "init256_ssim", "final256_ssim", "Initial SSIM", "Final SSIM", out / "scatter_ssim256.png")
    scatter(rows, "init_dobs_rel_l2", "final_dobs_rel_l2", "Initial Physics Rel-L2", "Final Physics Rel-L2", out / "scatter_physics_rel_l2.png")

    convergence_plot(histories, "image_mse_256", "MSE (256)", out / "convergence_mse256.png")
    convergence_plot(histories, "image_mse_480", "MSE (480)", out / "convergence_mse480.png")
    convergence_plot(histories, "dobs_abs_loss", "CBS measurement absolute error", out / "convergence_dobs_abs.png")
    convergence_plot(histories, "dobs_rel_l2", "CBS measurement Rel-L2", out / "convergence_dobs_rel_l2.png")

    print("Done. Outputs:", out.resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
