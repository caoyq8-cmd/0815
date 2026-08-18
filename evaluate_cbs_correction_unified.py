import os
import re
import csv
import json
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


LOWER_IS_BETTER = [
    "mse_256", "mae_256", "rmse_256",
    "mse_480", "mae_480", "rmse_480",
    "dobs_abs_loss", "dobs_rel_l1", "dobs_rel_l2",
]
HIGHER_IS_BETTER = [
    "psnr_256", "ssim_256", "psnr_480", "ssim_480",
]


def numeric_key(path):
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


def image_metrics_np(pred, target, value_min, value_max):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    diff = pred - target
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    psnr = 99.0 if mse <= 1e-12 else float(
        20.0 * math.log10((value_max - value_min) / math.sqrt(mse))
    )
    ssim = compute_ssim_np(pred, target, value_min, value_max)
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": ssim,
    }


def gaussian_window(window_size=11, sigma=1.5, channels=1, device="cpu", dtype=torch.float32):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g).unsqueeze(0).unsqueeze(0)
    return window_2d.repeat(channels, 1, 1, 1)


def compute_ssim_np(pred, target, value_min=1400.0, value_max=1605.0, window_size=11, sigma=1.5):
    pred_t = torch.from_numpy(pred.astype(np.float32))[None, None]
    target_t = torch.from_numpy(target.astype(np.float32))[None, None]
    pred_t = ((pred_t - value_min) / (value_max - value_min)).clamp(0.0, 1.0)
    target_t = ((target_t - value_min) / (value_max - value_min)).clamp(0.0, 1.0)

    window = gaussian_window(window_size, sigma, 1, pred_t.device, pred_t.dtype)
    mu1 = F.conv2d(pred_t, window, padding=window_size // 2)
    mu2 = F.conv2d(target_t, window, padding=window_size // 2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(pred_t * pred_t, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(target_t * target_t, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(pred_t * target_t, window, padding=window_size // 2) - mu1_mu2
    sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0.0)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12
    value = float((numerator / denominator).mean().item())
    return max(min(value, 1.0), -1.0)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(d, key):
    if key not in d:
        return None
    try:
        v = float(d[key])
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def extract_history_metrics(item, suffix):
    """将 runner history 命名转换为统一 evaluator 命名。"""
    out = {}
    mapping = {
        f"image_mse_{suffix}": f"mse_{suffix}",
        f"image_mae_{suffix}": f"mae_{suffix}",
        f"image_rmse_{suffix}": f"rmse_{suffix}",
        f"image_psnr_{suffix}": f"psnr_{suffix}",
        f"image_ssim_{suffix}": f"ssim_{suffix}",
    }
    for old, new in mapping.items():
        v = safe_float(item, old)
        if v is not None:
            out[new] = v
    return out


def load_case(case_dir, value_min, value_max):
    case_dir = Path(case_dir)
    hist = read_json(case_dir / "history.json")
    if len(hist) == 0:
        raise ValueError(f"Empty history: {case_dir}")
    init_h = hist[0]
    final_h = hist[-1]

    init = {}
    final = {}
    source = {"image_metrics": "history", "ssim_available": False}

    # 优先从 final_result.npz 离线重算全部图像指标，这样旧结果无需重跑 CBS。
    result_path = case_dir / "final_result.npz"
    if result_path.exists():
        z = np.load(result_path, allow_pickle=False)
        required = [
            "condition_256_phys", "corrected_256_phys", "target_256_phys",
            "condition_480", "corrected_480", "target_480",
        ]
        if all(k in z.files for k in required):
            m = image_metrics_np(z["condition_256_phys"], z["target_256_phys"], value_min, value_max)
            init.update({f"{k}_256": v for k, v in m.items()})
            m = image_metrics_np(z["corrected_256_phys"], z["target_256_phys"], value_min, value_max)
            final.update({f"{k}_256": v for k, v in m.items()})

            m = image_metrics_np(z["condition_480"], z["target_480"], value_min, value_max)
            init.update({f"{k}_480": v for k, v in m.items()})
            m = image_metrics_np(z["corrected_480"], z["target_480"], value_min, value_max)
            final.update({f"{k}_480": v for k, v in m.items()})
            source["image_metrics"] = "final_result.npz"
            source["ssim_available"] = True

    # 缺失字段用 history 回填（新 runner 的 history 本身也含 SSIM）。
    for k, v in extract_history_metrics(init_h, "256").items():
        init.setdefault(k, v)
    for k, v in extract_history_metrics(final_h, "256").items():
        final.setdefault(k, v)
    for k, v in extract_history_metrics(init_h, "480").items():
        init.setdefault(k, v)
    for k, v in extract_history_metrics(final_h, "480").items():
        final.setdefault(k, v)

    if "ssim_256" in init and "ssim_256" in final:
        source["ssim_available"] = True

    for metric in ["dobs_abs_loss", "dobs_rel_l1", "dobs_rel_l2"]:
        vi = safe_float(init_h, metric)
        vf = safe_float(final_h, metric)
        if vi is not None:
            init[metric] = vi
        if vf is not None:
            final[metric] = vf

    row = {
        "sample": case_dir.name,
        "metric_source": source["image_metrics"],
        "ssim_available": source["ssim_available"],
        "final_iter": int(final_h.get("iter", len(hist) - 1)),
    }

    for k, v in init.items():
        row[f"init_{k}"] = float(v)
    for k, v in final.items():
        row[f"final_{k}"] = float(v)

    for metric in LOWER_IS_BETTER:
        ik, fk = f"init_{metric}", f"final_{metric}"
        if ik in row and fk in row:
            row[f"{metric}_relative_reduction"] = float(
                (row[ik] - row[fk]) / (abs(row[ik]) + 1e-12)
            )
            row[f"{metric}_improved"] = bool(row[fk] < row[ik])

    for metric in HIGHER_IS_BETTER:
        ik, fk = f"init_{metric}", f"final_{metric}"
        if ik in row and fk in row:
            row[f"{metric}_delta"] = float(row[fk] - row[ik])
            row[f"{metric}_improved"] = bool(row[fk] > row[ik])

    run_summary_path = case_dir / "run_summary.json"
    if run_summary_path.exists():
        rs = read_json(run_summary_path)
        for key in [
            "runtime_seconds",
            "candidate_selection_forward_evals",
            "post_update_eval_forward_evals",
            "total_forward_like_evals_excluding_adjoint_solve",
        ]:
            if key in rs:
                row[key] = rs[key]

    return row


def save_csv(rows, path):
    keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def metric_values(rows, metric, prefix):
    key = f"{prefix}_{metric}"
    values = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
    return np.asarray(values, dtype=np.float64)


def summarize(rows):
    summary = {
        "num_samples": len(rows),
        "num_cases_with_ssim": int(sum(bool(r.get("ssim_available", False)) for r in rows)),
        "metrics": {},
    }

    for metric in LOWER_IS_BETTER + HIGHER_IS_BETTER:
        paired = [r for r in rows if f"init_{metric}" in r and f"final_{metric}" in r]
        if not paired:
            continue
        init = np.asarray([r[f"init_{metric}"] for r in paired], dtype=np.float64)
        final = np.asarray([r[f"final_{metric}"] for r in paired], dtype=np.float64)

        entry = {
            "n": len(paired),
            "init_mean": float(init.mean()),
            "init_std": float(init.std()),
            "final_mean": float(final.mean()),
            "final_std": float(final.std()),
        }
        if metric in LOWER_IS_BETTER:
            reductions = (init - final) / (np.abs(init) + 1e-12)
            entry["mean_relative_reduction"] = float(reductions.mean())
            entry["relative_reduction_std"] = float(reductions.std())
            entry["num_improved"] = int(np.sum(final < init))
        else:
            deltas = final - init
            entry["mean_delta"] = float(deltas.mean())
            entry["delta_std"] = float(deltas.std())
            entry["num_improved"] = int(np.sum(final > init))
        summary["metrics"][metric] = entry

    runtime = [float(r["runtime_seconds"]) for r in rows if "runtime_seconds" in r]
    if runtime:
        arr = np.asarray(runtime, dtype=np.float64)
        summary["runtime_seconds_mean"] = float(arr.mean())
        summary["runtime_seconds_std"] = float(arr.std())

    return summary


def build_main_table(summary):
    rows = []
    for metric in LOWER_IS_BETTER + HIGHER_IS_BETTER:
        if metric not in summary["metrics"]:
            continue
        e = summary["metrics"][metric]
        row = {
            "metric": metric,
            "n": e["n"],
            "initial_mean": e["init_mean"],
            "initial_std": e["init_std"],
            "corrected_mean": e["final_mean"],
            "corrected_std": e["final_std"],
            "num_improved": e["num_improved"],
        }
        if metric in LOWER_IS_BETTER:
            row["mean_relative_reduction_pct"] = 100.0 * e["mean_relative_reduction"]
            row["change_type"] = "relative reduction (%)"
        else:
            row["mean_delta"] = e["mean_delta"]
            row["change_type"] = "absolute delta"
        rows.append(row)
    return rows


def save_scatter(rows, metric, out_path, title, xlabel="Initial", ylabel="Corrected"):
    paired = [r for r in rows if f"init_{metric}" in r and f"final_{metric}" in r]
    if len(paired) == 0:
        return
    x = np.asarray([r[f"init_{metric}"] for r in paired], dtype=np.float64)
    y = np.asarray([r[f"final_{metric}"] for r in paired], dtype=np.float64)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.03 * max(hi - lo, 1e-12)

    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(x, y, s=28, alpha=0.8)
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(lo - pad, hi + pad)
    plt.ylim(lo - pad, hi + pad)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_improvement_hist(rows, metric, out_path, title):
    key = f"{metric}_relative_reduction"
    vals = [100.0 * float(r[key]) for r in rows if key in r]
    if not vals:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=min(15, max(5, len(vals) // 3)))
    plt.axvline(float(np.mean(vals)), linestyle="--", label=f"mean={np.mean(vals):.2f}%")
    plt.xlabel("Relative reduction (%)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline unified evaluation for CBS correction outputs; no PDE rerun required."
    )
    parser.add_argument("--result_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--sample_prefix", type=str, default="")
    parser.add_argument("--speed_min", type=float, default=1400.0)
    parser.add_argument("--speed_max", type=float, default=1605.0)
    parser.add_argument("--expected_samples", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.result_root)
    if not root.exists():
        raise FileNotFoundError(root)

    output_dir = Path(args.output_dir) if args.output_dir else root / "unified_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    if (root / "history.json").exists():
        case_dirs = [root]
    else:
        case_dirs = [
            p for p in root.iterdir()
            if p.is_dir()
            and (p / "history.json").exists()
            and (not args.sample_prefix or p.name.startswith(args.sample_prefix))
        ]
        case_dirs = sorted(case_dirs, key=numeric_key)

    if len(case_dirs) == 0:
        raise FileNotFoundError(f"No direct child case directories with history.json under {root}")

    if args.expected_samples > 0 and len(case_dirs) != args.expected_samples:
        msg = f"Expected {args.expected_samples} samples, found {len(case_dirs)}"
        if args.strict:
            raise RuntimeError(msg)
        print("[WARN]", msg)

    rows = []
    failures = []
    for i, case_dir in enumerate(case_dirs, start=1):
        try:
            row = load_case(case_dir, args.speed_min, args.speed_max)
            rows.append(row)
            print(f"[{i}/{len(case_dirs)}] {case_dir.name}: source={row['metric_source']}")
        except Exception as e:
            failures.append({"sample": case_dir.name, "error": repr(e)})
            print(f"[ERROR] {case_dir.name}: {e}")
            if args.strict:
                raise

    if not rows:
        raise RuntimeError("No cases were successfully evaluated.")

    summary = summarize(rows)
    main_table = build_main_table(summary)

    with open(output_dir / "metrics_per_case.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    save_csv(rows, output_dir / "metrics_per_case.csv")

    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    save_csv(main_table, output_dir / "table_main.csv")

    if failures:
        with open(output_dir / "failures.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)

    if not args.no_plots:
        plots = {
            "mse_256": "MSE (256x256)",
            "mse_480": "MSE (480x480)",
            "dobs_abs_loss": "CBS measurement absolute error",
            "dobs_rel_l2": "CBS measurement relative L2",
            "psnr_256": "PSNR (256x256)",
            "ssim_256": "SSIM (256x256)",
        }
        for metric, title in plots.items():
            save_scatter(
                rows,
                metric,
                output_dir / f"scatter_{metric}.png",
                title=f"Before vs After: {title}",
            )
        for metric, title in [
            ("mse_256", "Per-sample reduction: MSE 256"),
            ("mse_480", "Per-sample reduction: MSE 480"),
            ("dobs_abs_loss", "Per-sample reduction: CBS measurement error"),
        ]:
            save_improvement_hist(rows, metric, output_dir / f"hist_reduction_{metric}.png", title)

    print("=" * 100)
    print("[Unified evaluation summary]")
    print("=" * 100)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved to:", output_dir.resolve())
    if summary["num_cases_with_ssim"] < summary["num_samples"]:
        print(
            "[NOTE] 部分旧实验没有 final_result.npz 且 history 中没有 SSIM，"
            "因此这些样本的 SSIM 无法离线恢复；其它指标仍可由 history 统计。"
        )


if __name__ == "__main__":
    main()
