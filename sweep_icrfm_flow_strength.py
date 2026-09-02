#!/usr/bin/env python3
"""Validate partial IC-RFM updates without retraining.

For a frozen IC-RFM checkpoint, compute the full deterministic flow endpoint
once and evaluate

    x_alpha = condition + alpha * (flow_endpoint - condition)

on a development split.  Alpha is selected only when mean MSE improves, mean
MAE does not worsen beyond tolerance, and the SSIM guard is satisfied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_icrfm_oof_v2 import (
    ConditionCacheDataset,
    bootstrap_paired,
    denormalize_speed,
    file_manifest,
    integrate_flow,
    json_dump,
    load_model,
    metric_row,
    write_csv,
)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--condition_root", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--eval_start", type=int, default=1)
    ap.add_argument("--eval_end", type=int, default=20)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--solver", choices=("euler", "heun"), default="heun")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mae_tolerance", type=float, default=0.0)
    ap.add_argument("--ssim_guard", type=float, default=0.001)
    ap.add_argument("--min_mse_gain", type=float, default=0.0)
    ap.add_argument("--bootstrap_samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260902)
    args = ap.parse_args()

    alphas = sorted(set(float(a) for a in args.alphas))
    if not alphas or alphas[0] < 0 or alphas[-1] > 1:
        raise ValueError("alphas must be within [0,1]")
    if 0.0 not in alphas:
        raise ValueError("alphas must include 0 for the identity baseline")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg, ckpt = load_model(args.ckpt_path, device)
    speed_min = float(cfg.get("speed_min", 1400.0))
    speed_max = float(cfg.get("speed_max", 1605.0))
    data_range = speed_max - speed_min

    dataset = ConditionCacheDataset(
        args.condition_root,
        args.eval_split,
        args.eval_start,
        args.eval_end,
        -1,
        False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            condition = batch["condition"].to(device, non_blocking=True)
            full_endpoint = integrate_flow(
                model, condition, args.steps, args.solver, noise_scale=0.0
            )
            condition_speed = batch["condition_speed"].numpy()
            target_speed = batch["target_speed"].numpy()
            ids = batch["sample_id"].numpy()

            for alpha in alphas:
                pred_norm = condition + alpha * (full_endpoint - condition)
                pred_norm = torch.clamp(pred_norm, -1.0, 1.0)
                pred_speed = denormalize_speed(
                    pred_norm, speed_min, speed_max
                ).cpu().numpy()
                for i, sid in enumerate(ids):
                    cm = metric_row(
                        condition_speed[i, 0], target_speed[i, 0], data_range
                    )
                    pm = metric_row(
                        pred_speed[i, 0], target_speed[i, 0], data_range
                    )
                    rows.append({
                        "sample_id": int(sid),
                        "alpha": float(alpha),
                        **{f"condition_{k}": v for k, v in cm.items()},
                        **{f"flow_{k}": v for k, v in pm.items()},
                        "mse_improvement": cm["mse"] - pm["mse"],
                        "mae_improvement": cm["mae"] - pm["mae"],
                        "psnr_improvement": pm["psnr"] - cm["psnr"],
                        "ssim_improvement": pm["ssim"] - cm["ssim"],
                    })

    summary_rows = []
    for alpha in alphas:
        part = [r for r in rows if r["alpha"] == alpha]
        row = {"alpha": alpha, "num_samples": len(part)}
        for prefix in ("condition", "flow"):
            for metric in ("mse", "mae", "psnr", "ssim"):
                values = np.asarray([r[f"{prefix}_{metric}"] for r in part])
                row[f"{prefix}_{metric}_mean"] = float(values.mean())
                row[f"{prefix}_{metric}_std"] = float(values.std(ddof=1))
        for metric in ("mse", "mae", "psnr", "ssim"):
            values = np.asarray([r[f"{metric}_improvement"] for r in part])
            row[f"{metric}_improvement_mean"] = float(values.mean())
            row[f"{metric}_wins"] = int(np.sum(values > 0))
        row["mse_guard_pass"] = bool(
            row["flow_mse_mean"]
            <= row["condition_mse_mean"] - args.min_mse_gain
        )
        row["mae_guard_pass"] = bool(
            row["flow_mae_mean"]
            <= row["condition_mae_mean"] + args.mae_tolerance
        )
        row["ssim_guard_pass"] = bool(
            row["flow_ssim_mean"]
            >= row["condition_ssim_mean"] - args.ssim_guard
        )
        row["all_guards_pass"] = bool(
            row["mse_guard_pass"]
            and row["mae_guard_pass"]
            and row["ssim_guard_pass"]
        )
        summary_rows.append(row)

    feasible = [
        r for r in summary_rows
        if r["alpha"] > 0 and r["all_guards_pass"]
    ]
    if feasible:
        selected = min(feasible, key=lambda r: r["flow_mse_mean"])
        status = "PASS"
        reason = "nonzero alpha satisfies MSE, MAE, and SSIM guards"
    else:
        selected = next(r for r in summary_rows if r["alpha"] == 0.0)
        status = "REJECT"
        reason = "no nonzero alpha satisfies all preregistered guards"

    selected_alpha = float(selected["alpha"])
    selected_rows = [r for r in rows if r["alpha"] == selected_alpha]
    paired = {}
    for metric in ("mse", "mae", "psnr", "ssim"):
        delta = np.asarray([r[f"{metric}_improvement"] for r in selected_rows])
        paired[metric] = bootstrap_paired(
            delta, args.seed + int(round(1000 * selected_alpha)),
            args.bootstrap_samples,
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "strength_summary.csv", summary_rows)
    write_csv(out / "strength_per_sample.csv", rows)
    json_dump(out / "paired_statistics_selected.json", paired)
    json_dump(out / "selection.json", {
        "status": status,
        "reason": reason,
        "selected_alpha": selected_alpha,
        "selected_summary": selected,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint": str(Path(args.ckpt_path).resolve()),
        "eval_files": file_manifest(dataset.files),
        "guards": {
            "mae_tolerance": args.mae_tolerance,
            "ssim_guard": args.ssim_guard,
            "min_mse_gain": args.min_mse_gain,
        },
        "config": vars(args),
    })

    print("=" * 118)
    print("IC-RFM strength sweep")
    print("checkpoint epoch =", int(ckpt.get("epoch", -1)))
    print("status           =", status)
    print("selected alpha   =", selected_alpha)
    print("=" * 118)
    print(
        f"{'alpha':>7} {'MSE':>12} {'MAE':>10} {'PSNR':>10} "
        f"{'SSIM':>10} {'MSE+':>6} {'MAE+':>6} {'SSIM+':>7} {'PASS':>6}"
    )
    for r in summary_rows:
        print(
            f"{r['alpha']:7.3f} {r['flow_mse_mean']:12.3f} "
            f"{r['flow_mae_mean']:10.4f} {r['flow_psnr_mean']:10.4f} "
            f"{r['flow_ssim_mean']:10.6f} {r['mse_wins']:6d} "
            f"{r['mae_wins']:6d} {r['ssim_wins']:7d} "
            f"{str(r['all_guards_pass']):>6}"
        )
    print("saved to:", out.resolve())


if __name__ == "__main__":
    main()
