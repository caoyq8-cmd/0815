#!/usr/bin/env python3
"""VAL20 screening of complementary IC-RFM and frozen-DDPM directions.

The two frozen generative endpoints are combined in normalized clean space:

    x_hybrid = clip(x_cond
                    + alpha_flow  * (x_flow  - x_cond)
                    + alpha_prior * (x_ddpm  - x_cond), -1, 1)

The OOF-trained IC-RFM direction is intended to reduce MSE, while a weak
condition-centered DDPM/SDEdit direction acts as an unconditional structure
prior.  All weights are selected only on VAL20 (test1..20).  DEV30 is guarded.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from screen_ddpm_sdedit_val20 import load_ddpm, safe_paired, sdedit_ddim
from train_icrfm_oof_v2 import (
    ConditionCacheDataset,
    denormalize_speed,
    file_manifest,
    integrate_flow,
    json_dump,
    load_model,
    metric_row,
    seed_everything,
    write_csv,
)


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def direction_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    af = a.flatten(1)
    bf = b.flatten(1)
    numerator = torch.sum(af * bf, dim=1)
    denominator = torch.linalg.vector_norm(af, dim=1) * torch.linalg.vector_norm(
        bf, dim=1
    )
    return numerator / torch.clamp(denominator, min=1e-12)


def rms_mps(x: torch.Tensor, scale_half_range: float) -> torch.Tensor:
    return torch.sqrt(torch.mean(x * x, dim=(1, 2, 3))) * scale_half_range


def same_setting(row: Dict, selected: Dict) -> bool:
    return (
        int(row["ddpm_t_start"]) == int(selected["ddpm_t_start"])
        and float(row["alpha_flow"]) == float(selected["alpha_flow"])
        and float(row["alpha_prior"]) == float(selected["alpha_prior"])
    )


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--condition_root", required=True)
    ap.add_argument("--flow_ckpt", required=True)
    ap.add_argument("--ddpm_ckpt", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--eval_start", type=int, default=1)
    ap.add_argument("--eval_end", type=int, default=20)
    ap.add_argument("--flow_steps", type=int, default=4)
    ap.add_argument("--flow_solver", choices=("euler", "heun"), default="heun")
    ap.add_argument("--ddpm_t_starts", type=int, nargs="+", default=[25, 50, 100])
    ap.add_argument("--ddpm_steps", type=int, default=5)
    ap.add_argument(
        "--flow_alphas",
        type=float,
        nargs="+",
        default=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1],
    )
    ap.add_argument(
        "--prior_alphas",
        type=float,
        nargs="+",
        default=[0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25],
    )
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min_mse_gain", type=float, default=0.05)
    ap.add_argument("--mae_tolerance", type=float, default=0.0)
    ap.add_argument("--ssim_guard", type=float, default=0.0)
    ap.add_argument("--min_mse_wins", type=int, default=10)
    ap.add_argument("--bootstrap_samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260902)
    args = ap.parse_args()

    t_starts = sorted(set(int(x) for x in args.ddpm_t_starts))
    flow_alphas = sorted(set(float(x) for x in args.flow_alphas))
    prior_alphas = sorted(set(float(x) for x in args.prior_alphas))
    for name, values in (
        ("flow_alphas", flow_alphas),
        ("prior_alphas", prior_alphas),
    ):
        if not values or values[0] < 0 or values[-1] > 1 or 0.0 not in values:
            raise ValueError(f"{name} must include 0 and lie in [0,1]")
    if args.eval_split != "test" or args.eval_start != 1 or args.eval_end != 20:
        raise RuntimeError(
            "G3 screening is frozen to VAL20=test1..20. DEV30 must remain untouched."
        )

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    flow_model, flow_cfg, flow_ckpt = load_model(args.flow_ckpt, device)
    ddpm_model, diffusion, ddpm_cfg, ddpm_ckpt = load_ddpm(args.ddpm_ckpt, device)

    speed_min = float(flow_cfg.get("speed_min", 1400.0))
    speed_max = float(flow_cfg.get("speed_max", 1605.0))
    ddpm_min = float(ddpm_cfg.get("norm_min", 1400.0))
    ddpm_max = float(ddpm_cfg.get("norm_max", 1605.0))
    if abs(speed_min - ddpm_min) > 1e-6 or abs(speed_max - ddpm_max) > 1e-6:
        raise RuntimeError(
            f"normalization mismatch: flow=[{speed_min},{speed_max}], "
            f"DDPM=[{ddpm_min},{ddpm_max}]"
        )
    if any(t < 1 or t >= diffusion.timesteps for t in t_starts):
        raise ValueError(f"DDPM t_start must lie in [1,{diffusion.timesteps - 1}]")
    data_range = speed_max - speed_min
    half_range = 0.5 * data_range

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

    rows: List[Dict] = []
    flow_model.eval()
    ddpm_model.eval()
    with torch.no_grad():
        for batch in loader:
            condition = batch["condition"].to(device, non_blocking=True)
            sample_ids = batch["sample_id"]
            ids = sample_ids.numpy()
            condition_speed = batch["condition_speed"].numpy()
            target_speed = batch["target_speed"].numpy()

            flow_endpoint = integrate_flow(
                flow_model,
                condition,
                args.flow_steps,
                args.flow_solver,
                noise_scale=0.0,
            )
            flow_delta = flow_endpoint - condition

            for t_start in t_starts:
                prior_endpoint, ddpm_nfe = sdedit_ddim(
                    ddpm_model,
                    diffusion,
                    condition,
                    sample_ids,
                    t_start,
                    args.ddpm_steps,
                    args.seed,
                )
                prior_delta = prior_endpoint - condition
                cosines = direction_cosine(flow_delta, prior_delta).cpu().numpy()
                flow_rms = rms_mps(flow_delta, half_range).cpu().numpy()
                prior_rms = rms_mps(prior_delta, half_range).cpu().numpy()

                for alpha_flow in flow_alphas:
                    for alpha_prior in prior_alphas:
                        prediction = torch.clamp(
                            condition
                            + alpha_flow * flow_delta
                            + alpha_prior * prior_delta,
                            -1.0,
                            1.0,
                        )
                        hybrid_delta = prediction - condition
                        hybrid_rms = rms_mps(
                            hybrid_delta, half_range
                        ).cpu().numpy()
                        pred_speed = denormalize_speed(
                            prediction, speed_min, speed_max
                        ).cpu().numpy()

                        for i, sid in enumerate(ids):
                            cm = metric_row(
                                condition_speed[i, 0], target_speed[i, 0], data_range
                            )
                            hm = metric_row(
                                pred_speed[i, 0], target_speed[i, 0], data_range
                            )
                            rows.append(
                                {
                                    "sample_id": int(sid),
                                    "ddpm_t_start": int(t_start),
                                    "ddpm_nfe": int(ddpm_nfe),
                                    "alpha_flow": float(alpha_flow),
                                    "alpha_prior": float(alpha_prior),
                                    "direction_cosine": float(cosines[i]),
                                    "flow_endpoint_rms_mps": float(flow_rms[i]),
                                    "prior_endpoint_rms_mps": float(prior_rms[i]),
                                    "hybrid_update_rms_mps": float(hybrid_rms[i]),
                                    **{f"condition_{k}": v for k, v in cm.items()},
                                    **{f"hybrid_{k}": v for k, v in hm.items()},
                                    "mse_improvement": cm["mse"] - hm["mse"],
                                    "mae_improvement": cm["mae"] - hm["mae"],
                                    "psnr_improvement": hm["psnr"] - cm["psnr"],
                                    "ssim_improvement": hm["ssim"] - cm["ssim"],
                                }
                            )

    summary_rows: List[Dict] = []
    for t_start in t_starts:
        for alpha_flow in flow_alphas:
            for alpha_prior in prior_alphas:
                part = [
                    r
                    for r in rows
                    if r["ddpm_t_start"] == t_start
                    and r["alpha_flow"] == alpha_flow
                    and r["alpha_prior"] == alpha_prior
                ]
                summary: Dict = {
                    "ddpm_t_start": t_start,
                    "ddpm_nfe": int(part[0]["ddpm_nfe"]),
                    "alpha_flow": alpha_flow,
                    "alpha_prior": alpha_prior,
                    "num_samples": len(part),
                }
                for prefix in ("condition", "hybrid"):
                    for metric in ("mse", "mae", "psnr", "ssim"):
                        mean, std = mean_std([r[f"{prefix}_{metric}"] for r in part])
                        summary[f"{prefix}_{metric}_mean"] = mean
                        summary[f"{prefix}_{metric}_std"] = std
                for metric in ("mse", "mae", "psnr", "ssim"):
                    values = np.asarray(
                        [r[f"{metric}_improvement"] for r in part], np.float64
                    )
                    summary[f"{metric}_improvement_mean"] = float(values.mean())
                    summary[f"{metric}_wins"] = int(np.sum(values > 0))
                for metric in (
                    "direction_cosine",
                    "flow_endpoint_rms_mps",
                    "prior_endpoint_rms_mps",
                    "hybrid_update_rms_mps",
                ):
                    summary[f"{metric}_mean"] = float(
                        np.mean([r[metric] for r in part])
                    )
                summary["mse_guard_pass"] = bool(
                    summary["mse_improvement_mean"] >= args.min_mse_gain
                    and summary["mse_wins"] >= args.min_mse_wins
                )
                summary["mae_guard_pass"] = bool(
                    summary["hybrid_mae_mean"]
                    <= summary["condition_mae_mean"] + args.mae_tolerance
                )
                summary["ssim_guard_pass"] = bool(
                    summary["hybrid_ssim_mean"]
                    >= summary["condition_ssim_mean"] - args.ssim_guard
                )
                summary["all_guards_pass"] = bool(
                    summary["mse_guard_pass"]
                    and summary["mae_guard_pass"]
                    and summary["ssim_guard_pass"]
                )
                summary_rows.append(summary)

    nonidentity = [
        r
        for r in summary_rows
        if r["alpha_flow"] > 0 or r["alpha_prior"] > 0
    ]
    feasible = [r for r in nonidentity if r["all_guards_pass"]]
    if feasible:
        selected = min(
            feasible,
            key=lambda r: (
                r["hybrid_mse_mean"],
                -r["hybrid_ssim_mean"],
                r["hybrid_mae_mean"],
                r["ddpm_t_start"],
                r["alpha_flow"] + r["alpha_prior"],
            ),
        )
        status = "PASS"
        reason = "nonidentity fusion satisfies MSE, MAE, SSIM, and MSE-win guards"
    else:
        selected = next(
            r
            for r in summary_rows
            if r["ddpm_t_start"] == t_starts[0]
            and r["alpha_flow"] == 0.0
            and r["alpha_prior"] == 0.0
        )
        status = "REJECT"
        reason = "no fusion satisfies all preregistered G3 guards"

    selected_rows = [r for r in rows if same_setting(r, selected)]
    paired = {}
    for offset, metric in enumerate(("mse", "mae", "psnr", "ssim")):
        delta = np.asarray(
            [r[f"{metric}_improvement"] for r in selected_rows], np.float64
        )
        paired[metric] = safe_paired(
            delta, args.seed + 200000 + offset, args.bootstrap_samples
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "fusion_summary.csv", summary_rows)
    write_csv(out / "fusion_per_sample.csv", rows)
    json_dump(out / "paired_statistics_selected.json", paired)
    json_dump(
        out / "selection.json",
        {
            "status": status,
            "reason": reason,
            "selected_setting": selected,
            "flow_checkpoint_epoch": int(flow_ckpt.get("epoch", -1)),
            "ddpm_checkpoint_epoch": int(ddpm_ckpt.get("epoch", -1)),
            "flow_checkpoint": str(Path(args.flow_ckpt).resolve()),
            "ddpm_checkpoint": str(Path(args.ddpm_ckpt).resolve()),
            "eval_files": file_manifest(dataset.files),
            "guards": {
                "min_mse_gain": args.min_mse_gain,
                "min_mse_wins": args.min_mse_wins,
                "mae_tolerance": args.mae_tolerance,
                "ssim_guard": args.ssim_guard,
            },
            "config": vars(args),
        },
    )

    ranked = sorted(nonidentity, key=lambda r: r["hybrid_mse_mean"])
    print("=" * 142)
    print("G3 complementary generative fusion: OOF IC-RFM + frozen DDPM prior")
    print("status             =", status)
    print(
        "selected           =",
        f"t={selected['ddpm_t_start']} "
        f"a_flow={selected['alpha_flow']} "
        f"a_prior={selected['alpha_prior']}",
    )
    print("passing settings   =", len(feasible), "/", len(nonidentity))
    print("=" * 142)
    print("Top 20 nonidentity settings by MSE (PASS is the complete guard):")
    print(
        f"{'t':>4} {'aF':>6} {'aP':>6} {'MSE':>11} {'MAE':>9} {'SSIM':>10} "
        f"{'dMSE':>9} {'dMAE':>9} {'dSSIM':>10} {'MSE+':>5} {'PASS':>6}"
    )
    for r in ranked[:20]:
        print(
            f"{r['ddpm_t_start']:4d} {r['alpha_flow']:6.3f} "
            f"{r['alpha_prior']:6.3f} {r['hybrid_mse_mean']:11.3f} "
            f"{r['hybrid_mae_mean']:9.4f} {r['hybrid_ssim_mean']:10.6f} "
            f"{r['mse_improvement_mean']:9.4f} "
            f"{r['mae_improvement_mean']:9.5f} "
            f"{r['ssim_improvement_mean']:10.6f} "
            f"{r['mse_wins']:5d} {str(r['all_guards_pass']):>6}"
        )
    print("saved to:", out.resolve())


if __name__ == "__main__":
    main()
