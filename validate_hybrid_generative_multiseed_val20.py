#!/usr/bin/env python3
"""Pre-registered multi-seed confirmation for the frozen G3 fusion setting.

No hyperparameter selection is performed here.  The setting discovered on the
selection seed is fixed by default to

    DDPM t_start=25, NFE=5, alpha_flow=0.03, alpha_prior=0.15.

The first seed is the original selection seed; all remaining seeds are unseen
confirmation seeds.  The script reports both single-noise stability and a
pre-registered mean-DDPM-endpoint ensemble.  DEV30 is explicitly guarded.
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


def aggregate(rows: Sequence[Dict], args: argparse.Namespace) -> Dict:
    result: Dict = {"num_samples": len(rows)}
    for prefix in ("condition", "hybrid"):
        for metric in ("mse", "mae", "psnr", "ssim"):
            mean, std = mean_std([r[f"{prefix}_{metric}"] for r in rows])
            result[f"{prefix}_{metric}_mean"] = mean
            result[f"{prefix}_{metric}_std"] = std
    for metric in ("mse", "mae", "psnr", "ssim"):
        values = np.asarray(
            [r[f"{metric}_improvement"] for r in rows], dtype=np.float64
        )
        result[f"{metric}_improvement_mean"] = float(values.mean())
        result[f"{metric}_wins"] = int(np.sum(values > 0))
        result[f"{metric}_losses"] = int(np.sum(values < 0))
    result["hybrid_update_rms_mps_mean"] = float(
        np.mean([r["hybrid_update_rms_mps"] for r in rows])
    )
    result["mse_guard_pass"] = bool(
        result["mse_improvement_mean"] >= args.min_mse_gain
        and result["mse_wins"] >= args.min_mse_wins
    )
    result["mae_guard_pass"] = bool(
        result["hybrid_mae_mean"]
        <= result["condition_mae_mean"] + args.mae_tolerance
    )
    result["ssim_guard_pass"] = bool(
        result["hybrid_ssim_mean"]
        >= result["condition_ssim_mean"] - args.ssim_guard
    )
    result["all_guards_pass"] = bool(
        result["mse_guard_pass"]
        and result["mae_guard_pass"]
        and result["ssim_guard_pass"]
    )
    return result


def paired_statistics(
    rows: Sequence[Dict], base_seed: int, n_boot: int
) -> Dict:
    output = {}
    for offset, metric in enumerate(("mse", "mae", "psnr", "ssim")):
        delta = np.asarray(
            [r[f"{metric}_improvement"] for r in rows], dtype=np.float64
        )
        output[metric] = safe_paired(delta, base_seed + offset, n_boot)
    return output


def metric_record(
    sample_id: int,
    condition_speed: np.ndarray,
    target_speed: np.ndarray,
    prediction_speed: np.ndarray,
    update_rms_mps: float,
    data_range: float,
    seed: int | str,
) -> Dict:
    cm = metric_row(condition_speed, target_speed, data_range)
    hm = metric_row(prediction_speed, target_speed, data_range)
    return {
        "sample_id": int(sample_id),
        "seed": seed,
        "hybrid_update_rms_mps": float(update_rms_mps),
        **{f"condition_{k}": v for k, v in cm.items()},
        **{f"hybrid_{k}": v for k, v in hm.items()},
        "mse_improvement": cm["mse"] - hm["mse"],
        "mae_improvement": cm["mae"] - hm["mae"],
        "psnr_improvement": hm["psnr"] - cm["psnr"],
        "ssim_improvement": hm["ssim"] - cm["ssim"],
    }


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
    ap.add_argument("--ddpm_t_start", type=int, default=25)
    ap.add_argument("--ddpm_steps", type=int, default=5)
    ap.add_argument("--alpha_flow", type=float, default=0.03)
    ap.add_argument("--alpha_prior", type=float, default=0.15)
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260902, 20260903, 20260904, 20260905, 20260906],
        help="first is the selection seed; the rest are confirmation seeds",
    )
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min_mse_gain", type=float, default=0.05)
    ap.add_argument("--min_mse_wins", type=int, default=10)
    ap.add_argument("--mae_tolerance", type=float, default=0.0)
    ap.add_argument("--ssim_guard", type=float, default=0.0)
    ap.add_argument("--bootstrap_samples", type=int, default=10000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    seeds = list(dict.fromkeys(int(x) for x in args.seeds))
    if len(seeds) < 3:
        raise ValueError("provide at least three distinct seeds")
    if args.eval_split != "test" or args.eval_start != 1 or args.eval_end != 20:
        raise RuntimeError(
            "Multi-seed confirmation is frozen to VAL20=test1..20; DEV30 is guarded."
        )
    if not 0 <= args.alpha_flow <= 1 or not 0 <= args.alpha_prior <= 1:
        raise ValueError("alpha_flow and alpha_prior must lie in [0,1]")

    out = Path(args.output_dir)
    if (out / "validation.json").exists() and not args.overwrite:
        raise RuntimeError(
            f"completed output already exists: {out}. Use a new output_dir or --overwrite."
        )
    prediction_dir = out / "ensemble_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seeds[0])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    flow_model, flow_cfg, flow_ckpt = load_model(args.flow_ckpt, device)
    ddpm_model, diffusion, ddpm_cfg, ddpm_ckpt = load_ddpm(args.ddpm_ckpt, device)
    speed_min = float(flow_cfg.get("speed_min", 1400.0))
    speed_max = float(flow_cfg.get("speed_max", 1605.0))
    ddpm_min = float(ddpm_cfg.get("norm_min", 1400.0))
    ddpm_max = float(ddpm_cfg.get("norm_max", 1605.0))
    if abs(speed_min - ddpm_min) > 1e-6 or abs(speed_max - ddpm_max) > 1e-6:
        raise RuntimeError("flow and DDPM normalization ranges do not match")
    if not 1 <= args.ddpm_t_start < diffusion.timesteps:
        raise ValueError("invalid ddpm_t_start")
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

    per_seed_rows: List[Dict] = []
    ensemble_rows: List[Dict] = []
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
            prior_endpoints = []

            for noise_seed in seeds:
                prior_endpoint, _ = sdedit_ddim(
                    ddpm_model,
                    diffusion,
                    condition,
                    sample_ids,
                    args.ddpm_t_start,
                    args.ddpm_steps,
                    noise_seed,
                )
                prior_endpoints.append(prior_endpoint)
                prediction = torch.clamp(
                    condition
                    + args.alpha_flow * flow_delta
                    + args.alpha_prior * (prior_endpoint - condition),
                    -1.0,
                    1.0,
                )
                prediction_speed = denormalize_speed(
                    prediction, speed_min, speed_max
                ).cpu().numpy()
                update_rms = (
                    torch.sqrt(torch.mean((prediction - condition) ** 2, dim=(1, 2, 3)))
                    * half_range
                ).cpu().numpy()
                for i, sid in enumerate(ids):
                    per_seed_rows.append(
                        metric_record(
                            int(sid),
                            condition_speed[i, 0],
                            target_speed[i, 0],
                            prediction_speed[i, 0],
                            float(update_rms[i]),
                            data_range,
                            int(noise_seed),
                        )
                    )

            prior_mean = torch.stack(prior_endpoints, dim=0).mean(dim=0)
            ensemble_prediction = torch.clamp(
                condition
                + args.alpha_flow * flow_delta
                + args.alpha_prior * (prior_mean - condition),
                -1.0,
                1.0,
            )
            ensemble_speed = denormalize_speed(
                ensemble_prediction, speed_min, speed_max
            ).cpu().numpy()
            ensemble_rms = (
                torch.sqrt(
                    torch.mean((ensemble_prediction - condition) ** 2, dim=(1, 2, 3))
                )
                * half_range
            ).cpu().numpy()

            for i, sid in enumerate(ids):
                ensemble_rows.append(
                    metric_record(
                        int(sid),
                        condition_speed[i, 0],
                        target_speed[i, 0],
                        ensemble_speed[i, 0],
                        float(ensemble_rms[i]),
                        data_range,
                        "ensemble",
                    )
                )
                np.savez_compressed(
                    prediction_dir / f"test_{int(sid)}.npz",
                    sample_id=np.int64(sid),
                    seeds=np.asarray(seeds, dtype=np.int64),
                    condition_norm=condition[i].cpu().numpy().astype(np.float32),
                    target_speed=target_speed[i].astype(np.float32),
                    flow_endpoint_norm=flow_endpoint[i].cpu().numpy().astype(np.float32),
                    ddpm_endpoint_mean_norm=prior_mean[i].cpu().numpy().astype(np.float32),
                    hybrid_norm=ensemble_prediction[i].cpu().numpy().astype(np.float32),
                    hybrid_speed=ensemble_speed[i].astype(np.float32),
                )

    per_seed_summary = []
    per_seed_paired = {}
    for index, noise_seed in enumerate(seeds):
        part = [r for r in per_seed_rows if r["seed"] == noise_seed]
        summary = {"seed": noise_seed, "role": "selection" if index == 0 else "confirmation"}
        summary.update(aggregate(part, args))
        per_seed_summary.append(summary)
        per_seed_paired[str(noise_seed)] = paired_statistics(
            part, noise_seed + 300000, args.bootstrap_samples
        )

    ensemble_summary = aggregate(ensemble_rows, args)
    ensemble_paired = paired_statistics(
        ensemble_rows, seeds[0] + 400000, args.bootstrap_samples
    )
    confirmation = [r for r in per_seed_summary if r["role"] == "confirmation"]
    confirmation_pass_count = sum(r["all_guards_pass"] for r in confirmation)
    single_seed_stable = confirmation_pass_count == len(confirmation)
    ensemble_pass = bool(ensemble_summary["all_guards_pass"])
    if single_seed_stable:
        status = "STABLE_PASS"
        reason = "every unseen confirmation seed satisfies the frozen guards"
    elif ensemble_pass:
        status = "ENSEMBLE_PASS_ONLY"
        reason = "single-seed behavior is unstable, but the preregistered seed ensemble passes"
    else:
        status = "REJECT"
        reason = "the frozen setting does not survive multi-seed confirmation"

    write_csv(out / "per_seed_summary.csv", per_seed_summary)
    write_csv(out / "per_sample_seed.csv", per_seed_rows)
    write_csv(out / "ensemble_per_sample.csv", ensemble_rows)
    json_dump(out / "paired_statistics_per_seed.json", per_seed_paired)
    json_dump(out / "paired_statistics_ensemble.json", ensemble_paired)
    json_dump(
        out / "validation.json",
        {
            "status": status,
            "reason": reason,
            "selection_seed": seeds[0],
            "confirmation_seeds": seeds[1:],
            "confirmation_pass_count": confirmation_pass_count,
            "num_confirmation_seeds": len(confirmation),
            "single_seed_stable": single_seed_stable,
            "ensemble_pass": ensemble_pass,
            "frozen_setting": {
                "flow_steps": args.flow_steps,
                "flow_solver": args.flow_solver,
                "ddpm_t_start": args.ddpm_t_start,
                "ddpm_steps": args.ddpm_steps,
                "alpha_flow": args.alpha_flow,
                "alpha_prior": args.alpha_prior,
            },
            "guards": {
                "min_mse_gain": args.min_mse_gain,
                "min_mse_wins": args.min_mse_wins,
                "mae_tolerance": args.mae_tolerance,
                "ssim_guard": args.ssim_guard,
            },
            "per_seed_summary": per_seed_summary,
            "ensemble_summary": ensemble_summary,
            "flow_checkpoint_epoch": int(flow_ckpt.get("epoch", -1)),
            "ddpm_checkpoint_epoch": int(ddpm_ckpt.get("epoch", -1)),
            "flow_checkpoint": str(Path(args.flow_ckpt).resolve()),
            "ddpm_checkpoint": str(Path(args.ddpm_ckpt).resolve()),
            "eval_files": file_manifest(dataset.files),
            "config": vars(args),
        },
    )

    print("=" * 139)
    print("G3 frozen-setting multi-seed confirmation")
    print("status             =", status)
    print("confirmation pass  =", confirmation_pass_count, "/", len(confirmation))
    print("ensemble pass      =", ensemble_pass)
    print("=" * 139)
    print(
        f"{'role':>12} {'seed':>10} {'dMSE':>10} {'MSE+':>6} "
        f"{'dMAE':>10} {'MAE+':>6} {'dSSIM':>11} {'SSIM+':>7} {'PASS':>6}"
    )
    for r in per_seed_summary:
        print(
            f"{r['role']:>12} {r['seed']:10d} "
            f"{r['mse_improvement_mean']:10.5f} {r['mse_wins']:6d} "
            f"{r['mae_improvement_mean']:10.6f} {r['mae_wins']:6d} "
            f"{r['ssim_improvement_mean']:11.7f} {r['ssim_wins']:7d} "
            f"{str(r['all_guards_pass']):>6}"
        )
    r = ensemble_summary
    print(
        f"{'ensemble':>12} {'mean':>10} "
        f"{r['mse_improvement_mean']:10.5f} {r['mse_wins']:6d} "
        f"{r['mae_improvement_mean']:10.6f} {r['mae_wins']:6d} "
        f"{r['ssim_improvement_mean']:11.7f} {r['ssim_wins']:7d} "
        f"{str(r['all_guards_pass']):>6}"
    )
    print("saved to:", out.resolve())


if __name__ == "__main__":
    main()
