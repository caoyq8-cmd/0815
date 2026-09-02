#!/usr/bin/env python3
"""Screen a frozen unconditional speed-DDPM as a conservative SDEdit prior.

This script does not train or fine-tune the DDPM.  Starting from a cached
InversionNet reconstruction ``condition``, it

  1. adds a reproducible amount of forward-diffusion noise at ``t_start``;
  2. performs deterministic DDIM (eta=0) back to a clean speed map;
  3. blends only a fraction alpha of that prior projection into condition;
  4. selects a setting on VAL20 using pre-registered MSE/MAE/SSIM guards.

The DEV30 split must not be used by this screening script.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_icrfm_oof_v2 import (
    ConditionCacheDataset,
    bootstrap_paired,
    denormalize_speed,
    file_manifest,
    json_dump,
    metric_row,
    write_csv,
)
from train_speed_ddpm_prior_v73 import GaussianDiffusion, SimpleDDPMUNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_args(ckpt: Dict) -> Dict:
    cfg = ckpt.get("args", ckpt.get("config", {}))
    if hasattr(cfg, "__dict__"):
        cfg = vars(cfg)
    if not isinstance(cfg, dict):
        raise TypeError("checkpoint args/config must be a mapping")
    return dict(cfg)


def load_ddpm(
    ckpt_path: str, device: torch.device
) -> Tuple[torch.nn.Module, GaussianDiffusion, Dict, Dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = checkpoint_args(ckpt)

    model = SimpleDDPMUNet(
        in_ch=1,
        base_ch=int(cfg.get("base_ch", 48)),
        time_dim=int(cfg.get("time_dim", 192)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    state_key = "ema_model" if "ema_model" in ckpt else "model"
    if state_key not in ckpt:
        raise KeyError("checkpoint contains neither 'ema_model' nor 'model'")
    incompatible = model.load_state_dict(ckpt[state_key], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"state-dict mismatch: {incompatible}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    diffusion = GaussianDiffusion(
        timesteps=int(cfg.get("timesteps", 1000)), device=device
    )
    return model, diffusion, cfg, ckpt


def fixed_noise_like(
    x: torch.Tensor, sample_ids: torch.Tensor, seed: int
) -> torch.Tensor:
    """One deterministic noise realization per sample, independent of batching."""
    pieces = []
    for sid in sample_ids.detach().cpu().tolist():
        generator = torch.Generator(device=x.device)
        generator.manual_seed(int(seed) + 1009 * int(sid))
        pieces.append(
            torch.randn(
                (1, *x.shape[1:]),
                generator=generator,
                device=x.device,
                dtype=x.dtype,
            )
        )
    return torch.cat(pieces, dim=0)


def ddim_times(t_start: int, num_steps: int, device: torch.device) -> torch.Tensor:
    if t_start < 1:
        raise ValueError("t_start must be >= 1")
    if num_steps < 1:
        raise ValueError("ddim_steps must be >= 1")
    # Rounding can duplicate values if num_steps > t_start + 1.  Remove those
    # duplicates while keeping a strictly decreasing schedule.
    times = torch.linspace(t_start, 0, num_steps, device=device).round().long()
    times = torch.unique_consecutive(times)
    return times


@torch.no_grad()
def sdedit_ddim(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    condition: torch.Tensor,
    sample_ids: torch.Tensor,
    t_start: int,
    num_steps: int,
    seed: int,
) -> Tuple[torch.Tensor, int]:
    """Forward-noise condition and deterministically DDIM-project it to t=-1."""
    if t_start >= diffusion.timesteps:
        raise ValueError(
            f"t_start={t_start} must be below timesteps={diffusion.timesteps}"
        )
    batch_size = condition.shape[0]
    t_batch = torch.full(
        (batch_size,), t_start, device=condition.device, dtype=torch.long
    )
    noise = fixed_noise_like(condition, sample_ids, seed)
    x = diffusion.q_sample(condition, t_batch, noise=noise)

    times = ddim_times(t_start, num_steps, condition.device)
    time_pairs = list(zip(times[:-1], times[1:]))
    time_pairs.append((times[-1], torch.tensor(-1, device=condition.device)))

    for t_now, t_next in time_pairs:
        t = torch.full(
            (batch_size,), int(t_now.item()), device=condition.device, dtype=torch.long
        )
        pred_noise = model(x, t)
        alpha_now = diffusion.alphas_cumprod[t_now]
        x0_pred = (
            x - torch.sqrt(1.0 - alpha_now) * pred_noise
        ) / torch.sqrt(alpha_now)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        if int(t_next.item()) < 0:
            x = x0_pred
        else:
            alpha_next = diffusion.alphas_cumprod[t_next]
            # Deterministic DDIM: eta=0, so the model-predicted noise direction
            # is reused and no reverse-process random noise is introduced.
            x = (
                torch.sqrt(alpha_next) * x0_pred
                + torch.sqrt(1.0 - alpha_next) * pred_noise
            )

    return torch.clamp(x, -1.0, 1.0), len(time_pairs)


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def safe_paired(delta: np.ndarray, seed: int, n_boot: int) -> Dict:
    if np.allclose(delta, 0.0):
        return {
            "mean": 0.0,
            "median": 0.0,
            "bootstrap_mean_95ci": [0.0, 0.0],
            "wilcoxon_two_sided_p": 1.0,
            "wilcoxon_improvement_p": 1.0,
            "wins": 0,
            "losses": 0,
        }
    return bootstrap_paired(delta, seed, n_boot)


def setting_key(row: Dict) -> Tuple[int, int, float]:
    steps = row.get("ddim_steps", row.get("ddim_steps_requested"))
    if steps is None:
        raise KeyError("row has neither ddim_steps nor ddim_steps_requested")
    return int(row["t_start"]), int(steps), float(row["alpha"])


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--condition_root", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--eval_start", type=int, default=1)
    ap.add_argument("--eval_end", type=int, default=20)
    ap.add_argument("--t_starts", type=int, nargs="+", default=[25, 50, 100, 200])
    ap.add_argument("--ddim_steps", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument(
        "--alphas", type=float, nargs="+", default=[0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    )
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mae_tolerance", type=float, default=0.0)
    ap.add_argument("--ssim_guard", type=float, default=0.001)
    ap.add_argument("--min_mse_gain", type=float, default=0.0)
    ap.add_argument("--bootstrap_samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260902)
    args = ap.parse_args()

    t_starts = sorted(set(int(x) for x in args.t_starts))
    ddim_steps = sorted(set(int(x) for x in args.ddim_steps))
    alphas = sorted(set(float(x) for x in args.alphas))
    if not alphas or alphas[0] < 0.0 or alphas[-1] > 1.0:
        raise ValueError("alphas must lie in [0, 1]")
    if 0.0 not in alphas:
        raise ValueError("alphas must include 0 as the identity baseline")
    if args.eval_split == "test" and not (
        args.eval_start == 1 and args.eval_end == 20
    ):
        raise RuntimeError(
            "Primary screening is frozen to VAL20=test1..20. "
            "Edit the source deliberately if a later protocol requires another split."
        )

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, diffusion, cfg, ckpt = load_ddpm(args.ckpt_path, device)
    norm_min = float(cfg.get("norm_min", 1400.0))
    norm_max = float(cfg.get("norm_max", 1605.0))
    data_range = norm_max - norm_min

    if any(t >= diffusion.timesteps or t < 1 for t in t_starts):
        raise ValueError(
            f"all t_starts must be in [1, {diffusion.timesteps - 1}]"
        )

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
    model.eval()
    with torch.no_grad():
        for t_start in t_starts:
            for requested_steps in ddim_steps:
                for batch in loader:
                    condition = batch["condition"].to(device, non_blocking=True)
                    sample_ids = batch["sample_id"]
                    projected, actual_nfe = sdedit_ddim(
                        model,
                        diffusion,
                        condition,
                        sample_ids,
                        t_start,
                        requested_steps,
                        args.seed,
                    )
                    condition_speed = batch["condition_speed"].numpy()
                    target_speed = batch["target_speed"].numpy()
                    ids = sample_ids.numpy()

                    for alpha in alphas:
                        pred_norm = torch.clamp(
                            condition + alpha * (projected - condition), -1.0, 1.0
                        )
                        pred_speed = denormalize_speed(
                            pred_norm, norm_min, norm_max
                        ).cpu().numpy()
                        update_rms = torch.sqrt(
                            torch.mean(
                                (
                                    denormalize_speed(pred_norm, norm_min, norm_max)
                                    - denormalize_speed(condition, norm_min, norm_max)
                                )
                                ** 2,
                                dim=(1, 2, 3),
                            )
                        ).cpu().numpy()

                        for i, sid in enumerate(ids):
                            cm = metric_row(
                                condition_speed[i, 0], target_speed[i, 0], data_range
                            )
                            pm = metric_row(
                                pred_speed[i, 0], target_speed[i, 0], data_range
                            )
                            rows.append(
                                {
                                    "sample_id": int(sid),
                                    "t_start": int(t_start),
                                    "ddim_steps_requested": int(requested_steps),
                                    "ddim_nfe": int(actual_nfe),
                                    "alpha": float(alpha),
                                    "noise_seed": int(args.seed + 1009 * int(sid)),
                                    "update_rms_mps": float(update_rms[i]),
                                    **{f"condition_{k}": v for k, v in cm.items()},
                                    **{f"sdedit_{k}": v for k, v in pm.items()},
                                    "mse_improvement": cm["mse"] - pm["mse"],
                                    "mae_improvement": cm["mae"] - pm["mae"],
                                    "psnr_improvement": pm["psnr"] - cm["psnr"],
                                    "ssim_improvement": pm["ssim"] - cm["ssim"],
                                }
                            )

    summary_rows: List[Dict] = []
    for t_start in t_starts:
        for requested_steps in ddim_steps:
            for alpha in alphas:
                part = [
                    r
                    for r in rows
                    if r["t_start"] == t_start
                    and r["ddim_steps_requested"] == requested_steps
                    and r["alpha"] == alpha
                ]
                row: Dict = {
                    "t_start": t_start,
                    "ddim_steps": requested_steps,
                    "ddim_nfe": int(part[0]["ddim_nfe"]),
                    "alpha": alpha,
                    "num_samples": len(part),
                }
                for prefix in ("condition", "sdedit"):
                    for metric in ("mse", "mae", "psnr", "ssim"):
                        mean, std = mean_std([r[f"{prefix}_{metric}"] for r in part])
                        row[f"{prefix}_{metric}_mean"] = mean
                        row[f"{prefix}_{metric}_std"] = std
                for metric in ("mse", "mae", "psnr", "ssim"):
                    values = np.asarray(
                        [r[f"{metric}_improvement"] for r in part], dtype=np.float64
                    )
                    row[f"{metric}_improvement_mean"] = float(values.mean())
                    row[f"{metric}_wins"] = int(np.sum(values > 0))
                row["update_rms_mps_mean"] = float(
                    np.mean([r["update_rms_mps"] for r in part])
                )
                row["mse_guard_pass"] = bool(
                    row["sdedit_mse_mean"]
                    <= row["condition_mse_mean"] - args.min_mse_gain
                )
                row["mae_guard_pass"] = bool(
                    row["sdedit_mae_mean"]
                    <= row["condition_mae_mean"] + args.mae_tolerance
                )
                row["ssim_guard_pass"] = bool(
                    row["sdedit_ssim_mean"]
                    >= row["condition_ssim_mean"] - args.ssim_guard
                )
                row["all_guards_pass"] = bool(
                    row["mse_guard_pass"]
                    and row["mae_guard_pass"]
                    and row["ssim_guard_pass"]
                )
                summary_rows.append(row)

    feasible = [
        r for r in summary_rows if r["alpha"] > 0 and r["all_guards_pass"]
    ]
    if feasible:
        selected = min(
            feasible,
            key=lambda r: (
                r["sdedit_mse_mean"],
                r["ddim_nfe"],
                r["t_start"],
                r["alpha"],
            ),
        )
        status = "PASS"
        reason = "a nonzero setting satisfies all preregistered guards"
    else:
        selected = next(
            r
            for r in summary_rows
            if r["t_start"] == t_starts[0]
            and r["ddim_steps"] == ddim_steps[0]
            and r["alpha"] == 0.0
        )
        status = "REJECT"
        reason = "no nonzero setting satisfies the MSE, MAE, and SSIM guards"

    selected_rows = [r for r in rows if setting_key(r) == setting_key(selected)]
    paired = {}
    for offset, metric in enumerate(("mse", "mae", "psnr", "ssim")):
        delta = np.asarray(
            [r[f"{metric}_improvement"] for r in selected_rows], dtype=np.float64
        )
        paired[metric] = safe_paired(
            delta,
            args.seed + 100000 + offset,
            args.bootstrap_samples,
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "sdedit_summary.csv", summary_rows)
    write_csv(out / "sdedit_per_sample.csv", rows)
    json_dump(out / "paired_statistics_selected.json", paired)
    json_dump(
        out / "selection.json",
        {
            "status": status,
            "reason": reason,
            "selected_setting": selected,
            "checkpoint_epoch": int(ckpt.get("epoch", -1)),
            "checkpoint_best_val_loss": float(ckpt.get("best_val_loss", np.nan)),
            "checkpoint_state": "ema_model" if "ema_model" in ckpt else "model",
            "checkpoint": str(Path(args.ckpt_path).resolve()),
            "checkpoint_architecture": {
                "base_ch": int(cfg.get("base_ch", 48)),
                "time_dim": int(cfg.get("time_dim", 192)),
                "timesteps": int(cfg.get("timesteps", 1000)),
                "norm_min": norm_min,
                "norm_max": norm_max,
            },
            "eval_files": file_manifest(dataset.files),
            "guards": {
                "mae_tolerance": args.mae_tolerance,
                "ssim_guard": args.ssim_guard,
                "min_mse_gain": args.min_mse_gain,
            },
            "config": vars(args),
        },
    )

    print("=" * 145)
    print("Frozen DDPM condition-centered SDEdit/DDIM sweep")
    print("checkpoint epoch =", int(ckpt.get("epoch", -1)))
    print("checkpoint state =", "ema_model" if "ema_model" in ckpt else "model")
    print("status           =", status)
    print(
        "selected         =",
        f"t={selected['t_start']} steps={selected['ddim_steps']} "
        f"alpha={selected['alpha']}",
    )
    print("=" * 145)
    print(
        f"{'t':>5} {'NFE':>5} {'alpha':>7} {'MSE':>12} {'MAE':>10} "
        f"{'PSNR':>10} {'SSIM':>10} {'dRMS':>9} "
        f"{'MSE+':>6} {'MAE+':>6} {'SSIM+':>7} {'PASS':>6}"
    )
    for r in summary_rows:
        print(
            f"{r['t_start']:5d} {r['ddim_nfe']:5d} {r['alpha']:7.3f} "
            f"{r['sdedit_mse_mean']:12.3f} {r['sdedit_mae_mean']:10.4f} "
            f"{r['sdedit_psnr_mean']:10.4f} {r['sdedit_ssim_mean']:10.6f} "
            f"{r['update_rms_mps_mean']:9.3f} {r['mse_wins']:6d} "
            f"{r['mae_wins']:6d} {r['ssim_wins']:7d} "
            f"{str(r['all_guards_pass']):>6}"
        )
    print("saved to:", out.resolve())


if __name__ == "__main__":
    main()
