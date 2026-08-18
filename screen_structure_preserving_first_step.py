#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Low-cost first-step screening for structure-preserving CBS candidate selection.

Goal
----
The current Stable CBS rule picks the candidate with the smallest true CBS
measurement loss. Existing 50-sample diagnostics show that this strongly
improves MSE / Rel-L2 but often worsens MAE / SSIM from the first iteration.

This script reuses the *same* first-step candidate pool as the current stable
CBS code (zero + multi-scale +/- candidates), evaluates each non-zero candidate
with true CBS only once, and then retrospectively applies many structure-aware
selection rules without any additional CBS calls.

Selection rule (does NOT use ground truth)
-------------------------------------------
Let E0 be the current true-CBS measurement loss, and Ebest the best candidate
loss in the pool. For a physics-retention parameter rho in (0, 1], define

    eligible(rho) = {j : E0 - Ej >= rho * (E0 - Ebest)}.

Among eligible candidates, choose the one that minimizes a structure-drift
score relative to the fixed InversionNet condition:

    D = D_value_norm + gamma * D_grad_norm.

D_value is mean absolute speed drift, D_grad is mean absolute finite-difference
gradient drift. Both are normalized by the largest drift in the candidate pool
for dimensionless comparison.

rho=1 reproduces pure best-physics selection (up to ties). Smaller rho allows
trading a fraction of the best available physics improvement for less structural
drift. Ground truth is used *only after selection* to evaluate MSE/MAE/SSIM.

Recommended first run: 10 samples, rho={1,0.75,0.5,0.25}, gamma={0,0.5,1}.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from run_cbs_physics_correction import (
    upsample_256_to_480_np,
    downsample_480_to_256_np,
    forward_dobs,
    compute_meas_loss,
    compute_adjoint_grad,
    normalize_grad,
    image_metrics,
    psnr_from_mse,
    rel_l1,
    rel_l2,
)

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None


def to_2d(x):
    x = np.asarray(x)
    while x.ndim > 2:
        x = x[0]
    return x.astype(np.float32)


def numeric_id(path: Path):
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        digits = "".join(ch for ch in path.parent.name if ch.isdigit())
    return int(digits) if digits else 10**12


def find_npz_files(root: Path):
    files = list(root.rglob("*.npz"))
    files = [p for p in files if p.name not in {"final_result.npz", "mean_dobs.npz"}]
    return sorted(files, key=lambda p: (numeric_id(p), str(p)))


def build_index(root: Path):
    out = {}
    for p in find_npz_files(root):
        i = numeric_id(p)
        if i < 10**12 and i not in out:
            out[i] = p
    return out


def ssim2d(pred, target, data_range=205.0):
    if skimage_ssim is None:
        return float("nan")
    return float(skimage_ssim(
        target.astype(np.float32),
        pred.astype(np.float32),
        data_range=float(data_range),
    ))


def grad_drift(a, b):
    """Mean absolute finite-difference gradient difference, in m/s per pixel."""
    ax = a[:, 1:] - a[:, :-1]
    bx = b[:, 1:] - b[:, :-1]
    ay = a[1:, :] - a[:-1, :]
    by = b[1:, :] - b[:-1, :]
    return 0.5 * (float(np.mean(np.abs(ax - bx))) + float(np.mean(np.abs(ay - by))))


def image_eval(pred480, target480, target256, data_range):
    pred256 = downsample_480_to_256_np(pred480)
    mse256, mae256, rmse256 = image_metrics(pred256, target256)
    mse480, mae480, rmse480 = image_metrics(pred480, target480)
    return {
        "mse256": mse256,
        "mae256": mae256,
        "rmse256": rmse256,
        "psnr256": psnr_from_mse(mse256, data_range),
        "ssim256": ssim2d(pred256, target256, data_range),
        "mse480": mse480,
        "mae480": mae480,
        "rmse480": rmse480,
        "psnr480": psnr_from_mse(mse480, data_range),
        "ssim480": ssim2d(pred480, target480, data_range),
    }


def build_candidate_pool(current, condition, dobs_gt, src, rec, args, device):
    """Build/evaluate the current Stable-CBS first-step pool exactly once."""
    grad, _, dobs_before = compute_adjoint_grad(current, dobs_gt, src, rec, args, device)
    grad_norm, grad_rms = normalize_grad(grad)

    # At iteration 1 current == condition, so prior direction is exactly zero.
    if args.prior_tether > 0:
        prior_grad = current - condition
        prior_norm, prior_rms = normalize_grad(prior_grad)
        total_grad = grad_norm + args.prior_tether * prior_norm
    else:
        prior_rms = 0.0
        total_grad = grad_norm

    current_loss = compute_meas_loss(dobs_before, dobs_gt)
    pool = [{
        "name": "zero",
        "sign": "0",
        "factor": 0.0,
        "step_mps": 0.0,
        "speed": current.copy(),
        "dobs": dobs_before,
        "phys_abs": current_loss,
    }]

    for fac in args.step_factors:
        step = args.step_size_mps * fac
        for sign, mult in [("-", -1.0), ("+", 1.0)]:
            cand = np.clip(
                current + mult * step * total_grad,
                args.speed_min,
                args.speed_max,
            ).astype(np.float32)
            pred = forward_dobs(cand, src, rec, args, device)
            pool.append({
                "name": f"{sign}{fac:g}",
                "sign": sign,
                "factor": float(fac),
                "step_mps": float(step),
                "speed": cand,
                "dobs": pred,
                "phys_abs": compute_meas_loss(pred, dobs_gt),
            })

    # Cheap structure drifts: fixed data-driven condition is the reference.
    max_val = 0.0
    max_grad = 0.0
    for c in pool:
        c["value_drift"] = float(np.mean(np.abs(c["speed"] - condition)))
        c["grad_drift"] = grad_drift(c["speed"], condition)
        max_val = max(max_val, c["value_drift"])
        max_grad = max(max_grad, c["grad_drift"])

    for c in pool:
        c["value_drift_norm"] = c["value_drift"] / (max_val + 1e-12)
        c["grad_drift_norm"] = c["grad_drift"] / (max_grad + 1e-12)

    return pool, grad_rms, prior_rms


def choose_candidate(pool, rho, gamma):
    e0 = pool[0]["phys_abs"]
    ebest = min(c["phys_abs"] for c in pool)
    best_gain = max(e0 - ebest, 0.0)

    if best_gain <= 1e-15:
        return pool[0], {
            "best_phys_abs": ebest,
            "best_phys_gain": 0.0,
            "required_gain": 0.0,
            "num_eligible": 1,
        }

    required_gain = float(rho) * best_gain
    eligible = [
        c for c in pool
        if (e0 - c["phys_abs"]) >= required_gain - 1e-12
    ]
    if not eligible:
        # Numerical safeguard; the best-physics candidate must normally qualify.
        eligible = [min(pool, key=lambda c: c["phys_abs"])]

    def key(c):
        struct = c["value_drift_norm"] + float(gamma) * c["grad_drift_norm"]
        return (struct, c["phys_abs"])

    chosen = min(eligible, key=key)
    return chosen, {
        "best_phys_abs": ebest,
        "best_phys_gain": best_gain,
        "required_gain": required_gain,
        "num_eligible": len(eligible),
    }


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def aggregate(rows):
    out = {"n": len(rows)}
    if not rows:
        return out

    numeric_keys = [
        k for k, v in rows[0].items()
        if isinstance(v, (int, float, np.number)) and k not in {"sample_id"}
    ]
    for k in numeric_keys:
        arr = np.asarray([float(r[k]) for r in rows], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            out[f"{k}_mean"] = float(arr.mean())
            out[f"{k}_std"] = float(arr.std(ddof=0))

    for metric, higher in [
        ("mse256", False), ("mae256", False), ("psnr256", True),
        ("ssim256", True), ("phys_abs", False), ("phys_rel_l2", False),
    ]:
        a = np.asarray([r[f"init_{metric}"] for r in rows], dtype=float)
        b = np.asarray([r[f"final_{metric}"] for r in rows], dtype=float)
        if higher:
            improved = int(np.sum(b > a))
            worsened = int(np.sum(b < a))
        else:
            improved = int(np.sum(b < a))
            worsened = int(np.sum(b > a))
        out[f"{metric}_improved"] = improved
        out[f"{metric}_worsened"] = worsened
        out[f"{metric}_tied"] = int(len(rows) - improved - worsened)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_root", required=True)
    ap.add_argument("--condition_root", required=True)
    ap.add_argument("--output_dir", default="./final_thesis_results/structure_screen_first_step")
    ap.add_argument("--max_samples", type=int, default=10)
    ap.add_argument("--rhos", nargs="+", type=float, default=[1.0, 0.75, 0.5, 0.25])
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--step_factors", nargs="+", type=float, default=[1.0, 0.5, 0.25, 0.1])
    ap.add_argument("--step_size_mps", type=float, default=1.0)
    ap.add_argument("--prior_tether", type=float, default=0.1)

    ap.add_argument("--frequency", type=float, default=500000.0)
    ap.add_argument("--forward_iters", type=int, default=80)
    ap.add_argument("--adjoint_iters", type=int, default=80)
    ap.add_argument("--boundary_width", type=int, default=300)
    ap.add_argument("--boundary_strength", type=float, default=225.0)
    ap.add_argument("--boundary_type", default="PML3")
    ap.add_argument("--smooth_kernel", type=int, default=9)
    ap.add_argument("--speed_min", type=float, default=1400.0)
    ap.add_argument("--speed_max", type=float, default=1605.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if any(r <= 0 or r > 1 for r in args.rhos):
        raise ValueError("All --rhos must be in (0, 1].")
    if any(g < 0 for g in args.gammas):
        raise ValueError("All --gammas must be >= 0.")

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_idx = build_index(Path(args.sample_root))
    cond_idx = build_index(Path(args.condition_root))
    common = sorted(set(sample_idx) & set(cond_idx))
    if args.max_samples > 0:
        common = common[:args.max_samples]
    if not common:
        raise RuntimeError("No matched numeric sample IDs between sample_root and condition_root")

    settings = [(float(r), float(g)) for r in args.rhos for g in args.gammas]
    setting_rows = {(r, g): [] for r, g in settings}
    audit_rows = []

    print("device =", device)
    print("num_samples =", len(common))
    print("settings =", settings)
    print("Each sample uses one adjoint/forward pass plus", 2 * len(args.step_factors), "candidate CBS forwards.")

    t_all = time.perf_counter()
    for pos, sid in enumerate(common, 1):
        print("=" * 88)
        print(f"[{pos}/{len(common)}] sample_id={sid}")

        sc = np.load(sample_idx[sid])
        cc = np.load(cond_idx[sid])
        target480 = sc["target_480"].astype(np.float32)
        target256 = sc["target_256"].astype(np.float32)
        dobs_gt = sc["dobs_complex"].astype(np.complex64)
        src = sc["src_indices"].astype(np.int64)
        rec = sc["rec_indices"].astype(np.int64)

        cond256_img = to_2d(cc["condition_speed"])
        cond256 = cond256_img.T
        condition480 = np.clip(
            upsample_256_to_480_np(cond256),
            args.speed_min,
            args.speed_max,
        ).astype(np.float32)
        current = condition480.copy()

        t0 = time.perf_counter()
        pool, grad_rms, prior_rms = build_candidate_pool(
            current, condition480, dobs_gt, src, rec, args, device
        )
        elapsed_pool = time.perf_counter() - t0

        init_img = image_eval(current, target480, target256, args.speed_max - args.speed_min)
        init_phys_abs = pool[0]["phys_abs"]
        init_phys_rel1 = rel_l1(pool[0]["dobs"], dobs_gt)
        init_phys_rel2 = rel_l2(pool[0]["dobs"], dobs_gt)

        # Candidate-level audit with GT metrics for analysis only.
        for c in pool:
            im = image_eval(c["speed"], target480, target256, args.speed_max - args.speed_min)
            audit_rows.append({
                "sample_id": sid,
                "candidate": c["name"],
                "sign": c["sign"],
                "factor": c["factor"],
                "step_mps": c["step_mps"],
                "phys_abs": c["phys_abs"],
                "phys_rel_l1": rel_l1(c["dobs"], dobs_gt),
                "phys_rel_l2": rel_l2(c["dobs"], dobs_gt),
                "value_drift": c["value_drift"],
                "grad_drift": c["grad_drift"],
                **im,
            })

        for rho, gamma in settings:
            chosen, info = choose_candidate(pool, rho, gamma)
            final_img = image_eval(chosen["speed"], target480, target256, args.speed_max - args.speed_min)
            final_phys_rel1 = rel_l1(chosen["dobs"], dobs_gt)
            final_phys_rel2 = rel_l2(chosen["dobs"], dobs_gt)
            struct_score = chosen["value_drift_norm"] + gamma * chosen["grad_drift_norm"]

            row = {
                "sample_id": sid,
                "rho": rho,
                "gamma": gamma,
                "chosen": chosen["name"],
                "chosen_step_mps": chosen["step_mps"],
                "chosen_sign": chosen["sign"],
                "num_eligible": info["num_eligible"],
                "best_phys_gain": info["best_phys_gain"],
                "required_gain": info["required_gain"],
                "value_drift": chosen["value_drift"],
                "grad_drift": chosen["grad_drift"],
                "struct_score": struct_score,
                "grad_rms": grad_rms,
                "prior_rms": prior_rms,
                "pool_wall_time_sec": elapsed_pool,
                "init_mse256": init_img["mse256"],
                "final_mse256": final_img["mse256"],
                "init_mae256": init_img["mae256"],
                "final_mae256": final_img["mae256"],
                "init_psnr256": init_img["psnr256"],
                "final_psnr256": final_img["psnr256"],
                "init_ssim256": init_img["ssim256"],
                "final_ssim256": final_img["ssim256"],
                "init_mse480": init_img["mse480"],
                "final_mse480": final_img["mse480"],
                "init_mae480": init_img["mae480"],
                "final_mae480": final_img["mae480"],
                "init_psnr480": init_img["psnr480"],
                "final_psnr480": final_img["psnr480"],
                "init_ssim480": init_img["ssim480"],
                "final_ssim480": final_img["ssim480"],
                "init_phys_abs": init_phys_abs,
                "final_phys_abs": chosen["phys_abs"],
                "init_phys_rel_l1": init_phys_rel1,
                "final_phys_rel_l1": final_phys_rel1,
                "init_phys_rel_l2": init_phys_rel2,
                "final_phys_rel_l2": final_phys_rel2,
            }
            setting_rows[(rho, gamma)].append(row)

        print("best-physics candidate =", min(pool, key=lambda c: c["phys_abs"])["name"])
        print("pool wall time sec =", f"{elapsed_pool:.2f}")

    write_csv(audit_rows, out_dir / "candidate_audit.csv")

    summary_rows = []
    summary_json = {
        "num_samples": len(common),
        "config": vars(args),
        "settings": {},
        "note": "Ground truth metrics are evaluation-only; candidate selection uses true CBS measurement loss and condition-relative structure drift only.",
    }

    for rho, gamma in settings:
        rows = setting_rows[(rho, gamma)]
        tag = f"rho{rho:g}_gamma{gamma:g}".replace(".", "p")
        write_csv(rows, out_dir / f"per_sample_{tag}.csv")
        agg = aggregate(rows)
        summary_json["settings"][tag] = agg

        summary_rows.append({
            "setting": tag,
            "rho": rho,
            "gamma": gamma,
            "mse256": agg.get("final_mse256_mean", float("nan")),
            "mae256": agg.get("final_mae256_mean", float("nan")),
            "psnr256": agg.get("final_psnr256_mean", float("nan")),
            "ssim256": agg.get("final_ssim256_mean", float("nan")),
            "phys_rel_l2": agg.get("final_phys_rel_l2_mean", float("nan")),
            "phys_abs": agg.get("final_phys_abs_mean", float("nan")),
            "mse_improved": agg.get("mse256_improved", 0),
            "mae_improved": agg.get("mae256_improved", 0),
            "ssim_improved": agg.get("ssim256_improved", 0),
            "phys_improved": agg.get("phys_abs_improved", 0),
            "mean_value_drift": agg.get("value_drift_mean", float("nan")),
            "mean_grad_drift": agg.get("grad_drift_mean", float("nan")),
        })

    write_csv(summary_rows, out_dir / "screen_summary.csv")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    # Human-readable ranking: prioritize structural recovery while preserving physics.
    print("\n" + "=" * 120)
    print("SCREEN SUMMARY (one-step; GT used for evaluation only)")
    print("=" * 120)
    print(f"{'setting':22s} {'MSE256':>10s} {'MAE256':>9s} {'SSIM':>8s} {'RelL2':>9s} {'MSE+':>6s} {'MAE+':>6s} {'SSIM+':>7s} {'PHY+':>6s}")
    for r in summary_rows:
        print(
            f"{r['setting']:22s} "
            f"{r['mse256']:10.3f} {r['mae256']:9.4f} {r['ssim256']:8.4f} {r['phys_rel_l2']:9.5f} "
            f"{r['mse_improved']:6d} {r['mae_improved']:6d} {r['ssim_improved']:7d} {r['phys_improved']:6d}"
        )

    print("\nTotal wall time sec =", f"{time.perf_counter() - t_all:.2f}")
    print("Saved to:", out_dir.resolve())
    print("Key file:", (out_dir / "screen_summary.csv").resolve())


if __name__ == "__main__":
    main()
