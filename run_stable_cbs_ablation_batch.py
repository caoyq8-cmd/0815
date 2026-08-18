#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch ablation for the Stable CBS correction mechanism.

Place this script in the 0815 repository root (or run with that root on
PYTHONPATH). It reuses the verified forward/adjoint utilities from
``run_cbs_physics_correction.py`` so geometry and CBS parameters stay identical.

Ablations are cumulative and deliberately keep RMS normalization in the main
comparison to avoid the severe step-scale confounding caused by removing it:
  A0_fixed       : one normalized -gradient step, no candidate verification, no tether
  A1_backtrack   : negative direction, multi-scale + zero candidate, true CBS verification
  A2_bidirectional: multi-scale +/- directions + zero, true CBS verification
  A3_full        : A2 + prior tether (default 0.1)
  A4_no_smoothing: full method but smooth_kernel=1

The script reports image metrics, physical residuals, wall time, candidate
forward calls and adjoint calls for every variant/sample.
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


def to_2d(x):
    x = np.asarray(x)
    while x.ndim > 2:
        x = x[0]
    return x.astype(np.float32)


def numeric_id(path: Path):
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 10**12


def find_npz_files(root: Path):
    files = list(root.rglob("*.npz"))
    # exclude obvious result/aggregate files if roots are broad
    files = [p for p in files if p.name not in {"final_result.npz", "mean_dobs.npz"}]
    return sorted(files, key=lambda p: (numeric_id(p), str(p)))


def build_index(root: Path):
    out = {}
    for p in find_npz_files(root):
        i = numeric_id(p)
        if i < 10**12 and i not in out:
            out[i] = p
    return out


def metric_row(prefix, pred256, target256, pred480, target480, dobs, dobs_gt, data_range=205.0):
    m256, a256, r256 = image_metrics(pred256, target256)
    m480, a480, r480 = image_metrics(pred480, target480)
    return {
        f"{prefix}_mse256": m256,
        f"{prefix}_mae256": a256,
        f"{prefix}_rmse256": r256,
        f"{prefix}_psnr256": psnr_from_mse(m256, data_range),
        f"{prefix}_mse480": m480,
        f"{prefix}_mae480": a480,
        f"{prefix}_rmse480": r480,
        f"{prefix}_psnr480": psnr_from_mse(m480, data_range),
        f"{prefix}_dobs_abs": compute_meas_loss(dobs, dobs_gt),
        f"{prefix}_dobs_rel_l1": rel_l1(dobs, dobs_gt),
        f"{prefix}_dobs_rel_l2": rel_l2(dobs, dobs_gt),
    }


def variant_specs(prior_tether):
    return {
        "A0_fixed": dict(verify=False, both=False, factors=[1.0], tether=0.0, smooth=None),
        "A1_backtrack": dict(verify=True, both=False, factors=[1.0, 0.5, 0.25, 0.1], tether=0.0, smooth=None),
        "A2_bidirectional": dict(verify=True, both=True, factors=[1.0, 0.5, 0.25, 0.1], tether=0.0, smooth=None),
        "A3_full": dict(verify=True, both=True, factors=[1.0, 0.5, 0.25, 0.1], tether=prior_tether, smooth=None),
        "A4_no_smoothing": dict(verify=True, both=True, factors=[1.0, 0.5, 0.25, 0.1], tether=prior_tether, smooth=1),
    }


def correction_direction(current, condition, grad, tether):
    g, grad_rms = normalize_grad(grad)
    prior_rms = 0.0
    if tether > 0:
        pg, prior_rms = normalize_grad(current - condition)
        g = g + tether * pg
    return g.astype(np.float32), grad_rms, prior_rms


def verified_step(current, condition, dobs_gt, src_idx, rec_idx, args, device, spec):
    local_args = SimpleNamespace(**vars(args))
    if spec["smooth"] is not None:
        local_args.smooth_kernel = spec["smooth"]

    grad, _, dobs_before = compute_adjoint_grad(current, dobs_gt, src_idx, rec_idx, local_args, device)
    direction, grad_rms, prior_rms = correction_direction(current, condition, grad, spec["tether"])
    current_loss = compute_meas_loss(dobs_before, dobs_gt)

    # A0: direct normalized fixed step, no candidate CBS verification.
    if not spec["verify"]:
        chosen = np.clip(current - args.step_size_mps * direction, args.speed_min, args.speed_max).astype(np.float32)
        return chosen, None, {
            "loss_before": current_loss,
            "chosen": "direct_minus_1.0",
            "grad_rms": grad_rms,
            "prior_rms": prior_rms,
            "candidate_forward_calls": 0,
        }

    candidates = [("zero", current.copy(), 0.0, "0")]
    for fac in spec["factors"]:
        step = args.step_size_mps * fac
        candidates.append((f"-{fac}", np.clip(current - step * direction, args.speed_min, args.speed_max).astype(np.float32), step, "-"))
        if spec["both"]:
            candidates.append((f"+{fac}", np.clip(current + step * direction, args.speed_min, args.speed_max).astype(np.float32), step, "+"))

    best = dict(name="zero", speed=current.copy(), dobs=dobs_before, loss=current_loss, step=0.0, sign="0")
    cbs_calls = 0
    for name, speed, step, sign in candidates[1:]:
        pred = forward_dobs(speed, src_idx, rec_idx, local_args, device)
        cbs_calls += 1
        loss = compute_meas_loss(pred, dobs_gt)
        if loss < best["loss"]:
            best = dict(name=name, speed=speed, dobs=pred, loss=loss, step=step, sign=sign)

    return best["speed"], best["dobs"], {
        "loss_before": current_loss,
        "loss_after": best["loss"],
        "chosen": best["name"],
        "chosen_sign": best["sign"],
        "chosen_step": best["step"],
        "grad_rms": grad_rms,
        "prior_rms": prior_rms,
        "candidate_forward_calls": cbs_calls,
    }


def run_variant(sample, condition, args, device, spec):
    target480 = sample["target_480"].astype(np.float32)
    target256 = sample["target_256"].astype(np.float32)
    dobs_gt = sample["dobs_complex"].astype(np.complex64)
    src = sample["src_indices"].astype(np.int64)
    rec = sample["rec_indices"].astype(np.int64)

    cond256_img = to_2d(condition["condition_speed"])
    cond256 = cond256_img.T
    current = np.clip(upsample_256_to_480_np(cond256), args.speed_min, args.speed_max).astype(np.float32)
    condition480 = current.copy()

    t0 = time.perf_counter()
    dobs_init = forward_dobs(current, src, rec, args, device)
    initial = metric_row("init", cond256, target256, current, target480, dobs_init, dobs_gt, args.speed_max - args.speed_min)

    history = []
    candidate_calls = 0
    for it in range(1, args.num_iters + 1):
        current, chosen_dobs, info = verified_step(current, condition480, dobs_gt, src, rec, args, device, spec)
        candidate_calls += int(info["candidate_forward_calls"])
        history.append({"iter": it, **info})

    # If the direct-update variant did not verify the last update, evaluate it now.
    if chosen_dobs is None:
        chosen_dobs = forward_dobs(current, src, rec, args, device)
        candidate_calls += 1

    final256 = downsample_480_to_256_np(current)
    final = metric_row("final", final256, target256, current, target480, chosen_dobs, dobs_gt, args.speed_max - args.speed_min)
    elapsed = time.perf_counter() - t0

    row = {**initial, **final}
    row.update({
        "wall_time_sec": elapsed,
        "adjoint_calls": args.num_iters,
        # compute_adjoint_grad itself performs one CBS forward each iteration
        "forward_calls_total": 1 + args.num_iters + candidate_calls,
        "candidate_forward_calls": candidate_calls,
        "mse256_improve_ratio": (initial["init_mse256"] - final["final_mse256"]) / (initial["init_mse256"] + 1e-12),
        "mse480_improve_ratio": (initial["init_mse480"] - final["final_mse480"]) / (initial["init_mse480"] + 1e-12),
        "dobs_improve_ratio": (initial["init_dobs_abs"] - final["final_dobs_abs"]) / (initial["init_dobs_abs"] + 1e-12),
    })
    return row, history


def aggregate(rows):
    numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float, np.number)) and k not in {"sample_id"}]
    out = {"n": len(rows)}
    for k in numeric:
        a = np.asarray([float(r[k]) for r in rows], dtype=float)
        out[f"{k}_mean"] = float(a.mean())
        out[f"{k}_std"] = float(a.std(ddof=0))
    out["num_mse256_improved"] = int(sum(r["final_mse256"] < r["init_mse256"] for r in rows))
    out["num_mse480_improved"] = int(sum(r["final_mse480"] < r["init_mse480"] for r in rows))
    out["num_dobs_improved"] = int(sum(r["final_dobs_abs"] < r["init_dobs_abs"] for r in rows))
    return out


def write_csv(rows, path):
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_root", required=True)
    ap.add_argument("--condition_root", required=True)
    ap.add_argument("--output_root", default="./final_ablation_runs/stable_cbs_ablation")
    ap.add_argument("--variants", nargs="+", default=["A0_fixed", "A1_backtrack", "A2_bidirectional", "A3_full", "A4_no_smoothing"])
    ap.add_argument("--max_samples", type=int, default=20)
    ap.add_argument("--num_iters", type=int, default=8)
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

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sample_idx = build_index(Path(args.sample_root))
    cond_idx = build_index(Path(args.condition_root))
    common = sorted(set(sample_idx) & set(cond_idx))
    if args.max_samples > 0:
        common = common[: args.max_samples]
    if not common:
        raise RuntimeError("No matched numeric sample IDs between sample_root and condition_root")

    specs = variant_specs(args.prior_tether)
    unknown = [v for v in args.variants if v not in specs]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}; available={list(specs)}")

    out_root = Path(args.output_root); out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    all_summary = {}
    for variant in args.variants:
        print("=" * 88); print("VARIANT", variant); print("=" * 88)
        vdir = out_root / variant; vdir.mkdir(exist_ok=True)
        rows = []
        for j, sid in enumerate(common, 1):
            print(f"[{variant}] sample {j}/{len(common)} id={sid}")
            with np.load(sample_idx[sid]) as sc, np.load(cond_idx[sid]) as cc:
                row, history = run_variant(sc, cc, args, device, specs[variant])
            row = {"sample_id": sid, **row}
            rows.append(row)
            sdir = vdir / f"sample_{sid:04d}"; sdir.mkdir(exist_ok=True)
            with open(sdir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        write_csv(rows, vdir / "per_sample.csv")
        summ = aggregate(rows)
        all_summary[variant] = summ
        with open(vdir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=2, ensure_ascii=False)

    with open(out_root / "all_variants_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(all_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
