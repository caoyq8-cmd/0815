#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quantify neural-surrogate gradient vs. true CBS adjoint gradient.

Important discretization detail:
The residual measurement surrogate differentiates w.r.t. a low-resolution
(typically 240x240) sound-speed variable, whereas CBS returns a gradient on
480x480. To compare gradients in the SAME parameterization, this script uses
the exact transpose/pullback of PyTorch bilinear interpolation:

    c_480 = R(c_240),   g_240^CBS = R^T g_480^CBS.

A naive resize(g_480 -> 240) is NOT generally the adjoint of R and should not
be used for a thesis gradient-cosine experiment.

By default the neural gradient is evaluated with pure measurement MSE
(lambda_l1=0, lambda_mse=1) so its objective is closer to the squared-residual
objective represented by the CBS adjoint source.
"""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from run_neural_proposal_cbs_validated_correction import (
    load_residual_measurement_model,
    make_coord_maps,
    compute_neural_grad,
    resize_np,
)
from run_cbs_physics_correction import compute_adjoint_grad, forward_dobs, compute_meas_loss, normalize_grad


def to_2d(x):
    x = np.asarray(x)
    while x.ndim > 2:
        x = x[0]
    return x.astype(np.float32)


def numeric_id(path: Path):
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 10**12


def build_index(root: Path):
    out = {}
    for p in sorted(root.rglob("*.npz"), key=lambda p: (numeric_id(p), str(p))):
        if p.name in {"mean_dobs.npz", "final_result.npz"}:
            continue
        i = numeric_id(p)
        if i < 10**12 and i not in out:
            out[i] = p
    return out


def bilinear_pullback(g480: np.ndarray, source_size: int, target_size: int = 480) -> np.ndarray:
    """Return R^T g, where R = bilinear interpolate source_size -> target_size."""
    x = torch.zeros((1, 1, source_size, source_size), dtype=torch.float32, requires_grad=True)
    y = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)
    g = torch.from_numpy(g480.astype(np.float32))[None, None]
    inner = torch.sum(y * g)
    inner.backward()
    return x.grad.detach().cpu().numpy()[0, 0].astype(np.float32)


def cosine(a, b, eps=1e-12):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b) + eps
    return float(np.dot(a, b) / den)


def optimal_scale_rel_error(pred, ref, eps=1e-12):
    p = pred.astype(np.float64).ravel(); r = ref.astype(np.float64).ravel()
    alpha = float(np.dot(p, r) / (np.dot(p, p) + eps))
    err = float(np.linalg.norm(alpha * p - r) / (np.linalg.norm(r) + eps))
    return alpha, err


def verify_direction(current240, grad240, target_dobs, src, rec, args, device):
    if args.verify_step_mps <= 0:
        return {}
    g, _ = normalize_grad(grad240)
    base480 = resize_np(current240, 480)
    base = forward_dobs(base480, src, rec, args, device)
    base_loss = compute_meas_loss(base, target_dobs)
    out = {"base_loss": base_loss}
    for sign_name, sgn in [("minus", -1.0), ("plus", 1.0)]:
        cand240 = np.clip(current240 + sgn * args.verify_step_mps * g, args.speed_min, args.speed_max).astype(np.float32)
        pred = forward_dobs(resize_np(cand240, 480), src, rec, args, device)
        loss = compute_meas_loss(pred, target_dobs)
        out[f"{sign_name}_loss"] = loss
        out[f"{sign_name}_improve_ratio"] = float((base_loss - loss) / (base_loss + 1e-12))
    out["best_improve_ratio"] = max(out["minus_improve_ratio"], out["plus_improve_ratio"], 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_root", required=True)
    ap.add_argument("--condition_root", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--mean_dobs_path", required=True)
    ap.add_argument("--output_dir", default="./gradient_consistency_eval")
    ap.add_argument("--max_samples", type=int, default=20)
    ap.add_argument("--lambda_l1", type=float, default=0.0, help="Use 0 for objective alignment with squared-residual CBS adjoint")
    ap.add_argument("--lambda_mse", type=float, default=1.0)
    ap.add_argument("--verify_step_mps", type=float, default=0.0, help=">0: also CBS-verify +/- normalized neural and CBS directions")
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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, ckpt_args = load_residual_measurement_model(args.ckpt_path, device)
    image_size = int(ckpt_args.get("image_size", 240))
    speed_center = float(ckpt_args.get("speed_center", 1500.0))
    speed_scale = float(ckpt_args.get("speed_scale", 100.0))
    residual_scale = float(ckpt_args.get("residual_scale", 0.02))
    mean_dobs = np.load(args.mean_dobs_path)["mean_dobs"].astype(np.complex64)
    mean_real = torch.from_numpy(mean_dobs.real.astype(np.float32)).to(device)
    mean_imag = torch.from_numpy(mean_dobs.imag.astype(np.float32)).to(device)
    y_map, x_map = make_coord_maps(image_size, device)

    samples = build_index(Path(args.sample_root)); conditions = build_index(Path(args.condition_root))
    ids = sorted(set(samples) & set(conditions))
    if args.max_samples > 0:
        ids = ids[: args.max_samples]
    if not ids:
        raise RuntimeError("No matched sample IDs")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for j, sid in enumerate(ids, 1):
        print(f"[{j}/{len(ids)}] sample={sid}")
        with np.load(samples[sid]) as sc, np.load(conditions[sid]) as cc:
            target_dobs = sc["dobs_complex"].astype(np.complex64)
            src = sc["src_indices"].astype(np.int64)
            rec = sc["rec_indices"].astype(np.int64)
            condition256 = to_2d(cc["condition_speed"]).T

        condition480 = np.clip(resize_np(condition256, 480), args.speed_min, args.speed_max).astype(np.float32)
        current240 = resize_np(condition480, image_size)

        neural_loss, g_neural, neural_pred = compute_neural_grad(
            model, current240, target_dobs, mean_real, mean_imag,
            residual_scale, speed_center, speed_scale, y_map, x_map, device,
            lambda_l1=args.lambda_l1, lambda_mse=args.lambda_mse,
        )
        g480, adj_loss, cbs_pred = compute_adjoint_grad(condition480, target_dobs, src, rec, args, device)
        g_cbs240 = bilinear_pullback(g480, image_size, 480)

        cos = cosine(g_neural, g_cbs240)
        alpha, scaled_rel = optimal_scale_rel_error(g_neural, g_cbs240)
        row = {
            "sample_id": sid,
            "cosine_raw": cos,
            "cosine_axis": abs(cos),
            "optimal_scale_neural_to_cbs": alpha,
            "scaled_gradient_rel_l2": scaled_rel,
            "neural_grad_rms": float(np.sqrt(np.mean(g_neural.astype(np.float64) ** 2))),
            "cbs_pullback_grad_rms": float(np.sqrt(np.mean(g_cbs240.astype(np.float64) ** 2))),
            "neural_loss": float(neural_loss),
            "cbs_reported_residual_abs": float(adj_loss),
            "neural_dobs_abs": float(np.mean(np.abs(neural_pred - target_dobs))),
            "cbs_dobs_abs": float(np.mean(np.abs(cbs_pred - target_dobs))),
        }

        if args.verify_step_mps > 0:
            ncheck = verify_direction(current240, g_neural, target_dobs, src, rec, args, device)
            ccheck = verify_direction(current240, g_cbs240, target_dobs, src, rec, args, device)
            row.update({f"neural_{k}": v for k, v in ncheck.items()})
            row.update({f"cbs_{k}": v for k, v in ccheck.items()})
        rows.append(row)

    fields = list(rows[0].keys())
    with open(out / "gradient_consistency_per_sample.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    summary = {"num_samples": len(rows)}
    for k in fields:
        if k == "sample_id":
            continue
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float, np.number))]
        if vals:
            a = np.asarray(vals, dtype=float)
            summary[f"{k}_mean"] = float(a.mean())
            summary[f"{k}_std"] = float(a.std(ddof=0))
            summary[f"{k}_median"] = float(np.median(a))
    with open(out / "gradient_consistency_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    cosv = np.asarray([r["cosine_raw"] for r in rows], dtype=float)
    plt.figure(figsize=(6.0, 4.2)); plt.hist(cosv, bins=min(12, max(5, len(cosv)//2)))
    plt.xlabel("Cosine similarity: neural gradient vs CBS pullback gradient"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(out / "gradient_cosine_hist.png", dpi=220); plt.close()

    axisv = np.asarray([r["cosine_axis"] for r in rows], dtype=float)
    plt.figure(figsize=(6.0, 4.2)); plt.scatter(np.arange(1, len(axisv)+1), axisv)
    plt.xlabel("Sample index"); plt.ylabel("|Cosine similarity|")
    plt.ylim(0, 1.02); plt.tight_layout(); plt.savefig(out / "gradient_axis_alignment.png", dpi=220); plt.close()

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved to", out.resolve())


if __name__ == "__main__":
    main()
