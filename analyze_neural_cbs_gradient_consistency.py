#!/usr/bin/env python3
import os
import re
import csv
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from cbs_model import ConvergentBornSeries_Batch, ConvergentBornSeries_Batch_Adjoint
from run_neural_measurement_correction import (
    load_residual_measurement_model,
    compute_neural_grad,
    eval_neural_np,
    make_coord_maps,
    resize_np,
)


def numeric_key(path):
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


def complex_rrmse(pred, target):
    num = np.sqrt(np.mean(np.abs(pred - target) ** 2))
    den = np.sqrt(np.mean(np.abs(target) ** 2)) + 1e-12
    return float(num / den)


def complex_mae(pred, target):
    return float(np.mean(np.abs(pred - target)))


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)) + 1e-12)


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-20 or nb < 1e-20:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def normalize_rms(g):
    return g / rms(g)


def pullback_bilinear_grad(g480, in_size, device):
    """
    Exact transpose-Jacobian of:
        c_in -> F.interpolate(c_in, size=(480,480), mode='bilinear', align_corners=False)

    This is more correct than simply resizing a 480-grid gradient down to the neural grid.
    """
    x = torch.zeros(1, 1, in_size, in_size, device=device, dtype=torch.float32, requires_grad=True)
    y = F.interpolate(x, size=(480, 480), mode="bilinear", align_corners=False)
    g = torch.from_numpy(g480.astype(np.float32))[None, None].to(device)
    scalar = torch.sum(y * g)
    scalar.backward()
    out = x.grad.detach().cpu().numpy()[0, 0].astype(np.float32)
    return out


@torch.no_grad()
def compute_cbs_grad_and_forward(
    c480,
    target_dobs,
    src_indices,
    rec_indices,
    args,
    device,
):
    sos = torch.from_numpy(c480.astype(np.float32))[None, None].to(device)

    model = ConvergentBornSeries_Batch(
        f=args.frequency,
        sos=sos,
        boundary_width=[args.boundary_width, args.boundary_width],
        boundary_strength=args.boundary_strength,
        boundary_type=args.boundary_type,
        src_loc_set=src_indices.astype(np.int64),
        device=device,
    )

    u = model(max_iters=args.forward_iters)

    rec_t = torch.from_numpy(rec_indices.astype(np.int64)).long().to(device)
    dobs_pred_t = u[0, :, rec_t[:, 0], rec_t[:, 1]]
    dobs_pred = dobs_pred_t.detach().cpu().numpy().astype(np.complex64)

    target_t = torch.from_numpy(target_dobs.astype(np.complex64))[None].to(device)
    mask = np.ones_like(np.abs(target_dobs), dtype=np.float32)

    adj = ConvergentBornSeries_Batch_Adjoint(
        batch_model=model,
        rec_loc=rec_t,
        dobs_500k_batch=target_t,
        dobs_500k_mask=mask,
    )

    grad, reported_loss = adj(u, max_iters=args.adjoint_iters)
    grad_raw = grad[0, 0].detach().cpu().numpy().astype(np.float32)

    if args.smooth_kernel > 1:
        k = args.smooth_kernel
        grad_smooth_t = F.avg_pool2d(grad, kernel_size=k, stride=1, padding=k // 2)
        grad_smooth = grad_smooth_t[0, 0].detach().cpu().numpy().astype(np.float32)
    else:
        grad_smooth = grad_raw.copy()

    loss_abs = complex_mae(dobs_pred, target_dobs)

    del model, u, adj, grad, sos
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return grad_raw, grad_smooth, float(reported_loss.detach().cpu()), loss_abs, dobs_pred


@torch.no_grad()
def forward_cbs_dobs(c480, src_indices, rec_indices, args, device):
    sos = torch.from_numpy(c480.astype(np.float32))[None, None].to(device)
    model = ConvergentBornSeries_Batch(
        f=args.frequency,
        sos=sos,
        boundary_width=[args.boundary_width, args.boundary_width],
        boundary_strength=args.boundary_strength,
        boundary_type=args.boundary_type,
        src_loc_set=src_indices.astype(np.int64),
        device=device,
    )
    u = model(max_iters=args.forward_iters)
    rec_t = torch.from_numpy(rec_indices.astype(np.int64)).long().to(device)
    dobs = u[0, :, rec_t[:, 0], rec_t[:, 1]]
    out = dobs.detach().cpu().numpy().astype(np.complex64)
    del model, u, sos
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def load_groups(data_root, split):
    files = sorted(
        list((Path(data_root) / split).glob(f"{split}_*.npz")),
        key=numeric_key,
    )
    groups = defaultdict(list)
    for p in files:
        z = np.load(p, allow_pickle=True)
        base = str(z["base_sample"])
        alpha = float(z["alpha"])
        groups[base].append((alpha, p))
    for base in groups:
        groups[base] = sorted(groups[base], key=lambda x: x[0])
    return dict(sorted(groups.items(), key=lambda kv: numeric_key(kv[0])))


def find_gt_record(records):
    for alpha, p in records:
        if abs(alpha - 1.0) < 1e-6:
            return p
    raise RuntimeError("Each base sample must contain alpha=1.0 to define y_GT.")


def summarize(rows, key, valid_only=True):
    vals = []
    for r in rows:
        v = r.get(key, None)
        if v is None:
            continue
        try:
            v = float(v)
        except Exception:
            continue
        if valid_only and not np.isfinite(v):
            continue
        vals.append(v)
    if not vals:
        return {"n": 0, "mean": None, "std": None}
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
    }


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare neural measurement-surrogate gradients with true CBS adjoint gradients."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./self_consistent_cbs/sparse_64_local_alpha_train100_test20",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument(
        "--mean_dobs_path",
        type=str,
        default="./dobs_mean_baseline/local_alpha_train100_test20/mean_dobs_train.npz",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gradient_consistency_runs/pairwise_delta2_test20",
    )
    parser.add_argument("--max_base", type=int, default=5)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )

    # IMPORTANT:
    # CBS adjoint is driven by the measurement residual and corresponds to an L2-type
    # data fidelity gradient. To make the neural gradient comparison meaningful,
    # default to MSE-only here, regardless of the surrogate's training loss mix.
    parser.add_argument("--neural_lambda_l1", type=float, default=0.0)
    parser.add_argument("--neural_lambda_mse", type=float, default=1.0)

    parser.add_argument("--frequency", type=float, default=500000.0)
    parser.add_argument("--forward_iters", type=int, default=80)
    parser.add_argument("--adjoint_iters", type=int, default=80)
    parser.add_argument("--boundary_width", type=int, default=300)
    parser.add_argument("--boundary_strength", type=float, default=225.0)
    parser.add_argument("--boundary_type", type=str, default="PML3")
    parser.add_argument("--smooth_kernel", type=int, default=9)

    parser.add_argument("--speed_min", type=float, default=1400.0)
    parser.add_argument("--speed_max", type=float, default=1605.0)

    parser.add_argument("--descent_test", action="store_true")
    parser.add_argument("--descent_step_mps", type=float, default=0.5)
    parser.add_argument(
        "--descent_factors",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.25, 0.1],
    )

    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

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

    groups = load_groups(args.data_root, args.split)
    if args.max_base > 0:
        groups = dict(list(groups.items())[: args.max_base])

    selected_alphas = set(round(float(a), 6) for a in args.alphas)

    print("=" * 100)
    print("E4 neural-vs-CBS gradient consistency")
    print("=" * 100)
    print("device =", device)
    print("image_size =", image_size)
    print("num base samples =", len(groups))
    print("alphas =", sorted(selected_alphas))
    print("neural objective: lambda_l1 =", args.neural_lambda_l1,
          "lambda_mse =", args.neural_lambda_mse)
    print("CBS smooth kernel =", args.smooth_kernel)
    print("descent_test =", args.descent_test)
    print("=" * 100)

    rows = []

    for base_i, (base, records) in enumerate(groups.items(), start=1):
        gt_path = find_gt_record(records)
        gt = np.load(gt_path, allow_pickle=True)
        target_dobs = gt["dobs_complex"].astype(np.complex64)

        for alpha, path in records:
            if round(float(alpha), 6) not in selected_alphas:
                continue

            z = np.load(path, allow_pickle=True)
            local_480_stored = z["target_480"].astype(np.float32)
            src_indices = z["src_indices"].astype(np.int64)
            rec_indices = z["rec_indices"].astype(np.int64)

            # Neural model parameterization is image_size x image_size.
            c_neural = resize_np(local_480_stored, image_size)
            # True CBS evaluation must follow the same neural-grid -> physics-grid map
            # used when a neural proposal is verified by CBS.
            c480 = resize_np(c_neural, 480)
            c480 = np.clip(c480, args.speed_min, args.speed_max).astype(np.float32)

            # True CBS gradient of data-fidelity to y_GT.
            g_cbs_raw_480, g_cbs_smooth_480, reported_loss, true_abs_loss, true_forward = (
                compute_cbs_grad_and_forward(
                    c480,
                    target_dobs,
                    src_indices,
                    rec_indices,
                    args,
                    device,
                )
            )

            # Surrogate forward prediction at the same neural-grid representation.
            _, neural_forward = eval_neural_np(
                model=model,
                speed_np=c_neural,
                target_dobs=true_forward,  # only used to form a loss; prediction is what we need
                mean_real=mean_real,
                mean_imag=mean_imag,
                residual_scale=residual_scale,
                speed_center=speed_center,
                speed_scale=speed_scale,
                y_map=y_map,
                x_map=x_map,
                device=device,
                lambda_l1=0.0,
                lambda_mse=1.0,
            )

            forward_rrmse = complex_rrmse(neural_forward, true_forward)
            forward_mae = complex_mae(neural_forward, true_forward)

            # Neural data-fidelity gradient to the same target y_GT.
            neural_loss, g_nn, _ = compute_neural_grad(
                model=model,
                speed_np=c_neural,
                target_dobs=target_dobs,
                mean_real=mean_real,
                mean_imag=mean_imag,
                residual_scale=residual_scale,
                speed_center=speed_center,
                speed_scale=speed_scale,
                y_map=y_map,
                x_map=x_map,
                device=device,
                lambda_l1=args.neural_lambda_l1,
                lambda_mse=args.neural_lambda_mse,
            )

            # Put CBS gradient into the SAME neural parameter space using J_interp^T.
            g_cbs_raw_eff = pullback_bilinear_grad(
                g_cbs_raw_480,
                in_size=image_size,
                device=device,
            )
            g_cbs_smooth_eff = pullback_bilinear_grad(
                g_cbs_smooth_480,
                in_size=image_size,
                device=device,
            )

            cos_raw = cosine(g_nn, g_cbs_raw_eff)
            cos_smooth = cosine(g_nn, g_cbs_smooth_eff)

            nn_rms = rms(g_nn)
            cbs_raw_rms = rms(g_cbs_raw_eff)
            cbs_smooth_rms = rms(g_cbs_smooth_eff)

            row = {
                "base_sample": base,
                "alpha": float(alpha),
                "file": str(path),
                "true_cbs_abs_loss_to_gt": true_abs_loss,
                "cbs_reported_loss": reported_loss,
                "neural_loss_to_gt_mse_only": float(neural_loss),
                "forward_rrmse_neural_vs_true_cbs": forward_rrmse,
                "forward_mae_neural_vs_true_cbs": forward_mae,
                "grad_cos_raw": cos_raw,
                "grad_cos_smooth": cos_smooth,
                "grad_nn_rms": nn_rms,
                "grad_cbs_raw_eff_rms": cbs_raw_rms,
                "grad_cbs_smooth_eff_rms": cbs_smooth_rms,
                "grad_rms_ratio_nn_over_cbs_raw": float(nn_rms / cbs_raw_rms),
                "grad_rms_ratio_nn_over_cbs_smooth": float(nn_rms / cbs_smooth_rms),
            }

            if args.descent_test:
                gnn = normalize_rms(g_nn)
                gcbs = normalize_rms(g_cbs_raw_eff)

                best_nn = true_abs_loss
                best_nn_factor = 0.0
                best_cbs = true_abs_loss
                best_cbs_factor = 0.0

                for fac in args.descent_factors:
                    step = args.descent_step_mps * float(fac)

                    cand_nn = np.clip(
                        c_neural - step * gnn,
                        args.speed_min,
                        args.speed_max,
                    ).astype(np.float32)
                    cand_nn_480 = resize_np(cand_nn, 480)
                    pred_nn = forward_cbs_dobs(
                        cand_nn_480,
                        src_indices,
                        rec_indices,
                        args,
                        device,
                    )
                    loss_nn = complex_mae(pred_nn, target_dobs)
                    if loss_nn < best_nn:
                        best_nn = loss_nn
                        best_nn_factor = float(fac)

                    cand_cbs = np.clip(
                        c_neural - step * gcbs,
                        args.speed_min,
                        args.speed_max,
                    ).astype(np.float32)
                    cand_cbs_480 = resize_np(cand_cbs, 480)
                    pred_cbs = forward_cbs_dobs(
                        cand_cbs_480,
                        src_indices,
                        rec_indices,
                        args,
                        device,
                    )
                    loss_cbs = complex_mae(pred_cbs, target_dobs)
                    if loss_cbs < best_cbs:
                        best_cbs = loss_cbs
                        best_cbs_factor = float(fac)

                row.update({
                    "nn_descent_success": int(best_nn < true_abs_loss - 1e-12),
                    "nn_best_true_cbs_abs_loss": float(best_nn),
                    "nn_best_true_cbs_rel_reduction": float(
                        (true_abs_loss - best_nn) / (true_abs_loss + 1e-12)
                    ),
                    "nn_best_factor": best_nn_factor,
                    "cbs_descent_success": int(best_cbs < true_abs_loss - 1e-12),
                    "cbs_best_true_cbs_abs_loss": float(best_cbs),
                    "cbs_best_true_cbs_rel_reduction": float(
                        (true_abs_loss - best_cbs) / (true_abs_loss + 1e-12)
                    ),
                    "cbs_best_factor": best_cbs_factor,
                })

            rows.append(row)

            print(
                f"[{base_i:02d}/{len(groups):02d}] {base} alpha={alpha:.2f} | "
                f"FwdRRMSE={forward_rrmse:.5f} | "
                f"cos_raw={cos_raw:.4f} | cos_smooth={cos_smooth:.4f}"
                + (
                    f" | nn_desc={row['nn_descent_success']} "
                    f"cbs_desc={row['cbs_descent_success']}"
                    if args.descent_test else ""
                )
            )

    # Save detailed rows.
    write_csv(Path(args.output_dir) / "gradient_rows.csv", rows)
    with open(Path(args.output_dir) / "gradient_rows.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # Summary per alpha.
    by_alpha = defaultdict(list)
    for r in rows:
        by_alpha[float(r["alpha"])].append(r)

    summary_rows = []
    for alpha in sorted(by_alpha.keys()):
        rr = by_alpha[alpha]
        s = {
            "alpha": alpha,
            "n": len(rr),
        }
        for key in [
            "true_cbs_abs_loss_to_gt",
            "forward_rrmse_neural_vs_true_cbs",
            "forward_mae_neural_vs_true_cbs",
            "grad_cos_raw",
            "grad_cos_smooth",
            "grad_rms_ratio_nn_over_cbs_raw",
            "grad_rms_ratio_nn_over_cbs_smooth",
        ]:
            stat = summarize(rr, key)
            s[key + "_mean"] = stat["mean"]
            s[key + "_std"] = stat["std"]

        if args.descent_test:
            s["nn_descent_success_rate"] = float(np.mean([r["nn_descent_success"] for r in rr]))
            s["cbs_descent_success_rate"] = float(np.mean([r["cbs_descent_success"] for r in rr]))
            s["nn_best_true_cbs_rel_reduction_mean"] = float(
                np.mean([r["nn_best_true_cbs_rel_reduction"] for r in rr])
            )
            s["cbs_best_true_cbs_rel_reduction_mean"] = float(
                np.mean([r["cbs_best_true_cbs_rel_reduction"] for r in rr])
            )

        summary_rows.append(s)

    write_csv(Path(args.output_dir) / "summary_by_alpha.csv", summary_rows)
    with open(Path(args.output_dir) / "summary_by_alpha.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    overall = {
        "num_rows": len(rows),
        "num_base": len(groups),
        "forward_rrmse": summarize(rows, "forward_rrmse_neural_vs_true_cbs"),
        "grad_cos_raw": summarize(rows, "grad_cos_raw"),
        "grad_cos_smooth": summarize(rows, "grad_cos_smooth"),
        "grad_rms_ratio_nn_over_cbs_raw": summarize(rows, "grad_rms_ratio_nn_over_cbs_raw"),
        "grad_rms_ratio_nn_over_cbs_smooth": summarize(rows, "grad_rms_ratio_nn_over_cbs_smooth"),
    }
    if args.descent_test and rows:
        overall["nn_descent_success_rate"] = float(
            np.mean([r["nn_descent_success"] for r in rows])
        )
        overall["cbs_descent_success_rate"] = float(
            np.mean([r["cbs_descent_success"] for r in rows])
        )

    with open(Path(args.output_dir) / "summary_overall.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    # Plot 1: forward error vs gradient cosine.
    xs = [r["forward_rrmse_neural_vs_true_cbs"] for r in rows]
    ys = [r["grad_cos_raw"] for r in rows]
    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys, s=28)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Forward RRMSE: neural vs true CBS")
    plt.ylabel("Gradient cosine: neural vs CBS")
    plt.title("Forward accuracy vs gradient alignment")
    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "scatter_forward_rrmse_vs_grad_cos.png", dpi=180)
    plt.close()

    # Plot 2: mean cosine by alpha.
    alphas = [r["alpha"] for r in summary_rows]
    cos_means = [r["grad_cos_raw_mean"] for r in summary_rows]
    cos_stds = [r["grad_cos_raw_std"] for r in summary_rows]
    plt.figure(figsize=(6, 4))
    plt.errorbar(alphas, cos_means, yerr=cos_stds, marker="o", capsize=4)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("alpha")
    plt.ylabel("Gradient cosine")
    plt.title("Gradient alignment along local-alpha path")
    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "grad_cos_by_alpha.png", dpi=180)
    plt.close()

    print("\n" + "=" * 100)
    print("[DONE]")
    print("=" * 100)
    print(json.dumps(overall, indent=2, ensure_ascii=False))
    print("saved to:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
