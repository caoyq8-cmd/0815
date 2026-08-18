#!/usr/bin/env python3
"""
Directional finite-difference sanity check for:
  1) neural surrogate autograd gradient
  2) CBS adjoint gradient pulled back from 480x480 to the neural parameter grid

Purpose
-------
Before interpreting a low cosine between neural and CBS gradients as a model-quality
problem, verify that each gradient implementation is internally consistent with the
directional derivative of its own forward objective.

For random smooth directions v, compare:
    FD(v) = [L(c + eps v) - L(c - eps v)] / (2 eps)
with:
    <grad L(c), v>

For the neural surrogate, the absolute scaling should match (kappa ~ 1).
For CBS, a global scale factor may differ because the legacy CBS implementation uses
its own source/adjoint scaling and objective normalization. Therefore the primary CBS
checks are:
    - directional correlation/cosine near +1
    - sign agreement near 100%
    - small residual after fitting one global scale kappa

This script intentionally checks the RAW CBS adjoint gradient. The 9x9-smoothed
gradient used by the correction algorithm is a regularized update direction, not the
exact derivative, so it should not be expected to pass an exact finite-difference test.
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

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


def pullback_bilinear_grad(g480, in_size, device):
    """
    Exact transpose-Jacobian of:
        c_in -> bilinear_interpolate(c_in, 480x480)
    """
    x = torch.zeros(
        1, 1, in_size, in_size,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    y = F.interpolate(x, size=(480, 480), mode="bilinear", align_corners=False)
    g = torch.from_numpy(g480.astype(np.float32))[None, None].to(device)
    scalar = torch.sum(y * g)
    scalar.backward()
    return x.grad.detach().cpu().numpy()[0, 0].astype(np.float32)


def complex_mse(pred, target):
    d = pred - target
    return float(np.mean(np.abs(d) ** 2))


@torch.no_grad()
def cbs_forward(c480, src_indices, rec_indices, args, device):
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


@torch.no_grad()
def cbs_gradient(c480, target_dobs, src_indices, rec_indices, args, device):
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
    pred_t = u[0, :, rec_t[:, 0], rec_t[:, 1]]
    pred = pred_t.detach().cpu().numpy().astype(np.complex64)

    target_t = torch.from_numpy(target_dobs.astype(np.complex64))[None].to(device)
    mask = np.ones_like(np.abs(target_dobs), dtype=np.float32)

    adj = ConvergentBornSeries_Batch_Adjoint(
        batch_model=model,
        rec_loc=rec_t,
        dobs_500k_batch=target_t,
        dobs_500k_mask=mask,
    )
    grad, _ = adj(u, max_iters=args.adjoint_iters)
    g480 = grad[0, 0].detach().cpu().numpy().astype(np.float32)

    del model, u, adj, grad, sos
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return g480, pred


def make_smooth_direction(size, seed, smooth_kernel):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((size, size)).astype(np.float32)
    xt = torch.from_numpy(x)[None, None]

    if smooth_kernel > 1:
        k = int(smooth_kernel)
        if k % 2 == 0:
            k += 1
        xt = F.avg_pool2d(xt, kernel_size=k, stride=1, padding=k // 2)

    v = xt.numpy()[0, 0].astype(np.float64)
    v = v - np.mean(v)
    r = np.sqrt(np.mean(v * v)) + 1e-12
    v = v / r
    return v.astype(np.float32)


def neural_objective(
    model,
    speed_np,
    target_dobs,
    mean_real,
    mean_imag,
    residual_scale,
    speed_center,
    speed_scale,
    y_map,
    x_map,
    device,
):
    loss, _ = eval_neural_np(
        model=model,
        speed_np=speed_np.astype(np.float32),
        target_dobs=target_dobs,
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
    return float(loss)


def fit_directional_scale(ip, fd):
    """
    Fit fd ~= kappa * ip through the origin and report scale-free agreement.
    """
    ip = np.asarray(ip, dtype=np.float64)
    fd = np.asarray(fd, dtype=np.float64)

    denom = float(np.dot(ip, ip))
    if denom < 1e-30:
        kappa = np.nan
    else:
        kappa = float(np.dot(ip, fd) / denom)

    nip = np.linalg.norm(ip)
    nfd = np.linalg.norm(fd)
    if nip < 1e-30 or nfd < 1e-30:
        cos = np.nan
    else:
        cos = float(np.dot(ip, fd) / (nip * nfd))

    if np.isfinite(kappa) and nfd > 1e-30:
        rel_resid = float(np.linalg.norm(fd - kappa * ip) / nfd)
    else:
        rel_resid = np.nan

    # Ignore almost-zero finite differences for sign statistics.
    threshold = max(1e-14, 1e-5 * float(np.max(np.abs(fd))) if len(fd) else 1e-14)
    mask = np.abs(fd) > threshold
    if np.any(mask):
        sign_agreement = float(np.mean(np.sign(fd[mask]) == np.sign(ip[mask])))
        n_sign = int(np.sum(mask))
    else:
        sign_agreement = np.nan
        n_sign = 0

    return {
        "kappa": kappa,
        "directional_cosine": cos,
        "relative_residual_after_scale": rel_resid,
        "sign_agreement": sign_agreement,
        "n_significant": n_sign,
    }


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        type=str,
        default="./self_consistent_cbs/sparse_64_local_alpha_train100_test20",
    )
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument(
        "--mean_dobs_path",
        type=str,
        default="./dobs_mean_baseline/local_alpha_train100_test20/mean_dobs_train.npz",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="./gradient_consistency_runs/directional_fd_check",
    )

    ap.add_argument("--max_base", type=int, default=1)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 0.75])
    ap.add_argument("--num_directions", type=int, default=5)
    ap.add_argument("--direction_smooth_kernel", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--epsilons_mps", type=float, nargs="+", default=[0.1, 0.25, 0.5])

    ap.add_argument("--frequency", type=float, default=500000.0)
    ap.add_argument("--forward_iters", type=int, default=80)
    ap.add_argument("--adjoint_iters", type=int, default=80)
    ap.add_argument("--boundary_width", type=int, default=300)
    ap.add_argument("--boundary_strength", type=float, default=225.0)
    ap.add_argument("--boundary_type", type=str, default="PML3")

    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w", encoding="utf-8") as f:
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

    alpha_set = {round(float(a), 6) for a in args.alphas}

    print("=" * 100)
    print("Directional finite-difference gradient sanity check")
    print("=" * 100)
    print("device:", device)
    print("image_size:", image_size)
    print("num base samples:", len(groups))
    print("alphas:", sorted(alpha_set))
    print("directions per point:", args.num_directions)
    print("epsilons [m/s]:", args.epsilons_mps)
    print("=" * 100)

    detail_rows = []
    point_summaries = []

    for base_idx, (base, records) in enumerate(groups.items(), start=1):
        gt_path = find_gt_record(records)
        zgt = np.load(gt_path, allow_pickle=True)
        target_dobs = zgt["dobs_complex"].astype(np.complex64)

        for alpha, path in records:
            if round(float(alpha), 6) not in alpha_set:
                continue

            z = np.load(path, allow_pickle=True)
            c480_stored = z["target_480"].astype(np.float32)
            src_indices = z["src_indices"].astype(np.int64)
            rec_indices = z["rec_indices"].astype(np.int64)

            # Same 240-parameterization used in E4.
            c = resize_np(c480_stored, image_size).astype(np.float32)
            c480 = resize_np(c, 480).astype(np.float32)

            # Neural gradient of its own MSE objective.
            neural_loss, g_nn, _ = compute_neural_grad(
                model=model,
                speed_np=c,
                target_dobs=target_dobs,
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
            g_nn = g_nn.astype(np.float64)

            # Raw CBS adjoint gradient, then exact pullback to the 240 parameter grid.
            g480, pred0 = cbs_gradient(
                c480,
                target_dobs,
                src_indices,
                rec_indices,
                args,
                device,
            )
            g_cbs = pullback_bilinear_grad(g480, image_size, device).astype(np.float64)
            cbs_loss0 = 0.5 * complex_mse(pred0, target_dobs)

            dirs = [
                make_smooth_direction(
                    image_size,
                    args.seed + 100000 * base_idx + 1000 * int(round(alpha * 100)) + d,
                    args.direction_smooth_kernel,
                ).astype(np.float64)
                for d in range(args.num_directions)
            ]

            for eps in args.epsilons_mps:
                nn_ips, nn_fds = [], []
                cbs_ips, cbs_fds = [], []

                for d, v in enumerate(dirs):
                    cp = (c.astype(np.float64) + eps * v).astype(np.float32)
                    cm = (c.astype(np.float64) - eps * v).astype(np.float32)

                    # Neural finite difference.
                    nlp = neural_objective(
                        model, cp, target_dobs,
                        mean_real, mean_imag, residual_scale,
                        speed_center, speed_scale, y_map, x_map, device,
                    )
                    nlm = neural_objective(
                        model, cm, target_dobs,
                        mean_real, mean_imag, residual_scale,
                        speed_center, speed_scale, y_map, x_map, device,
                    )
                    nn_fd = (nlp - nlm) / (2.0 * eps)
                    nn_ip = float(np.sum(g_nn * v))

                    # CBS finite difference of 0.5 * mean |A(c)-y|^2.
                    cp480 = resize_np(cp, 480).astype(np.float32)
                    cm480 = resize_np(cm, 480).astype(np.float32)

                    pp = cbs_forward(cp480, src_indices, rec_indices, args, device)
                    pm = cbs_forward(cm480, src_indices, rec_indices, args, device)

                    clp = 0.5 * complex_mse(pp, target_dobs)
                    clm = 0.5 * complex_mse(pm, target_dobs)
                    cbs_fd = (clp - clm) / (2.0 * eps)
                    cbs_ip = float(np.sum(g_cbs * v))

                    nn_ips.append(nn_ip)
                    nn_fds.append(nn_fd)
                    cbs_ips.append(cbs_ip)
                    cbs_fds.append(cbs_fd)

                    detail_rows.append({
                        "base_sample": base,
                        "alpha": float(alpha),
                        "epsilon_mps": float(eps),
                        "direction_index": d,
                        "neural_loss_center": float(neural_loss),
                        "cbs_loss_center_half_mse": float(cbs_loss0),
                        "nn_inner_product": nn_ip,
                        "nn_finite_difference": nn_fd,
                        "cbs_inner_product": cbs_ip,
                        "cbs_finite_difference": cbs_fd,
                    })

                nn_stat = fit_directional_scale(nn_ips, nn_fds)
                cbs_stat = fit_directional_scale(cbs_ips, cbs_fds)

                summary = {
                    "base_sample": base,
                    "alpha": float(alpha),
                    "epsilon_mps": float(eps),
                    "num_directions": args.num_directions,

                    "nn_kappa_fd_over_ip": nn_stat["kappa"],
                    "nn_directional_cosine": nn_stat["directional_cosine"],
                    "nn_rel_residual_after_scale": nn_stat["relative_residual_after_scale"],
                    "nn_sign_agreement": nn_stat["sign_agreement"],

                    "cbs_kappa_fd_over_ip": cbs_stat["kappa"],
                    "cbs_directional_cosine": cbs_stat["directional_cosine"],
                    "cbs_rel_residual_after_scale": cbs_stat["relative_residual_after_scale"],
                    "cbs_sign_agreement": cbs_stat["sign_agreement"],
                }
                point_summaries.append(summary)

                print(
                    f"[{base_idx:02d}/{len(groups):02d}] {base} alpha={alpha:.2f} eps={eps:g} | "
                    f"NN cos={nn_stat['directional_cosine']:.5f} "
                    f"k={nn_stat['kappa']:.5g} "
                    f"res={nn_stat['relative_residual_after_scale']:.3e} "
                    f"sign={nn_stat['sign_agreement']:.2f} | "
                    f"CBS cos={cbs_stat['directional_cosine']:.5f} "
                    f"k={cbs_stat['kappa']:.5g} "
                    f"res={cbs_stat['relative_residual_after_scale']:.3e} "
                    f"sign={cbs_stat['sign_agreement']:.2f}"
                )

    write_csv(out / "fd_detail.csv", detail_rows)
    write_csv(out / "fd_summary.csv", point_summaries)

    def vals(key):
        x = [float(r[key]) for r in point_summaries if np.isfinite(float(r[key]))]
        return x

    overall = {
        "num_point_epsilon_checks": len(point_summaries),
        "num_detail_rows": len(detail_rows),
        "neural_directional_cosine_mean": float(np.mean(vals("nn_directional_cosine"))),
        "neural_directional_cosine_min": float(np.min(vals("nn_directional_cosine"))),
        "neural_rel_residual_after_scale_mean": float(np.mean(vals("nn_rel_residual_after_scale"))),
        "neural_sign_agreement_mean": float(np.mean(vals("nn_sign_agreement"))),

        "cbs_directional_cosine_mean": float(np.mean(vals("cbs_directional_cosine"))),
        "cbs_directional_cosine_min": float(np.min(vals("cbs_directional_cosine"))),
        "cbs_rel_residual_after_scale_mean": float(np.mean(vals("cbs_rel_residual_after_scale"))),
        "cbs_sign_agreement_mean": float(np.mean(vals("cbs_sign_agreement"))),

        "interpretation": {
            "neural": "Expect directional cosine near +1, sign agreement near 1, residual small, and kappa close to 1 if eval_neural_np and compute_neural_grad use exactly the same MSE normalization.",
            "cbs": "Expect directional cosine near +1 and sign agreement near 1. The fitted kappa need not equal 1 because the legacy CBS adjoint uses source/adjoint scaling and may correspond to a differently normalized least-squares objective.",
        },
    }

    with open(out / "fd_overall.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("[DONE]")
    print("=" * 100)
    print(json.dumps(overall, indent=2, ensure_ascii=False))
    print("saved to:", os.path.abspath(out))


if __name__ == "__main__":
    main()
