#!/usr/bin/env python3
"""
Precompute true CBS adjoint-gradient teachers for adjoint-aware surrogate fine-tuning.

Each cached sample corresponds to one local-alpha speed field c_alpha.
The inversion target is the alpha=1 measurement y_GT from the same base sample.

We use the same parameterization as E4:
    c_480(stored) -> bilinear downsample to image_size -> bilinear upsample to 480 -> CBS.

Saved teacher gradients:
    teacher_grad_raw_240:
        exact transpose-Jacobian pullback of the raw CBS adjoint gradient to image_size.
    teacher_grad_smooth_240:
        same, after smoothing the 480-grid CBS gradient with avg_pool.

The cache also stores the original local-alpha measurement target used by the existing
forward surrogate training, so fine-tuning can preserve forward accuracy while adding
gradient-direction supervision.
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from cbs_model import ConvergentBornSeries_Batch, ConvergentBornSeries_Batch_Adjoint


def numeric_key(path):
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


def resize_np(x, size):
    x_t = torch.from_numpy(x.astype(np.float32))[None, None]
    y_t = F.interpolate(
        x_t, size=(size, size), mode="bilinear", align_corners=False
    )
    return y_t[0, 0].cpu().numpy().astype(np.float32)


def pullback_bilinear_grad(g480, in_size, device):
    x = torch.zeros(
        1, 1, in_size, in_size,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    y = F.interpolate(x, size=(480, 480), mode="bilinear", align_corners=False)
    g = torch.from_numpy(g480.astype(np.float32))[None, None].to(device)
    torch.sum(y * g).backward()
    return x.grad.detach().cpu().numpy()[0, 0].astype(np.float32)


def complex_mse(pred, target):
    d = pred - target
    return float(np.mean(np.abs(d) ** 2))


def group_local_alpha(data_root, split):
    files = sorted(
        list((Path(data_root) / split).glob(f"{split}_*.npz")),
        key=numeric_key,
    )
    groups = defaultdict(list)
    for p in files:
        z = np.load(p, allow_pickle=True)
        groups[str(z["base_sample"])].append((float(z["alpha"]), p))
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: x[0])
    return dict(sorted(groups.items(), key=lambda kv: numeric_key(kv[0])))


@torch.no_grad()
def compute_cbs_teacher(
    c480,
    target_gt_dobs,
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
    pred_t = u[0, :, rec_t[:, 0], rec_t[:, 1]]
    pred = pred_t.detach().cpu().numpy().astype(np.complex64)

    target_t = torch.from_numpy(target_gt_dobs.astype(np.complex64))[None].to(device)
    mask = np.ones_like(np.abs(target_gt_dobs), dtype=np.float32)

    adj = ConvergentBornSeries_Batch_Adjoint(
        batch_model=model,
        rec_loc=rec_t,
        dobs_500k_batch=target_t,
        dobs_500k_mask=mask,
    )
    grad, reported_loss = adj(u, max_iters=args.adjoint_iters)
    g_raw = grad[0, 0].detach().cpu().numpy().astype(np.float32)

    if args.smooth_kernel > 1:
        k = int(args.smooth_kernel)
        g_smooth = F.avg_pool2d(
            grad, kernel_size=k, stride=1, padding=k // 2
        )[0, 0].detach().cpu().numpy().astype(np.float32)
    else:
        g_smooth = g_raw.copy()

    loss_mse = complex_mse(pred, target_gt_dobs)

    del model, u, adj, grad, sos
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return g_raw, g_smooth, pred, float(reported_loss.detach().cpu()), loss_mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        type=str,
        default="./self_consistent_cbs/sparse_64_local_alpha_train100_test20",
    )
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--output_root", type=str, required=True)

    ap.add_argument("--image_size", type=int, default=240)
    ap.add_argument("--max_base", type=int, default=-1)
    ap.add_argument(
        "--base_start",
        type=int,
        default=1,
        help="1-based inclusive base-sample position after numeric sorting.",
    )
    ap.add_argument(
        "--base_end",
        type=int,
        default=-1,
        help="1-based inclusive base-sample position after numeric sorting; -1 means through the end.",
    )
    ap.add_argument(
        "--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75]
    )

    ap.add_argument("--frequency", type=float, default=500000.0)
    ap.add_argument("--forward_iters", type=int, default=80)
    ap.add_argument("--adjoint_iters", type=int, default=80)
    ap.add_argument("--boundary_width", type=int, default=300)
    ap.add_argument("--boundary_strength", type=float, default=225.0)
    ap.add_argument("--boundary_type", type=str, default="PML3")
    ap.add_argument("--smooth_kernel", type=int, default=9)

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    groups_all = group_local_alpha(args.data_root, args.split)
    items = list(groups_all.items())

    start = max(int(args.base_start), 1)
    end = int(args.base_end)
    if end < 0:
        end = len(items)
    if end < start:
        raise ValueError(f"base_end ({end}) must be >= base_start ({start})")

    # 1-based inclusive slicing.
    items = items[start - 1 : end]

    if args.max_base > 0:
        items = items[: args.max_base]

    groups = dict(items)
    alpha_set = {round(float(a), 6) for a in args.alphas}

    manifest = []
    n_total = 0
    n_done = 0
    n_skipped = 0

    print("=" * 100)
    print("Precompute CBS adjoint teacher gradients")
    print("=" * 100)
    print("split =", args.split)
    print("num base =", len(groups))
    print("alphas =", sorted(alpha_set))
    print("image_size =", args.image_size)
    print("device =", device)
    print("=" * 100)

    for bi, (base, records) in enumerate(groups.items(), start=1):
        gt_candidates = [(a, p) for a, p in records if abs(a - 1.0) < 1e-6]
        if not gt_candidates:
            raise RuntimeError(f"Missing alpha=1.0 for base={base}")
        gt_path = gt_candidates[0][1]
        zgt = np.load(gt_path, allow_pickle=True)
        target_gt_dobs = zgt["dobs_complex"].astype(np.complex64)

        for alpha, p in records:
            if round(float(alpha), 6) not in alpha_set:
                continue

            n_total += 1
            stem = p.stem
            save_path = out / f"{stem}_adjoint_teacher.npz"

            if args.resume and save_path.exists():
                n_skipped += 1
                manifest.append({
                    "source_file": str(p),
                    "cache_file": str(save_path),
                    "base_sample": base,
                    "alpha": float(alpha),
                    "status": "skipped_existing",
                })
                continue

            z = np.load(p, allow_pickle=True)
            speed_480_stored = z["target_480"].astype(np.float32)
            point_dobs_original = z["dobs_complex"].astype(np.complex64)
            src_indices = z["src_indices"].astype(np.int64)
            rec_indices = z["rec_indices"].astype(np.int64)

            speed_240 = resize_np(speed_480_stored, args.image_size)
            speed_480_projected = resize_np(speed_240, 480)

            g_raw_480, g_smooth_480, pred_projected, reported_loss, true_mse = (
                compute_cbs_teacher(
                    speed_480_projected,
                    target_gt_dobs,
                    src_indices,
                    rec_indices,
                    args,
                    device,
                )
            )

            g_raw_240 = pullback_bilinear_grad(g_raw_480, args.image_size, device)
            g_smooth_240 = pullback_bilinear_grad(
                g_smooth_480, args.image_size, device
            )

            np.savez_compressed(
                save_path,
                speed_240=speed_240.astype(np.float32),
                point_dobs_original=point_dobs_original.astype(np.complex64),
                target_gt_dobs=target_gt_dobs.astype(np.complex64),
                projected_cbs_dobs=pred_projected.astype(np.complex64),
                teacher_grad_raw_240=g_raw_240.astype(np.float32),
                teacher_grad_smooth_240=g_smooth_240.astype(np.float32),
                src_indices=src_indices.astype(np.int64),
                rec_indices=rec_indices.astype(np.int64),
                alpha=np.array(float(alpha), dtype=np.float32),
                base_sample=np.array(base),
                source_file=np.array(str(p)),
                cbs_reported_loss=np.array(reported_loss, dtype=np.float64),
                true_cbs_mse_to_gt=np.array(true_mse, dtype=np.float64),
            )

            row = {
                "source_file": str(p),
                "cache_file": str(save_path),
                "base_sample": base,
                "alpha": float(alpha),
                "true_cbs_mse_to_gt": true_mse,
                "raw_grad_rms_240": float(np.sqrt(np.mean(g_raw_240.astype(np.float64) ** 2))),
                "smooth_grad_rms_240": float(np.sqrt(np.mean(g_smooth_240.astype(np.float64) ** 2))),
                "status": "generated",
            }
            manifest.append(row)
            n_done += 1

            print(
                f"[{bi:03d}/{len(groups):03d}] {base} alpha={alpha:.2f} | "
                f"MSE={true_mse:.6e} | "
                f"rawRMS={row['raw_grad_rms_240']:.4e} | "
                f"smoothRMS={row['smooth_grad_rms_240']:.4e}"
            )

    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    summary = {
        "num_selected_points": n_total,
        "num_generated": n_done,
        "num_skipped_existing": n_skipped,
        "num_base": len(groups),
        "base_start": int(args.base_start),
        "base_end": int(args.base_end),
        "alphas": sorted(alpha_set),
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[DONE]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("saved to:", out.resolve())


if __name__ == "__main__":
    main()
