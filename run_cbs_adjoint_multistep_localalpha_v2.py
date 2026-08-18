#!/usr/bin/env python3
"""
Matched full-CBS multi-step baseline on the local-alpha sparse-64 dataset.

This script mirrors run_adjoint_aware_multistep_correction.py as closely as possible,
except the update direction is obtained from the true CBS adjoint rather than a neural
surrogate. It is intended for an apples-to-apples comparison:

  Adjoint-aware hybrid:
      neural gradient + CBS candidate verification

  Full-CBS baseline:
      CBS adjoint gradient + CBS candidate verification

Both use:
  - the same alpha=0 initialization / alpha=1 GT measurement
  - the same 240 -> 480 parameterization
  - RMS-normalized update directions
  - the same multi-scale factors
  - the same true-CBS candidate-selection objective
  - the same no-update candidate / optional opposite-sign fallback
"""

import os
import re
import csv
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from cbs_model import ConvergentBornSeries_Batch, ConvergentBornSeries_Batch_Adjoint

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None


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


def find_alpha_record(records, target_alpha):
    best = None
    for alpha, p in records:
        d = abs(float(alpha) - float(target_alpha))
        if best is None or d < best[0]:
            best = (d, alpha, p)
    if best is None or best[0] > 1e-6:
        raise RuntimeError(f"Could not find alpha={target_alpha}")
    return best[2]


def resize_np(x, size):
    xt = torch.from_numpy(np.asarray(x, np.float32))[None, None]
    yt = F.interpolate(xt, size=(size, size), mode="bilinear", align_corners=False)
    return yt[0, 0].cpu().numpy().astype(np.float32)


def pullback_bilinear_grad(g480, in_size, device):
    # cbs_adjoint_gradient_240() is intentionally wrapped in @torch.no_grad()
    # to avoid building a graph through the expensive CBS solver.  The exact
    # 240->480 interpolation pullback, however, does require autograd.  Re-enable
    # gradients only for this tiny interpolation operation.
    with torch.enable_grad():
        x = torch.zeros(
            1, 1, in_size, in_size,
            dtype=torch.float32, device=device, requires_grad=True
        )
        y = F.interpolate(x, size=(480, 480), mode="bilinear", align_corners=False)
        g = torch.from_numpy(g480.astype(np.float32))[None, None].to(device)
        scalar = torch.sum(y * g)
        grad_x = torch.autograd.grad(
            scalar,
            x,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
    return grad_x.detach().cpu().numpy()[0, 0].astype(np.float32)


def complex_mse(pred, target):
    d = pred - target
    return float(np.mean(np.abs(d) ** 2))


def complex_mae(pred, target):
    return float(np.mean(np.abs(pred - target)))


def complex_rrmse(pred, target):
    num = np.sqrt(np.mean(np.abs(pred - target) ** 2))
    den = np.sqrt(np.mean(np.abs(target) ** 2)) + 1e-12
    return float(num / den)


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)) + 1e-12)


def normalize_rms(g):
    return np.asarray(g, np.float32) / rms(g)


def image_metrics(pred, target, data_range=205.0):
    pred = np.asarray(pred, np.float32)
    target = np.asarray(target, np.float32)
    d = pred - target
    mse = float(np.mean(d * d))
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(mse))
    psnr = float(10.0 * np.log10((data_range * data_range) / (mse + 1e-12)))
    if skimage_ssim is not None:
        try:
            ssim = float(skimage_ssim(target, pred, data_range=data_range))
        except Exception:
            ssim = float("nan")
    else:
        ssim = float("nan")
    return {"mse": mse, "mae": mae, "rmse": rmse, "psnr": psnr, "ssim": ssim}


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
    out = u[0, :, rec_t[:, 0], rec_t[:, 1]].detach().cpu().numpy().astype(np.complex64)
    del model, u, sos
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def cbs_adjoint_gradient_240(c240, target_dobs, src_indices, rec_indices, args, device):
    """
    Return the RAW or 480-smoothed CBS adjoint gradient pulled back exactly to the
    240 parameter grid. This recomputes the current forward wavefield, so one call
    counts as one CBS forward + one CBS adjoint.
    """
    c480 = resize_np(c240, 480).astype(np.float32)
    sos = torch.from_numpy(c480)[None, None].to(device)

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
    target_t = torch.from_numpy(target_dobs.astype(np.complex64))[None].to(device)
    mask = np.ones_like(np.abs(target_dobs), dtype=np.float32)

    adj = ConvergentBornSeries_Batch_Adjoint(
        batch_model=model,
        rec_loc=rec_t,
        dobs_500k_batch=target_t,
        dobs_500k_mask=mask,
    )
    grad, _ = adj(u, max_iters=args.adjoint_iters)

    if args.smooth_kernel > 1:
        k = int(args.smooth_kernel)
        grad_use = F.avg_pool2d(
            grad, kernel_size=k, stride=1, padding=k // 2
        )
    else:
        grad_use = grad

    g480 = grad_use[0, 0].detach().cpu().numpy().astype(np.float32)
    g240 = pullback_bilinear_grad(g480, args.image_size, device)

    del model, u, adj, grad, grad_use, sos
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return g240


def write_csv(path, rows):
    if not rows:
        return
    keys, seen = [], set()
    for row in rows:
        for k in row:
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
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--max_base", type=int, default=5)
    ap.add_argument("--init_alpha", type=float, default=0.0)
    ap.add_argument("--gt_alpha", type=float, default=1.0)

    ap.add_argument("--image_size", type=int, default=240)
    ap.add_argument("--num_steps", type=int, default=8)
    ap.add_argument("--base_step_mps", type=float, default=0.5)
    ap.add_argument("--step_factors", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1])
    ap.add_argument("--speed_min", type=float, default=1400.0)
    ap.add_argument("--speed_max", type=float, default=1605.0)
    ap.add_argument("--smooth_kernel", type=int, default=9)

    ap.add_argument("--fallback_opposite", action="store_true")
    ap.add_argument("--stop_on_no_update", action="store_true")

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

    groups = load_groups(args.data_root, args.split)
    if args.max_base > 0:
        groups = dict(list(groups.items())[: args.max_base])

    print("=" * 110)
    print("Matched full-CBS multi-step baseline")
    print("=" * 110)
    print("num bases =", len(groups))
    print("image_size =", args.image_size)
    print("smooth_kernel =", args.smooth_kernel)
    print("num_steps =", args.num_steps)
    print("base_step_mps =", args.base_step_mps)
    print("step_factors =", args.step_factors)
    print("=" * 110)

    sample_rows = []

    for bi, (base, records) in enumerate(groups.items(), start=1):
        t0 = time.time()

        init_path = find_alpha_record(records, args.init_alpha)
        gt_path = find_alpha_record(records, args.gt_alpha)

        zi = np.load(init_path, allow_pickle=True)
        zg = np.load(gt_path, allow_pickle=True)

        init_480_stored = zi["target_480"].astype(np.float32)
        gt_480 = zg["target_480"].astype(np.float32)
        target_dobs = zg["dobs_complex"].astype(np.complex64)
        src_indices = zi["src_indices"].astype(np.int64)
        rec_indices = zi["rec_indices"].astype(np.int64)

        c = resize_np(init_480_stored, args.image_size).astype(np.float32)
        gt_240 = resize_np(gt_480, args.image_size).astype(np.float32)

        c480 = resize_np(c, 480)
        pred_true = forward_cbs_dobs(c480, src_indices, rec_indices, args, device)

        # Accounting:
        # initial true forward = 1.
        # each CBS adjoint gradient call adds 1 current forward + 1 adjoint.
        # each candidate verification adds 1 forward.
        cbs_forward_calls_algorithm = 1
        cbs_adjoint_calls = 0

        current_true_mse = complex_mse(pred_true, target_dobs)
        current_true_mae = complex_mae(pred_true, target_dobs)
        current_true_rrmse = complex_rrmse(pred_true, target_dobs)

        init_m240 = image_metrics(c, gt_240)
        init_m480 = image_metrics(c480, gt_480)

        history = [{
            "iter": 0,
            "selected_direction": "init",
            "selected_factor": 0.0,
            "true_cbs_mse": current_true_mse,
            "true_cbs_mae": current_true_mae,
            "true_cbs_rrmse": current_true_rrmse,
            "image_mse_240": init_m240["mse"],
            "image_mae_240": init_m240["mae"],
            "image_psnr_240": init_m240["psnr"],
            "image_ssim_240": init_m240["ssim"],
            "candidate_records": [],
        }]

        print(
            f"[{bi:02d}/{len(groups):02d}] {base} INIT | "
            f"CBS-MSE={current_true_mse:.6e} | imgMSE240={init_m240['mse']:.3f}"
        )

        for it in range(1, args.num_steps + 1):
            g = cbs_adjoint_gradient_240(
                c, target_dobs, src_indices, rec_indices, args, device
            )
            cbs_forward_calls_algorithm += 1
            cbs_adjoint_calls += 1
            g = normalize_rms(g)

            candidates = [{
                "direction": "zero",
                "factor": 0.0,
                "speed": c.copy(),
                "selection_loss": current_true_mse,
                "true_pred": pred_true,
            }]

            def eval_direction(sign, label):
                nonlocal cbs_forward_calls_algorithm
                out_local = []
                for fac in args.step_factors:
                    step = args.base_step_mps * float(fac)
                    cand = np.clip(
                        c + sign * step * g,
                        args.speed_min, args.speed_max
                    ).astype(np.float32)
                    cand480 = resize_np(cand, 480)
                    pred = forward_cbs_dobs(
                        cand480, src_indices, rec_indices, args, device
                    )
                    cbs_forward_calls_algorithm += 1
                    loss = complex_mse(pred, target_dobs)
                    out_local.append({
                        "direction": label,
                        "factor": float(fac),
                        "speed": cand,
                        "selection_loss": loss,
                        "true_pred": pred,
                    })
                return out_local

            candidates.extend(eval_direction(-1.0, "-"))
            best = min(candidates, key=lambda x: x["selection_loss"])

            if args.fallback_opposite and best["direction"] == "zero":
                candidates.extend(eval_direction(+1.0, "+"))
                best = min(candidates, key=lambda x: x["selection_loss"])

            updated = best["direction"] != "zero"

            if updated:
                c = best["speed"].astype(np.float32)
                pred_true = best["true_pred"]
                current_true_mse = float(best["selection_loss"])

            current_true_mae = complex_mae(pred_true, target_dobs)
            current_true_rrmse = complex_rrmse(pred_true, target_dobs)

            c480 = resize_np(c, 480)
            m240 = image_metrics(c, gt_240)
            m480 = image_metrics(c480, gt_480)

            history.append({
                "iter": it,
                "grad_rms": rms(g),
                "selected_direction": best["direction"],
                "selected_factor": best["factor"],
                "updated": int(updated),
                "true_cbs_mse": current_true_mse,
                "true_cbs_mae": current_true_mae,
                "true_cbs_rrmse": current_true_rrmse,
                "image_mse_240": m240["mse"],
                "image_mae_240": m240["mae"],
                "image_psnr_240": m240["psnr"],
                "image_ssim_240": m240["ssim"],
                "image_mse_480": m480["mse"],
                "candidate_records": [
                    {
                        "direction": cc["direction"],
                        "factor": cc["factor"],
                        "selection_loss": cc["selection_loss"],
                    }
                    for cc in candidates
                ],
            })

            print(
                f"    iter={it:02d} | choose={best['direction']}{best['factor']:.2g} | "
                f"CBS-MSE={current_true_mse:.6e} | imgMSE240={m240['mse']:.3f}"
            )

            if not updated and args.stop_on_no_update:
                print("    early stop: no improving candidate")
                break

        final_c480 = resize_np(c, 480)
        final_m240 = image_metrics(c, gt_240)
        final_m480 = image_metrics(final_c480, gt_480)

        sample_out = out / base
        sample_out.mkdir(parents=True, exist_ok=True)

        with open(sample_out / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        np.savez_compressed(
            sample_out / "final_result.npz",
            init_speed_240=resize_np(init_480_stored, args.image_size).astype(np.float32),
            final_speed_240=c.astype(np.float32),
            gt_speed_240=gt_240.astype(np.float32),
            final_speed_480=final_c480.astype(np.float32),
            gt_speed_480=gt_480.astype(np.float32),
            target_dobs=target_dobs.astype(np.complex64),
            final_true_cbs_dobs=pred_true.astype(np.complex64),
        )

        init_true_mse = history[0]["true_cbs_mse"]
        final_true_mse = history[-1]["true_cbs_mse"]

        row = {
            "base_sample": base,
            "num_updates": int(sum(h.get("updated", 0) for h in history[1:])),
            "num_iters_run": len(history) - 1,

            "init_true_cbs_mse": init_true_mse,
            "final_true_cbs_mse": final_true_mse,
            "true_cbs_mse_rel_reduction": float(
                (init_true_mse - final_true_mse) / (init_true_mse + 1e-20)
            ),

            "init_mse_240": init_m240["mse"],
            "final_mse_240": final_m240["mse"],
            "mse240_rel_reduction": float(
                (init_m240["mse"] - final_m240["mse"]) / (init_m240["mse"] + 1e-20)
            ),
            "init_mae_240": init_m240["mae"],
            "final_mae_240": final_m240["mae"],
            "init_psnr_240": init_m240["psnr"],
            "final_psnr_240": final_m240["psnr"],
            "init_ssim_240": init_m240["ssim"],
            "final_ssim_240": final_m240["ssim"],

            "init_mse_480": init_m480["mse"],
            "final_mse_480": final_m480["mse"],
            "mse480_rel_reduction": float(
                (init_m480["mse"] - final_m480["mse"]) / (init_m480["mse"] + 1e-20)
            ),

            "cbs_forward_calls_algorithm": cbs_forward_calls_algorithm,
            "cbs_adjoint_calls": cbs_adjoint_calls,
            "neural_grad_calls": 0,
            "runtime_sec": float(time.time() - t0),
        }
        sample_rows.append(row)

        print(
            f"[{base}] FINAL | CBS reduction={100*row['true_cbs_mse_rel_reduction']:.2f}% | "
            f"MSE240 reduction={100*row['mse240_rel_reduction']:.2f}% | "
            f"CBS-fw={row['cbs_forward_calls_algorithm']} | "
            f"CBS-adj={row['cbs_adjoint_calls']} | "
            f"time={row['runtime_sec']:.1f}s"
        )

    write_csv(out / "sample_results.csv", sample_rows)

    def stats(key):
        a = np.asarray([float(r[key]) for r in sample_rows], dtype=np.float64)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    summary = {
        "num_samples": len(sample_rows),
        "true_cbs_mse_rel_reduction": stats("true_cbs_mse_rel_reduction"),
        "mse240_rel_reduction": stats("mse240_rel_reduction"),
        "mse480_rel_reduction": stats("mse480_rel_reduction"),
        "init_mse_240": stats("init_mse_240"),
        "final_mse_240": stats("final_mse_240"),
        "init_mse_480": stats("init_mse_480"),
        "final_mse_480": stats("final_mse_480"),
        "init_psnr_240": stats("init_psnr_240"),
        "final_psnr_240": stats("final_psnr_240"),
        "cbs_forward_calls_algorithm": stats("cbs_forward_calls_algorithm"),
        "cbs_adjoint_calls": stats("cbs_adjoint_calls"),
        "runtime_sec": stats("runtime_sec"),
        "num_physics_improved": int(
            sum(r["true_cbs_mse_rel_reduction"] > 0 for r in sample_rows)
        ),
        "num_image240_improved": int(
            sum(r["mse240_rel_reduction"] > 0 for r in sample_rows)
        ),
    }

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 110)
    print("[DONE]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("saved to:", out.resolve())


if __name__ == "__main__":
    main()
