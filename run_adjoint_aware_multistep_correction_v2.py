#!/usr/bin/env python3
"""
Multi-step adjoint-aware neural correction for the local-alpha sparse-64 dataset.

Goal
----
Test whether the adjoint-aware surrogate gradient remains useful OFF the predefined
local-alpha path when iterated for multiple inversion updates.

Default algorithm (hybrid, recommended):
    1) Compute neural inverse-loss gradient g_theta(c, y_GT).
    2) RMS-normalize it.
    3) Propose multi-scale steps in the learned descent direction.
    4) Validate candidates with true CBS forward solves.
    5) Include the current iterate as a no-update candidate.
    6) Optional fallback: if the learned sign fails, try the opposite sign.
    7) NO CBS adjoint solve is used.

This isolates the value of learned adjoint sensitivity:
    full CBS correction: CBS adjoint + CBS candidate verification
    this hybrid:         neural gradient + CBS candidate verification

The script runs directly on local-alpha base groups:
    alpha=0   -> initialization
    alpha=1   -> GT image + observed y_GT

Main outputs
------------
sample_results.csv
summary.json
<base_sample>/history.json
<base_sample>/final_result.npz
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

from cbs_model import ConvergentBornSeries_Batch
from run_neural_measurement_correction import (
    load_residual_measurement_model,
    compute_neural_grad,
    eval_neural_np,
    make_coord_maps,
    resize_np,
)

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
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
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
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": ssim,
    }


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
    loss, pred = eval_neural_np(
        model=model,
        speed_np=speed_np,
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
    return float(loss), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        type=str,
        default="./self_consistent_cbs/sparse_64_local_alpha_train100_test20",
    )
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--mean_dobs_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--max_base", type=int, default=-1)
    ap.add_argument("--base_start", type=int, default=1,
                    help="1-based inclusive base position after numeric sorting.")
    ap.add_argument("--base_end", type=int, default=-1,
                    help="1-based inclusive base position; -1 means through the end.")
    ap.add_argument("--init_alpha", type=float, default=0.0)
    ap.add_argument("--gt_alpha", type=float, default=1.0)

    ap.add_argument("--num_steps", type=int, default=8)
    ap.add_argument("--base_step_mps", type=float, default=0.5)
    ap.add_argument(
        "--step_factors",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.25, 0.1],
    )
    ap.add_argument("--speed_min", type=float, default=1400.0)
    ap.add_argument("--speed_max", type=float, default=1605.0)

    ap.add_argument(
        "--candidate_validation",
        choices=["cbs", "neural"],
        default="cbs",
        help="cbs = true CBS candidate verification; neural = fully surrogate selection.",
    )
    ap.add_argument(
        "--fallback_opposite",
        action="store_true",
        help="With CBS validation, if -g gives no improving candidate, try +g.",
    )
    ap.add_argument(
        "--stop_on_no_update",
        action="store_true",
        help="Stop early when no candidate improves the selection objective.",
    )

    ap.add_argument("--frequency", type=float, default=500000.0)
    ap.add_argument("--forward_iters", type=int, default=80)
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

    groups_all = load_groups(args.data_root, args.split)
    items = list(groups_all.items())
    start = max(int(args.base_start), 1)
    end = int(args.base_end)
    if end < 0:
        end = len(items)
    if end < start:
        raise ValueError(f"base_end ({end}) must be >= base_start ({start})")
    items = items[start - 1:end]
    if args.max_base > 0:
        items = items[:args.max_base]
    groups = dict(items)

    print("=" * 110)
    print("Adjoint-aware multi-step correction")
    print("=" * 110)
    print("device =", device)
    print("image_size =", image_size)
    print("num bases =", len(groups))
    print("candidate_validation =", args.candidate_validation)
    print("fallback_opposite =", args.fallback_opposite)
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

        # Same parameterization as E4 / adjoint-aware training.
        c = resize_np(init_480_stored, image_size).astype(np.float32)
        gt_neural = resize_np(gt_480, image_size).astype(np.float32)

        # Save accepted trajectory states for later off-path adjoint-teacher generation.
        # State 0 is the local-alpha initialization; subsequent entries are accepted
        # updates only (no duplicate state is added for a rejected/no-update iteration).
        trajectory_speeds_240 = [c.copy()]
        trajectory_iter_indices = [0]
        trajectory_true_cbs_mse = []

        # Initial true physics evaluation.
        c480 = resize_np(c, 480).astype(np.float32)
        pred_true = forward_cbs_dobs(c480, src_indices, rec_indices, args, device)
        cbs_forward_calls_algorithm = 1
        cbs_forward_calls_diagnostic = 0
        neural_grad_calls = 0
        neural_forward_calls = 0

        current_true_mse = complex_mse(pred_true, target_dobs)
        current_true_mae = complex_mae(pred_true, target_dobs)
        current_true_rrmse = complex_rrmse(pred_true, target_dobs)
        trajectory_true_cbs_mse.append(current_true_mse)

        if args.candidate_validation == "neural":
            current_select_loss, _ = neural_objective(
                model, c, target_dobs,
                mean_real, mean_imag, residual_scale,
                speed_center, speed_scale, y_map, x_map, device,
            )
            neural_forward_calls += 1
        else:
            current_select_loss = current_true_mse

        history = []
        init_m240 = image_metrics(c, gt_neural)
        init_m480 = image_metrics(c480, gt_480)

        history.append({
            "iter": 0,
            "selected_direction": "init",
            "selected_factor": 0.0,
            "true_cbs_mse": current_true_mse,
            "true_cbs_mae": current_true_mae,
            "true_cbs_rrmse": current_true_rrmse,
            "selection_loss": current_select_loss,
            "image_mse_240": init_m240["mse"],
            "image_mae_240": init_m240["mae"],
            "image_psnr_240": init_m240["psnr"],
            "image_ssim_240": init_m240["ssim"],
            "image_mse_480": init_m480["mse"],
            "image_mae_480": init_m480["mae"],
            "image_psnr_480": init_m480["psnr"],
            "image_ssim_480": init_m480["ssim"],
            "candidate_records": [],
        })

        print(
            f"[{bi:02d}/{len(groups):02d}] {base} INIT | "
            f"CBS-MSE={current_true_mse:.6e} | "
            f"imgMSE240={init_m240['mse']:.3f}"
        )

        for it in range(1, args.num_steps + 1):
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
            neural_grad_calls += 1
            g = normalize_rms(g_nn)

            candidates = []
            # Current point is always a valid no-update candidate.
            candidates.append({
                "direction": "zero",
                "factor": 0.0,
                "speed": c.copy(),
                "selection_loss": current_select_loss,
                "true_mse": current_true_mse,
                "true_pred": pred_true,
            })

            def evaluate_direction(sign, label):
                nonlocal cbs_forward_calls_algorithm, neural_forward_calls
                local = []
                for fac in args.step_factors:
                    step = args.base_step_mps * float(fac)
                    cand = np.clip(
                        c + sign * step * g,
                        args.speed_min,
                        args.speed_max,
                    ).astype(np.float32)

                    if args.candidate_validation == "cbs":
                        cand480 = resize_np(cand, 480).astype(np.float32)
                        pred = forward_cbs_dobs(
                            cand480, src_indices, rec_indices, args, device
                        )
                        cbs_forward_calls_algorithm += 1
                        sel = complex_mse(pred, target_dobs)
                        true_mse = sel
                    else:
                        sel, _ = neural_objective(
                            model, cand, target_dobs,
                            mean_real, mean_imag, residual_scale,
                            speed_center, speed_scale, y_map, x_map, device,
                        )
                        neural_forward_calls += 1
                        pred = None
                        true_mse = None

                    local.append({
                        "direction": label,
                        "factor": float(fac),
                        "speed": cand,
                        "selection_loss": float(sel),
                        "true_mse": true_mse,
                        "true_pred": pred,
                    })
                return local

            # Learned descent direction is c - step*g.
            minus_candidates = evaluate_direction(-1.0, "-")
            candidates.extend(minus_candidates)

            best_minus = min(candidates, key=lambda x: x["selection_loss"])

            # Optional safety valve: only pay for opposite sign when -g fails.
            if (
                args.candidate_validation == "cbs"
                and args.fallback_opposite
                and best_minus["direction"] == "zero"
            ):
                plus_candidates = evaluate_direction(+1.0, "+")
                candidates.extend(plus_candidates)

            best = min(candidates, key=lambda x: x["selection_loss"])
            updated = best["direction"] != "zero"

            if updated:
                c = best["speed"].astype(np.float32)

                if args.candidate_validation == "cbs":
                    pred_true = best["true_pred"]
                    current_true_mse = float(best["true_mse"])
                    current_select_loss = float(best["selection_loss"])
                else:
                    current_select_loss = float(best["selection_loss"])
                    c480 = resize_np(c, 480).astype(np.float32)
                    pred_true = forward_cbs_dobs(
                        c480, src_indices, rec_indices, args, device
                    )
                    cbs_forward_calls_diagnostic += 1
                    current_true_mse = complex_mse(pred_true, target_dobs)
            else:
                # No update; current true prediction/loss are still valid.
                pass

            if updated:
                trajectory_speeds_240.append(c.copy())
                trajectory_iter_indices.append(int(it))
                trajectory_true_cbs_mse.append(float(current_true_mse))

            current_true_mae = complex_mae(pred_true, target_dobs)
            current_true_rrmse = complex_rrmse(pred_true, target_dobs)

            c480 = resize_np(c, 480).astype(np.float32)
            m240 = image_metrics(c, gt_neural)
            m480 = image_metrics(c480, gt_480)

            candidate_records = []
            for cc in candidates:
                candidate_records.append({
                    "direction": cc["direction"],
                    "factor": cc["factor"],
                    "selection_loss": cc["selection_loss"],
                    "true_mse": cc["true_mse"],
                })

            history.append({
                "iter": it,
                "neural_loss": float(neural_loss),
                "grad_rms": rms(g_nn),
                "selected_direction": best["direction"],
                "selected_factor": best["factor"],
                "updated": int(updated),
                "true_cbs_mse": current_true_mse,
                "true_cbs_mae": current_true_mae,
                "true_cbs_rrmse": current_true_rrmse,
                "selection_loss": current_select_loss,
                "image_mse_240": m240["mse"],
                "image_mae_240": m240["mae"],
                "image_psnr_240": m240["psnr"],
                "image_ssim_240": m240["ssim"],
                "image_mse_480": m480["mse"],
                "image_mae_480": m480["mae"],
                "image_psnr_480": m480["psnr"],
                "image_ssim_480": m480["ssim"],
                "candidate_records": candidate_records,
            })

            print(
                f"    iter={it:02d} | choose={best['direction']}{best['factor']:.2g} | "
                f"CBS-MSE={current_true_mse:.6e} | "
                f"imgMSE240={m240['mse']:.3f}"
            )

            if not updated and args.stop_on_no_update:
                print("    early stop: no improving candidate")
                break

        final_m240 = image_metrics(c, gt_neural)
        final_c480 = resize_np(c, 480).astype(np.float32)
        final_m480 = image_metrics(final_c480, gt_480)

        sample_out = out / base
        sample_out.mkdir(parents=True, exist_ok=True)
        with open(sample_out / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        np.savez_compressed(
            sample_out / "trajectory_states.npz",
            speeds_240=np.stack(trajectory_speeds_240, axis=0).astype(np.float32),
            iter_indices=np.asarray(trajectory_iter_indices, dtype=np.int32),
            true_cbs_mse=np.asarray(trajectory_true_cbs_mse, dtype=np.float64),
            target_gt_dobs=target_dobs.astype(np.complex64),
            src_indices=src_indices.astype(np.int64),
            rec_indices=rec_indices.astype(np.int64),
            gt_speed_240=gt_neural.astype(np.float32),
            gt_speed_480=gt_480.astype(np.float32),
            base_sample=np.array(base),
        )

        np.savez_compressed(
            sample_out / "final_result.npz",
            init_speed_240=resize_np(init_480_stored, image_size).astype(np.float32),
            final_speed_240=c.astype(np.float32),
            gt_speed_240=gt_neural.astype(np.float32),
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
                (init_m240["mse"] - final_m240["mse"])
                / (init_m240["mse"] + 1e-20)
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
                (init_m480["mse"] - final_m480["mse"])
                / (init_m480["mse"] + 1e-20)
            ),

            "cbs_forward_calls_algorithm": cbs_forward_calls_algorithm,
            "cbs_forward_calls_diagnostic": cbs_forward_calls_diagnostic,
            "cbs_adjoint_calls": 0,
            "neural_grad_calls": neural_grad_calls,
            "neural_forward_calls": neural_forward_calls,
            "runtime_sec": float(time.time() - t0),
        }
        sample_rows.append(row)

        print(
            f"[{base}] FINAL | CBS reduction={100*row['true_cbs_mse_rel_reduction']:.2f}% | "
            f"MSE240 reduction={100*row['mse240_rel_reduction']:.2f}% | "
            f"CBS-fw={row['cbs_forward_calls_algorithm']} | "
            f"NN-grad={row['neural_grad_calls']} | "
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
        "neural_grad_calls": stats("neural_grad_calls"),
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
