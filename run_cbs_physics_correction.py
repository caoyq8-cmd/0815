import os
import json
import math
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from cbs_model import ConvergentBornSeries_Batch, ConvergentBornSeries_Batch_Adjoint


# ============================================================
# basic utils
# ============================================================

def upsample_256_to_480_np(x256):
    x = torch.from_numpy(x256.astype(np.float32))[None, None]
    x480 = F.interpolate(x, size=(480, 480), mode="bilinear", align_corners=False)
    return x480[0, 0].cpu().numpy().astype(np.float32)


def downsample_480_to_256_np(x480):
    x = torch.from_numpy(x480.astype(np.float32))[None, None]
    x256 = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    return x256[0, 0].cpu().numpy().astype(np.float32)


def rel_l1(a, b):
    return float(np.mean(np.abs(a - b)) / (np.mean(np.abs(b)) + 1e-12))


def rel_l2(a, b):
    return float(
        np.sqrt(np.mean(np.abs(a - b) ** 2))
        / (np.sqrt(np.mean(np.abs(b) ** 2)) + 1e-12)
    )


def image_metrics(pred, target):
    diff = pred.astype(np.float32) - target.astype(np.float32)
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    return mse, mae, rmse


def psnr_from_mse(mse, data_range=205.0):
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(data_range / math.sqrt(mse)))


def gaussian_window(window_size=11, sigma=1.5, channels=1, device="cpu", dtype=torch.float32):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g).unsqueeze(0).unsqueeze(0)
    return window_2d.repeat(channels, 1, 1, 1)


def compute_ssim_np(pred, target, value_min=1400.0, value_max=1605.0, window_size=11, sigma=1.5):
    """
    与项目 eval_inversionnet_ablation.py 一致的 SSIM 定义：
    先按固定物理范围归一化到 [0,1]，再使用高斯窗口标准 SSIM。
    """
    pred_t = torch.from_numpy(pred.astype(np.float32))[None, None]
    target_t = torch.from_numpy(target.astype(np.float32))[None, None]

    pred_t = ((pred_t - value_min) / (value_max - value_min)).clamp(0.0, 1.0)
    target_t = ((target_t - value_min) / (value_max - value_min)).clamp(0.0, 1.0)

    window = gaussian_window(
        window_size=window_size,
        sigma=sigma,
        channels=1,
        device=pred_t.device,
        dtype=pred_t.dtype,
    )

    mu1 = F.conv2d(pred_t, window, padding=window_size // 2)
    mu2 = F.conv2d(target_t, window, padding=window_size // 2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred_t * pred_t, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(target_t * target_t, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(pred_t * target_t, window, padding=window_size // 2) - mu1_mu2

    sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12
    ssim_val = float((numerator / denominator).mean().item())
    return max(min(ssim_val, 1.0), -1.0)


def build_image_metric_dict(pred, target, prefix, value_min, value_max):
    mse, mae, rmse = image_metrics(pred, target)
    return {
        f"image_mse_{prefix}": mse,
        f"image_mae_{prefix}": mae,
        f"image_rmse_{prefix}": rmse,
        f"image_psnr_{prefix}": psnr_from_mse(mse, value_max - value_min),
        f"image_ssim_{prefix}": compute_ssim_np(pred, target, value_min, value_max),
    }


def save_image(img, path, title=None, cmap="inferno", vmin=None, vmax=None):
    plt.figure(figsize=(5, 4))
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_compare_vis(condition_480, corrected_480, target_480, out_dir, prefix, vmin=1400.0, vmax=1605.0):
    os.makedirs(out_dir, exist_ok=True)
    err_before = condition_480 - target_480
    err_after = corrected_480 - target_480
    update = corrected_480 - condition_480

    items = [
        ("condition_480", condition_480, "inferno", vmin, vmax),
        ("corrected_480", corrected_480, "inferno", vmin, vmax),
        ("target_480", target_480, "inferno", vmin, vmax),
        ("update", update, "bwr", None, None),
        ("error_before", err_before, "bwr", None, None),
        ("error_after", err_after, "bwr", None, None),
    ]
    for name, img, cmap, lo, hi in items:
        if cmap == "bwr":
            m = max(abs(float(img.min())), abs(float(img.max())), 1.0)
            lo, hi = -m, m
        save_image(
            img,
            os.path.join(out_dir, f"{prefix}_{name}.png"),
            title=name,
            cmap=cmap,
            vmin=lo,
            vmax=hi,
        )


def save_dobs_vis(dobs_gt, dobs_pred, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    residual = dobs_pred - dobs_gt
    items = [
        ("dobs_gt_abs", np.abs(dobs_gt), "viridis"),
        ("dobs_pred_abs", np.abs(dobs_pred), "viridis"),
        ("dobs_residual_abs", np.abs(residual), "viridis"),
        ("dobs_residual_real", residual.real, "seismic"),
        ("dobs_residual_imag", residual.imag, "seismic"),
    ]
    for name, img, cmap in items:
        save_image(
            img,
            os.path.join(out_dir, f"{prefix}_{name}.png"),
            title=name,
            cmap=cmap,
        )


# ============================================================
# CBS forward / adjoint
# ============================================================

@torch.no_grad()
def forward_dobs(sos_480, src_indices, rec_indices, args, device):
    sos = torch.from_numpy(sos_480.astype(np.float32))[None, None].to(device)
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
    rec_t = torch.from_numpy(rec_indices.astype(np.int64)).to(device)
    dobs = u[0, :, rec_t[:, 0], rec_t[:, 1]]
    dobs_np = dobs.detach().cpu().numpy().astype(np.complex64)

    del model, u, sos
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dobs_np


def compute_meas_loss(dobs_pred, dobs_gt):
    return float(np.mean(np.abs(dobs_pred - dobs_gt)))


@torch.no_grad()
def compute_adjoint_grad(sos_480, dobs_gt, src_indices, rec_indices, args, device):
    """
    返回：
      grad_np: [480,480] float32
      loss_value: CBS adjoint 实现返回的损失值
      dobs_pred: 当前模型的接收矩阵
    """
    sos = torch.from_numpy(sos_480.astype(np.float32))[None, None].to(device)
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

    dobs_t = torch.from_numpy(dobs_gt.astype(np.complex64))[None].to(device)
    mask = np.ones_like(np.abs(dobs_gt), dtype=np.float32)

    adj = ConvergentBornSeries_Batch_Adjoint(
        batch_model=model,
        rec_loc=rec_t,
        dobs_500k_batch=dobs_t,
        dobs_500k_mask=mask,
    )
    grad, loss_value = adj(u, max_iters=args.adjoint_iters)

    if args.smooth_kernel > 1:
        k = args.smooth_kernel
        if k % 2 == 0:
            raise ValueError("--smooth_kernel 必须为奇数，或设置为 1 表示不平滑。")
        grad = F.avg_pool2d(grad, kernel_size=k, stride=1, padding=k // 2)

    grad_np = grad[0, 0].detach().cpu().numpy().astype(np.float32)
    loss_float = float(loss_value.detach().cpu())

    del model, adj, u, sos, grad
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return grad_np, loss_float, dobs_pred


def normalize_grad(grad, eps=1e-12):
    g = grad.astype(np.float32)
    rms = float(np.sqrt(np.mean(g ** 2)) + eps)
    return g / rms, rms


def parse_step_factors(values):
    factors = [float(x) for x in values]
    if len(factors) == 0:
        raise ValueError("--step_factors 至少需要一个值。")
    if any(x <= 0 for x in factors):
        raise ValueError("--step_factors 必须全部 > 0。")
    return factors


def build_candidates(current, total_grad, args):
    """仅构造候选，不做真实 CBS 评价。"""
    candidates = []
    if args.include_no_update:
        candidates.append({
            "name": "zero",
            "sos": current.copy(),
            "step_used": 0.0,
            "step_factor": 0.0,
            "sign": "0",
        })

    signs = []
    if args.direction_mode in ("minus", "both"):
        signs.append(("-", -1.0))
    if args.direction_mode in ("plus", "both"):
        signs.append(("+", +1.0))

    for fac in args.step_factors:
        step = args.step_size_mps * fac
        for sign_name, sign_value in signs:
            cand = np.clip(
                current + sign_value * step * total_grad,
                args.speed_min,
                args.speed_max,
            ).astype(np.float32)
            candidates.append({
                "name": f"{sign_name}{fac:g}",
                "sos": cand,
                "step_used": float(step),
                "step_factor": float(fac),
                "sign": sign_name,
            })

    if len(candidates) == 0:
        raise RuntimeError("没有构造出任何候选，请检查 direction_mode/include_no_update。")
    return candidates


def choose_unverified_candidate(candidates, args):
    """
    No-CBS-verification 消融：不使用候选真实测量损失进行筛选。
    经典梯度下降默认选择 nominal minus 方向 + 最大 step factor。
    如果只配置 plus，则选择 plus。
    no-update 候选不会在该模式下被自动选择。
    """
    nonzero = [c for c in candidates if c["sign"] != "0"]
    if len(nonzero) == 0:
        return candidates[0]

    preferred_sign = "-" if any(c["sign"] == "-" for c in nonzero) else "+"
    same_sign = [c for c in nonzero if c["sign"] == preferred_sign]
    return max(same_sign, key=lambda c: c["step_factor"])


# ============================================================
# one physics-correction step
# ============================================================

def correction_step(current, condition, dobs_gt, src_indices, rec_indices, args, device):
    """
    可控消融版 correction step。

    candidate_validation='cbs':
      使用真实 CBS forward 对所有候选进行回溯/验证并选最优。

    candidate_validation='none':
      不使用 CBS 评价来选择候选；直接执行 nominal 梯度更新。
      为了记录论文评价指标，更新后仍额外做 1 次 CBS forward，
      但该 forward 只用于 evaluation，不参与 candidate selection。
    """
    t0 = time.perf_counter()

    grad, loss_before_from_adjoint, dobs_pred_before = compute_adjoint_grad(
        current, dobs_gt, src_indices, rec_indices, args, device
    )
    current_loss = compute_meas_loss(dobs_pred_before, dobs_gt)

    grad_norm, grad_rms = normalize_grad(grad)

    if args.prior_tether > 0:
        prior_grad = current - condition
        prior_norm, prior_rms = normalize_grad(prior_grad)
        total_grad = grad_norm + args.prior_tether * prior_norm
    else:
        prior_rms = 0.0
        total_grad = grad_norm

    total_grad_rms = float(np.sqrt(np.mean(total_grad.astype(np.float64) ** 2)) + 1e-12)
    candidates = build_candidates(current, total_grad, args)

    candidate_selection_forward_evals = 0
    post_update_eval_forward_evals = 0

    if args.candidate_validation == "cbs":
        best = None
        candidate_records = []
        for cand in candidates:
            if cand["sign"] == "0":
                dobs_cand = dobs_pred_before
            else:
                dobs_cand = forward_dobs(cand["sos"], src_indices, rec_indices, args, device)
                candidate_selection_forward_evals += 1

            loss_cand = compute_meas_loss(dobs_cand, dobs_gt)
            candidate_records.append({
                "name": cand["name"],
                "sign": cand["sign"],
                "step_used": cand["step_used"],
                "step_factor": cand["step_factor"],
                "dobs_abs_loss": loss_cand,
            })
            if best is None or loss_cand < best["loss"]:
                best = {
                    **cand,
                    "dobs": dobs_cand,
                    "loss": loss_cand,
                }

        chosen = best["sos"]
        chosen_dobs = best["dobs"]
        chosen_loss = best["loss"]
        chosen_name = best["name"]
        chosen_sign = best["sign"]
        chosen_step_used = best["step_used"]
        chosen_step_factor = best["step_factor"]
    else:
        chosen_cand = choose_unverified_candidate(candidates, args)
        chosen = chosen_cand["sos"]
        chosen_name = chosen_cand["name"]
        chosen_sign = chosen_cand["sign"]
        chosen_step_used = chosen_cand["step_used"]
        chosen_step_factor = chosen_cand["step_factor"]
        candidate_records = []

        # 仅用于结果统计，不参与选择。
        if chosen_sign == "0":
            chosen_dobs = dobs_pred_before
        else:
            chosen_dobs = forward_dobs(chosen, src_indices, rec_indices, args, device)
            post_update_eval_forward_evals = 1
        chosen_loss = compute_meas_loss(chosen_dobs, dobs_gt)

    elapsed = time.perf_counter() - t0
    info = {
        "loss_before_from_adjoint": float(loss_before_from_adjoint),
        "loss_before_abs": float(current_loss),
        "loss_after": float(chosen_loss),
        "loss_decreased": float(chosen_loss <= current_loss),
        "chosen_candidate": chosen_name,
        "chosen_sign": chosen_sign,
        "chosen_step_used": float(chosen_step_used),
        "chosen_step_factor": float(chosen_step_factor),
        "grad_rms": float(grad_rms),
        "prior_rms": float(prior_rms),
        "total_grad_rms": float(total_grad_rms),
        "update_abs_mean": float(np.mean(np.abs(chosen - current))),
        "update_abs_max": float(np.max(np.abs(chosen - current))),
        "dobs_rel_l1_after": rel_l1(chosen_dobs, dobs_gt),
        "dobs_rel_l2_after": rel_l2(chosen_dobs, dobs_gt),
        "candidate_validation": args.candidate_validation,
        "direction_mode": args.direction_mode,
        "candidate_selection_forward_evals": int(candidate_selection_forward_evals),
        "post_update_eval_forward_evals": int(post_update_eval_forward_evals),
        "candidate_count": int(len(candidates)),
        "candidate_records": candidate_records,
        "step_runtime_seconds": float(elapsed),
    }
    return chosen, info, chosen_dobs, grad


# ============================================================
# main experiment
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stable CBS physics correction with controllable ablations."
    )

    parser.add_argument("--sample_path", type=str, required=True)
    parser.add_argument("--condition_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_iters", type=int, default=8)
    parser.add_argument("--step_size_mps", type=float, default=1.0)
    parser.add_argument("--prior_tether", type=float, default=0.1)
    parser.add_argument(
        "--step_factors",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.25, 0.1],
        help="候选步长比例，例如 1.0 0.5 0.25 0.1",
    )
    parser.add_argument(
        "--direction_mode",
        type=str,
        choices=["both", "minus", "plus"],
        default="both",
        help="both=正负双方向；minus=标准减梯度；plus=加梯度。",
    )
    parser.add_argument(
        "--candidate_validation",
        type=str,
        choices=["cbs", "none"],
        default="cbs",
        help="cbs=真实 CBS 筛候选；none=不验证，直接 nominal gradient update。",
    )

    no_update_group = parser.add_mutually_exclusive_group()
    no_update_group.add_argument("--include_no_update", dest="include_no_update", action="store_true")
    no_update_group.add_argument("--no_include_no_update", dest="include_no_update", action="store_false")
    parser.set_defaults(include_no_update=True)

    # 向后兼容旧命令。旧脚本中该 flag 实际未控制逻辑；新脚本中若出现则强制 both。
    parser.add_argument("--try_both_signs", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--frequency", type=float, default=500e3)
    parser.add_argument("--forward_iters", type=int, default=80)
    parser.add_argument("--adjoint_iters", type=int, default=80)
    parser.add_argument("--boundary_width", type=int, default=300)
    parser.add_argument("--boundary_strength", type=float, default=225.0)
    parser.add_argument("--boundary_type", type=str, default="PML3")
    parser.add_argument("--smooth_kernel", type=int, default=9)

    parser.add_argument("--speed_min", type=float, default=1400.0)
    parser.add_argument("--speed_max", type=float, default=1605.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--skip_initial_vis", action="store_true")

    args = parser.parse_args()
    args.step_factors = parse_step_factors(args.step_factors)
    if args.try_both_signs:
        args.direction_mode = "both"
    if args.smooth_kernel < 1:
        raise ValueError("--smooth_kernel 必须 >= 1；设为 1 表示不平滑。")
    if args.smooth_kernel > 1 and args.smooth_kernel % 2 == 0:
        raise ValueError("--smooth_kernel 必须为奇数。")
    if args.speed_max <= args.speed_min:
        raise ValueError("speed_max 必须大于 speed_min。")
    return args


def main():
    args = parse_args()
    run_t0 = time.perf_counter()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device =", device)
    print("ablation config =", {
        "prior_tether": args.prior_tether,
        "step_factors": args.step_factors,
        "direction_mode": args.direction_mode,
        "candidate_validation": args.candidate_validation,
        "include_no_update": args.include_no_update,
        "smooth_kernel": args.smooth_kernel,
    })

    sc = np.load(args.sample_path)
    cc = np.load(args.condition_path)

    target_480 = sc["target_480"].astype(np.float32)
    target_256_phys = sc["target_256"].astype(np.float32)
    dobs_gt = sc["dobs_complex"].astype(np.complex64)
    src_indices = sc["src_indices"].astype(np.int64)
    rec_indices = sc["rec_indices"].astype(np.int64)

    # condition cache 是 image coordinate，需要转置到 physics coordinate。
    condition_256_img = cc["condition_speed"][0].astype(np.float32)
    target_256_img = cc["target_speed"][0].astype(np.float32)
    condition_256_phys = condition_256_img.T
    target_256_from_cache_phys = target_256_img.T

    target_align_mse = float(np.mean((target_256_phys - target_256_from_cache_phys) ** 2))
    print("target alignment MSE =", target_align_mse)

    condition_480 = upsample_256_to_480_np(condition_256_phys)
    condition_480 = np.clip(condition_480, args.speed_min, args.speed_max).astype(np.float32)
    current = condition_480.copy()
    condition_fixed = condition_480.copy()

    # 初始真实 CBS measurement。
    dobs_init = forward_dobs(current, src_indices, rec_indices, args, device)
    init_metrics_256 = build_image_metric_dict(
        condition_256_phys, target_256_phys, "256", args.speed_min, args.speed_max
    )
    init_metrics_480 = build_image_metric_dict(
        current, target_480, "480", args.speed_min, args.speed_max
    )

    init_item = {
        "iter": 0,
        **init_metrics_256,
        **init_metrics_480,
        "dobs_rel_l1": rel_l1(dobs_init, dobs_gt),
        "dobs_rel_l2": rel_l2(dobs_init, dobs_gt),
        "dobs_abs_loss": compute_meas_loss(dobs_init, dobs_gt),
        "chosen_candidate": "init",
        "chosen_sign": "init",
        "chosen_step_used": 0.0,
        "chosen_step_factor": 0.0,
        "candidate_selection_forward_evals": 0,
        "post_update_eval_forward_evals": 0,
        "target_alignment_mse": target_align_mse,
    }

    print("=" * 80)
    print("[Initial]")
    print("=" * 80)
    for k, v in init_item.items():
        if isinstance(v, float):
            print(f"{k}: {v:.8e}")
        else:
            print(f"{k}: {v}")

    history = [init_item]
    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    if not args.skip_initial_vis:
        save_compare_vis(
            condition_480, current, target_480, args.output_dir,
            prefix="iter_000", vmin=args.speed_min, vmax=args.speed_max
        )
        save_dobs_vis(dobs_gt, dobs_init, args.output_dir, prefix="iter_000")

    total_candidate_selection_forward_evals = 0
    total_post_update_eval_forward_evals = 0

    for it in range(1, args.num_iters + 1):
        print("=" * 80)
        print(f"[Iteration {it}/{args.num_iters}]")
        print("=" * 80)

        current, info, dobs_current, grad = correction_step(
            current,
            condition_fixed,
            dobs_gt,
            src_indices,
            rec_indices,
            args,
            device,
        )

        current_256_phys = downsample_480_to_256_np(current)
        item = {
            "iter": it,
            **build_image_metric_dict(
                current_256_phys, target_256_phys, "256", args.speed_min, args.speed_max
            ),
            **build_image_metric_dict(
                current, target_480, "480", args.speed_min, args.speed_max
            ),
            "dobs_rel_l1": rel_l1(dobs_current, dobs_gt),
            "dobs_rel_l2": rel_l2(dobs_current, dobs_gt),
            "dobs_abs_loss": compute_meas_loss(dobs_current, dobs_gt),
        }
        item.update(info)
        history.append(item)

        total_candidate_selection_forward_evals += info["candidate_selection_forward_evals"]
        total_post_update_eval_forward_evals += info["post_update_eval_forward_evals"]

        for k, v in item.items():
            if k == "candidate_records":
                continue
            if isinstance(v, float):
                print(f"{k}: {v:.8e}")
            else:
                print(f"{k}: {v}")

        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        if args.save_every > 0 and it % args.save_every == 0:
            save_compare_vis(
                condition_480,
                current,
                target_480,
                args.output_dir,
                prefix=f"iter_{it:03d}",
                vmin=args.speed_min,
                vmax=args.speed_max,
            )
            save_dobs_vis(dobs_gt, dobs_current, args.output_dir, prefix=f"iter_{it:03d}")
            save_image(
                grad,
                os.path.join(args.output_dir, f"iter_{it:03d}_grad.png"),
                title="adjoint grad",
                cmap="bwr",
            )

        np.savez_compressed(
            os.path.join(args.output_dir, "final_result.npz"),
            corrected_480=current.astype(np.float32),
            condition_480=condition_480.astype(np.float32),
            target_480=target_480.astype(np.float32),
            corrected_256_phys=current_256_phys.astype(np.float32),
            condition_256_phys=condition_256_phys.astype(np.float32),
            target_256_phys=target_256_phys.astype(np.float32),
        )

    run_seconds = time.perf_counter() - run_t0
    final_item = history[-1]
    run_summary = {
        "sample_path": str(Path(args.sample_path).resolve()),
        "condition_path": str(Path(args.condition_path).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "num_iters": args.num_iters,
        "runtime_seconds": float(run_seconds),
        "initial": init_item,
        "final": final_item,
        "mse256_improve_ratio": float(
            (init_item["image_mse_256"] - final_item["image_mse_256"])
            / (init_item["image_mse_256"] + 1e-12)
        ),
        "mse480_improve_ratio": float(
            (init_item["image_mse_480"] - final_item["image_mse_480"])
            / (init_item["image_mse_480"] + 1e-12)
        ),
        "dobs_abs_improve_ratio": float(
            (init_item["dobs_abs_loss"] - final_item["dobs_abs_loss"])
            / (init_item["dobs_abs_loss"] + 1e-12)
        ),
        # initial forward 1 + 每轮 adjoint 内部 forward 1；下面单独统计 candidate/eval forward。
        "initial_forward_evals": 1,
        "adjoint_step_forward_evals": args.num_iters,
        "candidate_selection_forward_evals": int(total_candidate_selection_forward_evals),
        "post_update_eval_forward_evals": int(total_post_update_eval_forward_evals),
        "total_forward_like_evals_excluding_adjoint_solve": int(
            1 + args.num_iters + total_candidate_selection_forward_evals + total_post_update_eval_forward_evals
        ),
    }
    with open(os.path.join(args.output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("[Done]")
    print("=" * 80)
    print(f"runtime_seconds = {run_seconds:.2f}")
    print("saved to:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
