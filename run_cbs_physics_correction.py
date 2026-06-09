import os
import json
import math
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
    return float(np.sqrt(np.mean(np.abs(a - b) ** 2)) / (np.sqrt(np.mean(np.abs(b) ** 2)) + 1e-12))


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
        loss_value: mean abs residual
        dobs_pred: [M,L] complex64
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


# ============================================================
# 已替换：稳定版 correction_step（多尺度 + 不更新候选）
# ============================================================
def correction_step(current, condition, dobs_gt, src_indices, rec_indices, args, device):
    """
    对当前 sos 做一步 adjoint correction。
    稳定版：
    - 计算 adjoint gradient；
    - 梯度 RMS 归一化；
    - 试多个 step scale 和正负方向；
    - 包含 no-update 候选；
    - 选择 measurement loss 最小的候选。
    """
    grad, loss_before, dobs_pred_before = compute_adjoint_grad(
        current,
        dobs_gt,
        src_indices,
        rec_indices,
        args,
        device,
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

    # 候选列表：包含不更新，保证不会强制变差
    candidates = []

    candidates.append({
        "name": "zero",
        "sos": current.copy(),
        "step_used": 0.0,
        "sign": "0",
    })

    # 多尺度回溯
    step_factors = [1.0, 0.5, 0.25, 0.1]

    for fac in step_factors:
        step = args.step_size_mps * fac

        cand_minus = np.clip(
            current - step * total_grad,
            args.speed_min,
            args.speed_max,
        ).astype(np.float32)

        cand_plus = np.clip(
            current + step * total_grad,
            args.speed_min,
            args.speed_max,
        ).astype(np.float32)

        candidates.append({
            "name": f"-{fac}",
            "sos": cand_minus,
            "step_used": step,
            "sign": "-",
        })
        candidates.append({
            "name": f"+{fac}",
            "sos": cand_plus,
            "step_used": step,
            "sign": "+",
        })

    best = None

    for cand in candidates:
        if cand["name"] == "zero":
            dobs_cand = dobs_pred_before
        else:
            dobs_cand = forward_dobs(cand["sos"], src_indices, rec_indices, args, device)

        loss_cand = compute_meas_loss(dobs_cand, dobs_gt)

        if best is None or loss_cand < best["loss"]:
            best = {
                "sos": cand["sos"],
                "dobs": dobs_cand,
                "loss": loss_cand,
                "name": cand["name"],
                "step_used": cand["step_used"],
                "sign": cand["sign"],
            }

    chosen = best["sos"]
    chosen_dobs = best["dobs"]
    chosen_loss = best["loss"]

    info = {
        "loss_before_from_adjoint": loss_before,
        "loss_before_abs": current_loss,
        "loss_after": chosen_loss,
        "loss_decreased": float(chosen_loss <= current_loss),
        "chosen_candidate": best["name"],
        "chosen_sign": best["sign"],
        "chosen_step_used": best["step_used"],
        "grad_rms": grad_rms,
        "prior_rms": prior_rms,
        "update_abs_mean": float(np.mean(np.abs(chosen - current))),
        "update_abs_max": float(np.max(np.abs(chosen - current))),
        "dobs_rel_l1_after": rel_l1(chosen_dobs, dobs_gt),
        "dobs_rel_l2_after": rel_l2(chosen_dobs, dobs_gt),
    }

    return chosen, info, chosen_dobs, grad


# ============================================================
# main experiment
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_path", type=str, required=True)
    parser.add_argument("--condition_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_iters", type=int, default=5)
    parser.add_argument("--step_size_mps", type=float, default=1.0)
    parser.add_argument("--prior_tether", type=float, default=0.0)
    parser.add_argument("--try_both_signs", action="store_true")

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

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    sc = np.load(args.sample_path)
    cc = np.load(args.condition_path)

    target_480 = sc["target_480"].astype(np.float32)
    target_256_phys = sc["target_256"].astype(np.float32)
    dobs_gt = sc["dobs_complex"].astype(np.complex64)
    src_indices = sc["src_indices"].astype(np.int64)
    rec_indices = sc["rec_indices"].astype(np.int64)

    # condition cache 是 image coordinate，需要转置到 physics coordinate
    condition_256_img = cc["condition_speed"][0].astype(np.float32)
    target_256_img = cc["target_speed"][0].astype(np.float32)

    condition_256_phys = condition_256_img.T
    target_256_from_cache_phys = target_256_img.T

    # 检查 target 对齐
    target_align_mse = float(np.mean((target_256_phys - target_256_from_cache_phys) ** 2))
    print("target alignment MSE =", target_align_mse)

    condition_480 = upsample_256_to_480_np(condition_256_phys)
    condition_480 = np.clip(condition_480, args.speed_min, args.speed_max).astype(np.float32)

    current = condition_480.copy()
    condition_fixed = condition_480.copy()

    # 初始 dobs
    dobs_init = forward_dobs(current, src_indices, rec_indices, args, device)

    init_mse_480, init_mae_480, init_rmse_480 = image_metrics(current, target_480)
    init_mse_256, init_mae_256, init_rmse_256 = image_metrics(condition_256_phys, target_256_phys)

    print("=" * 80)
    print("[Initial]")
    print("=" * 80)
    print("image_mse_256 =", init_mse_256)
    print("image_mse_480 =", init_mse_480)
    print("dobs_rel_l1   =", rel_l1(dobs_init, dobs_gt))
    print("dobs_rel_l2   =", rel_l2(dobs_init, dobs_gt))
    print("dobs_abs_loss =", compute_meas_loss(dobs_init, dobs_gt))

    history = []

    history.append({
        "iter": 0,
        "image_mse_256": init_mse_256,
        "image_mae_256": init_mae_256,
        "image_rmse_256": init_rmse_256,
        "image_psnr_256": psnr_from_mse(init_mse_256, args.speed_max - args.speed_min),
        "image_mse_480": init_mse_480,
        "image_mae_480": init_mae_480,
        "image_rmse_480": init_rmse_480,
        "image_psnr_480": psnr_from_mse(init_mse_480, args.speed_max - args.speed_min),
        "dobs_rel_l1": rel_l1(dobs_init, dobs_gt),
        "dobs_rel_l2": rel_l2(dobs_init, dobs_gt),
        "dobs_abs_loss": compute_meas_loss(dobs_init, dobs_gt),
        "chosen_sign": "init",
    })

    save_compare_vis(
        condition_480,
        current,
        target_480,
        args.output_dir,
        prefix="iter_000",
        vmin=args.speed_min,
        vmax=args.speed_max,
    )
    save_dobs_vis(dobs_gt, dobs_init, args.output_dir, prefix="iter_000")

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

        mse_480, mae_480, rmse_480 = image_metrics(current, target_480)
        mse_256, mae_256, rmse_256 = image_metrics(current_256_phys, target_256_phys)

        item = {
            "iter": it,
            "image_mse_256": mse_256,
            "image_mae_256": mae_256,
            "image_rmse_256": rmse_256,
            "image_psnr_256": psnr_from_mse(mse_256, args.speed_max - args.speed_min),

            "image_mse_480": mse_480,
            "image_mae_480": mae_480,
            "image_rmse_480": rmse_480,
            "image_psnr_480": psnr_from_mse(mse_480, args.speed_max - args.speed_min),

            "dobs_rel_l1": rel_l1(dobs_current, dobs_gt),
            "dobs_rel_l2": rel_l2(dobs_current, dobs_gt),
            "dobs_abs_loss": compute_meas_loss(dobs_current, dobs_gt),
        }
        item.update(info)
        history.append(item)

        for k, v in item.items():
            if isinstance(v, float):
                print(f"{k}: {v:.8e}")
            else:
                print(f"{k}: {v}")

        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        if it % args.save_every == 0:
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
            history=np.array(history, dtype=object),
        )

    print("=" * 80)
    print("[Done]")
    print("=" * 80)
    print("saved to:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()