import os
import json
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_dobs_matrix_no import DobsMatrixCNN


# ============================================================
# basic utils
# ============================================================

def to_2d(x):
    x = np.asarray(x)
    while x.ndim > 2:
        x = x[0]
    return x.astype(np.float32)


def resize_np(x, size):
    x_t = torch.from_numpy(x.astype(np.float32))[None, None]
    y_t = F.interpolate(x_t, size=(size, size), mode="bilinear", align_corners=False)
    return y_t[0, 0].cpu().numpy().astype(np.float32)


def image_metrics(pred, target):
    diff = pred.astype(np.float32) - target.astype(np.float32)
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    return mse, mae, rmse


def rel_l1_complex(pred, target):
    return float(np.mean(np.abs(pred - target)) / (np.mean(np.abs(target)) + 1e-12))


def rel_l2_complex(pred, target):
    return float(
        np.sqrt(np.mean(np.abs(pred - target) ** 2))
        / (np.sqrt(np.mean(np.abs(target) ** 2)) + 1e-12)
    )


def abs_loss_complex(pred, target):
    return float(np.mean(np.abs(pred - target)))


def normalize_grad(g, eps=1e-12):
    rms = float(np.sqrt(np.mean(g ** 2)) + eps)
    return g / rms, rms


def make_coord_maps(size, device):
    yy = torch.linspace(-1.0, 1.0, size, device=device).view(1, 1, size, 1)
    xx = torch.linspace(-1.0, 1.0, size, device=device).view(1, 1, 1, size)

    y_map = yy.repeat(1, 1, 1, size)
    x_map = xx.repeat(1, 1, size, 1)

    return y_map, x_map


# ============================================================
# neural measurement model
# ============================================================

def load_residual_measurement_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt["args"]

    model = DobsMatrixCNN(
        in_ch=3,
        base_ch=ckpt_args.get("base_ch", 32),
        num_sources=64,
        num_receivers=64,
        spatial_pool=ckpt_args.get("spatial_pool", 8),
        head_dim=ckpt_args.get("head_dim", 1024),
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    return model, ckpt_args


def predict_dobs_torch(
    model,
    speed_240,
    mean_real,
    mean_imag,
    residual_scale,
    speed_center,
    speed_scale,
    y_map,
    x_map,
):
    """
    speed_240: [1,1,H,W], physical m/s
    返回 pred_real, pred_imag: [64,64]
    """
    speed_norm = (speed_240 - speed_center) / speed_scale
    inp = torch.cat([speed_norm, y_map, x_map], dim=1)

    out = model(inp)  # [1,2,64,64]

    pred_res_real = out[0, 0] * residual_scale
    pred_res_imag = out[0, 1] * residual_scale

    pred_real = mean_real + pred_res_real
    pred_imag = mean_imag + pred_res_imag

    return pred_real, pred_imag


def neural_loss_and_pred(
    model,
    speed_240,
    target_real,
    target_imag,
    mean_real,
    mean_imag,
    residual_scale,
    speed_center,
    speed_scale,
    y_map,
    x_map,
    lambda_l1=1.0,
    lambda_mse=1.0,
):
    pred_real, pred_imag = predict_dobs_torch(
        model,
        speed_240,
        mean_real,
        mean_imag,
        residual_scale,
        speed_center,
        speed_scale,
        y_map,
        x_map,
    )

    dr = pred_real - target_real
    di = pred_imag - target_imag

    complex_abs = torch.sqrt(dr ** 2 + di ** 2 + 1e-12)

    loss_l1 = complex_abs.mean()
    loss_mse = (dr ** 2 + di ** 2).mean()

    loss = lambda_l1 * loss_l1 + lambda_mse * loss_mse

    return loss, pred_real, pred_imag


@torch.no_grad()
def eval_neural_np(
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
    lambda_l1=1.0,
    lambda_mse=1.0,
):
    speed = torch.from_numpy(speed_np.astype(np.float32))[None, None].to(device)

    target_real = torch.from_numpy(target_dobs.real.astype(np.float32)).to(device)
    target_imag = torch.from_numpy(target_dobs.imag.astype(np.float32)).to(device)

    loss, pred_real, pred_imag = neural_loss_and_pred(
        model,
        speed,
        target_real,
        target_imag,
        mean_real,
        mean_imag,
        residual_scale,
        speed_center,
        speed_scale,
        y_map,
        x_map,
        lambda_l1=lambda_l1,
        lambda_mse=lambda_mse,
    )

    pred_np = (
        pred_real.detach().cpu().numpy()
        + 1j * pred_imag.detach().cpu().numpy()
    ).astype(np.complex64)

    return float(loss.detach().cpu()), pred_np


def compute_neural_grad(
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
    lambda_l1=1.0,
    lambda_mse=1.0,
):
    speed = torch.from_numpy(speed_np.astype(np.float32))[None, None].to(device)
    speed.requires_grad_(True)

    target_real = torch.from_numpy(target_dobs.real.astype(np.float32)).to(device)
    target_imag = torch.from_numpy(target_dobs.imag.astype(np.float32)).to(device)

    loss, pred_real, pred_imag = neural_loss_and_pred(
        model,
        speed,
        target_real,
        target_imag,
        mean_real,
        mean_imag,
        residual_scale,
        speed_center,
        speed_scale,
        y_map,
        x_map,
        lambda_l1=lambda_l1,
        lambda_mse=lambda_mse,
    )

    loss.backward()

    grad = speed.grad.detach().cpu().numpy()[0, 0].astype(np.float32)

    pred_np = (
        pred_real.detach().cpu().numpy()
        + 1j * pred_imag.detach().cpu().numpy()
    ).astype(np.complex64)

    return float(loss.detach().cpu()), grad, pred_np


# ============================================================
# optional true CBS check
# ============================================================

@torch.no_grad()
def forward_cbs_dobs(sos_480, src_indices, rec_indices, args, device):
    from cbs_model import ConvergentBornSeries_Batch

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

    u = model(max_iters=args.cbs_iters)

    rec_t = torch.from_numpy(rec_indices.astype(np.int64)).long().to(device)
    dobs = u[0, :, rec_t[:, 0], rec_t[:, 1]]
    dobs_np = dobs.detach().cpu().numpy().astype(np.complex64)

    del model, u, sos
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dobs_np


# ============================================================
# correction
# ============================================================

def correction_step(
    current,
    condition,
    target_dobs,
    model,
    mean_real,
    mean_imag,
    residual_scale,
    speed_center,
    speed_scale,
    y_map,
    x_map,
    args,
    device,
):
    current_loss, grad, pred_current = compute_neural_grad(
        model,
        current,
        target_dobs,
        mean_real,
        mean_imag,
        residual_scale,
        speed_center,
        speed_scale,
        y_map,
        x_map,
        device,
        lambda_l1=args.lambda_l1,
        lambda_mse=args.lambda_mse,
    )

    grad_norm, grad_rms = normalize_grad(grad)

    if args.prior_tether > 0:
        prior_grad = current - condition
        prior_norm, prior_rms = normalize_grad(prior_grad)
        total_grad = grad_norm + args.prior_tether * prior_norm
    else:
        prior_rms = 0.0
        total_grad = grad_norm

    candidates = []

    candidates.append({
        "name": "zero",
        "speed": current.copy(),
        "step_used": 0.0,
        "sign": "0",
    })

    for fac in args.step_factors:
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
            "speed": cand_minus,
            "step_used": step,
            "sign": "-",
        })
        candidates.append({
            "name": f"+{fac}",
            "speed": cand_plus,
            "step_used": step,
            "sign": "+",
        })

    best = None

    for cand in candidates:
        if cand["name"] == "zero":
            loss_cand = current_loss
            pred_cand = pred_current
        else:
            loss_cand, pred_cand = eval_neural_np(
                model,
                cand["speed"],
                target_dobs,
                mean_real,
                mean_imag,
                residual_scale,
                speed_center,
                speed_scale,
                y_map,
                x_map,
                device,
                lambda_l1=args.lambda_l1,
                lambda_mse=args.lambda_mse,
            )

        if best is None or loss_cand < best["loss"]:
            best = {
                "speed": cand["speed"],
                "loss": loss_cand,
                "pred": pred_cand,
                "name": cand["name"],
                "step_used": cand["step_used"],
                "sign": cand["sign"],
            }

    chosen = best["speed"]
    chosen_pred = best["pred"]

    info = {
        "neural_loss_before": current_loss,
        "neural_loss_after": best["loss"],
        "neural_loss_decreased": float(best["loss"] <= current_loss),
        "chosen_candidate": best["name"],
        "chosen_sign": best["sign"],
        "chosen_step_used": best["step_used"],
        "grad_rms": grad_rms,
        "prior_rms": prior_rms,
        "update_abs_mean": float(np.mean(np.abs(chosen - current))),
        "update_abs_max": float(np.max(np.abs(chosen - current))),
        "neural_dobs_rel_l1": rel_l1_complex(chosen_pred, target_dobs),
        "neural_dobs_rel_l2": rel_l2_complex(chosen_pred, target_dobs),
        "neural_dobs_abs_loss": abs_loss_complex(chosen_pred, target_dobs),
    }

    return chosen, info, chosen_pred, grad


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_path", type=str, required=True)
    parser.add_argument("--condition_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--mean_dobs_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_iters", type=int, default=20)
    parser.add_argument("--step_size_mps", type=float, default=1.0)
    parser.add_argument("--prior_tether", type=float, default=0.1)
    parser.add_argument("--step_factors", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1])

    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_mse", type=float, default=1.0)

    parser.add_argument("--speed_min", type=float, default=1400.0)
    parser.add_argument("--speed_max", type=float, default=1605.0)

    parser.add_argument("--check_true_cbs", action="store_true")
    parser.add_argument("--true_cbs_every", type=int, default=5)

    parser.add_argument("--frequency", type=float, default=500000.0)
    parser.add_argument("--cbs_iters", type=int, default=80)
    parser.add_argument("--boundary_width", type=int, default=300)
    parser.add_argument("--boundary_strength", type=float, default=225.0)
    parser.add_argument("--boundary_type", type=str, default="PML3")

    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    model, ckpt_args = load_residual_measurement_model(args.ckpt_path, device)

    image_size = ckpt_args.get("image_size", 240)
    speed_center = ckpt_args.get("speed_center", 1500.0)
    speed_scale = ckpt_args.get("speed_scale", 100.0)
    residual_scale = ckpt_args.get("residual_scale", 0.02)

    print("model image_size =", image_size)
    print("residual_scale   =", residual_scale)

    mean_dobs = np.load(args.mean_dobs_path)["mean_dobs"].astype(np.complex64)
    mean_real = torch.from_numpy(mean_dobs.real.astype(np.float32)).to(device)
    mean_imag = torch.from_numpy(mean_dobs.imag.astype(np.float32)).to(device)

    y_map, x_map = make_coord_maps(image_size, device)

    sc = np.load(args.sample_path)
    cc = np.load(args.condition_path)

    target_480 = sc["target_480"].astype(np.float32)
    target_dobs = sc["dobs_complex"].astype(np.complex64)

    src_indices = sc["src_indices"].astype(np.int64)
    rec_indices = sc["rec_indices"].astype(np.int64)

    condition_256_img = to_2d(cc["condition_speed"])
    target_256_img = to_2d(cc["target_speed"])

    # coordinate convention:
    # condition cache is image coordinate, CBS data is physics coordinate
    condition_256_phys = condition_256_img.T
    target_256_phys = target_256_img.T

    # sanity check with self target_256
    if "target_256" in sc.files:
        align_mse = float(np.mean((target_256_phys - sc["target_256"].astype(np.float32)) ** 2))
        print("target alignment MSE =", align_mse)

    condition_480 = resize_np(condition_256_phys, 480)
    condition_480 = np.clip(condition_480, args.speed_min, args.speed_max).astype(np.float32)

    target_240 = resize_np(target_480, image_size)
    condition_240 = resize_np(condition_480, image_size)

    current = condition_240.copy()
    condition_fixed = condition_240.copy()

    init_loss, init_pred = eval_neural_np(
        model,
        current,
        target_dobs,
        mean_real,
        mean_imag,
        residual_scale,
        speed_center,
        speed_scale,
        y_map,
        x_map,
        device,
        lambda_l1=args.lambda_l1,
        lambda_mse=args.lambda_mse,
    )

    init_mse_240, init_mae_240, init_rmse_240 = image_metrics(current, target_240)
    current_480 = resize_np(current, 480)
    init_mse_480, init_mae_480, init_rmse_480 = image_metrics(current_480, target_480)

    history = []

    init_row = {
        "iter": 0,
        "image_mse_240": init_mse_240,
        "image_mae_240": init_mae_240,
        "image_rmse_240": init_rmse_240,
        "image_mse_480_up": init_mse_480,
        "image_mae_480_up": init_mae_480,
        "image_rmse_480_up": init_rmse_480,
        "neural_loss": init_loss,
        "neural_dobs_rel_l1": rel_l1_complex(init_pred, target_dobs),
        "neural_dobs_rel_l2": rel_l2_complex(init_pred, target_dobs),
        "neural_dobs_abs_loss": abs_loss_complex(init_pred, target_dobs),
    }

    if args.check_true_cbs:
        true_pred = forward_cbs_dobs(current_480, src_indices, rec_indices, args, device)
        init_row["true_cbs_dobs_rel_l1"] = rel_l1_complex(true_pred, target_dobs)
        init_row["true_cbs_dobs_rel_l2"] = rel_l2_complex(true_pred, target_dobs)
        init_row["true_cbs_dobs_abs_loss"] = abs_loss_complex(true_pred, target_dobs)

    history.append(init_row)

    print("=" * 80)
    print("[Initial]")
    print("=" * 80)
    for k, v in init_row.items():
        print(f"{k}: {v}")

    for it in range(1, args.num_iters + 1):
        print("=" * 80)
        print(f"[Iteration {it}/{args.num_iters}]")
        print("=" * 80)

        current, info, pred_current, grad = correction_step(
            current,
            condition_fixed,
            target_dobs,
            model,
            mean_real,
            mean_imag,
            residual_scale,
            speed_center,
            speed_scale,
            y_map,
            x_map,
            args,
            device,
        )

        mse_240, mae_240, rmse_240 = image_metrics(current, target_240)
        current_480 = resize_np(current, 480)
        mse_480, mae_480, rmse_480 = image_metrics(current_480, target_480)

        row = {
            "iter": it,
            "image_mse_240": mse_240,
            "image_mae_240": mae_240,
            "image_rmse_240": rmse_240,
            "image_mse_480_up": mse_480,
            "image_mae_480_up": mae_480,
            "image_rmse_480_up": rmse_480,
        }
        row.update(info)

        if args.check_true_cbs and (it % args.true_cbs_every == 0 or it == args.num_iters):
            true_pred = forward_cbs_dobs(current_480, src_indices, rec_indices, args, device)
            row["true_cbs_dobs_rel_l1"] = rel_l1_complex(true_pred, target_dobs)
            row["true_cbs_dobs_rel_l2"] = rel_l2_complex(true_pred, target_dobs)
            row["true_cbs_dobs_abs_loss"] = abs_loss_complex(true_pred, target_dobs)

        history.append(row)

        for k, v in row.items():
            if isinstance(v, float):
                print(f"{k}: {v:.8e}")
            else:
                print(f"{k}: {v}")

        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        np.savez_compressed(
            os.path.join(args.output_dir, "final_result.npz"),
            corrected_240=current.astype(np.float32),
            corrected_480_up=current_480.astype(np.float32),
            condition_240=condition_fixed.astype(np.float32),
            condition_480=condition_480.astype(np.float32),
            target_240=target_240.astype(np.float32),
            target_480=target_480.astype(np.float32),
            target_dobs=target_dobs.astype(np.complex64),
            neural_pred_dobs=pred_current.astype(np.complex64),
        )

    print("[Done]")
    print("saved to:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()