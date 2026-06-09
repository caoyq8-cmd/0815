import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except:
    pass

import json
import math
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset_openbreastus_oldstyle import OpenBreastUSOldStyleDataset
from train_hf_refiner import (
    load_frozen_inversion_model,
    HFRefinerUNet,
    TwoStageHFModel,
    denormalize_target,
    highpass,
)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(data_range / math.sqrt(mse))


def gaussian_window(window_size=11, sigma=1.5, channels=1, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g)
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)
    window_2d = window_2d.repeat(channels, 1, 1, 1)
    return window_2d


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float, window_size=11, sigma=1.5) -> float:
    device = pred.device
    channels = pred.shape[1]
    window = gaussian_window(window_size, sigma, channels, device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12
    )
    return float(ssim_map.mean().item())


def save_case_figure(x_cpu, y_cpu, inv_cpu, final_cpu, hf_gt, hf_inv, hf_final, save_path):
    err_inv = inv_cpu - y_cpu
    err_final = final_cpu - y_cpu

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes[0, 0].imshow(x_cpu[0], cmap="gray")
    axes[0, 0].set_title("input ch0")

    axes[0, 1].imshow(y_cpu, cmap="magma")
    axes[0, 1].set_title("gt")

    axes[0, 2].imshow(inv_cpu, cmap="magma")
    axes[0, 2].set_title("inv pred")

    axes[0, 3].imshow(final_cpu, cmap="magma")
    axes[0, 3].set_title("final pred")

    axes[1, 0].imshow(x_cpu[1], cmap="gray")
    axes[1, 0].set_title("input ch1")

    axes[1, 1].imshow(hf_gt, cmap="bwr")
    axes[1, 1].set_title("gt highpass")

    axes[1, 2].imshow(hf_inv, cmap="bwr")
    axes[1, 2].set_title("inv highpass")

    axes[1, 3].imshow(hf_final, cmap="bwr")
    axes[1, 3].set_title("final highpass")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def save_profile_figure(y_cpu, inv_cpu, final_cpu, save_path):
    h, w = y_cpu.shape
    row = h // 2
    col = w // 2

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(y_cpu[row, :], label="gt")
    plt.plot(inv_cpu[row, :], label="inv")
    plt.plot(final_cpu[row, :], label="final")
    plt.title("center row profile")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(y_cpu[:, col], label="gt")
    plt.plot(inv_cpu[:, col], label="inv")
    plt.plot(final_cpu[:, col], label="final")
    plt.title("center col profile")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eval_indices", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.output_dir)

    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    test_set_full = OpenBreastUSOldStyleDataset(
        root_dir=cfg["data_root"],
        split="test",
        normalize_input=True,
        normalize_target=False,
    )

    if args.eval_indices.strip():
        indices = [int(x) for x in args.eval_indices.split(",")]
        test_set = Subset(test_set_full, indices)
        used_indices = indices
    else:
        test_set = test_set_full
        used_indices = list(range(len(test_set_full)))

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    inversion_model = load_frozen_inversion_model(cfg["inversion_ckpt"], device)
    refiner_model = HFRefinerUNet(
        in_channels=3,
        out_channels=1,
        base_ch=cfg["refiner_base_ch"],
    ).to(device)

    model = TwoStageHFModel(inversion_model, refiner_model).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    target_min = cfg["target_min"]
    target_max = cfg["target_max"]
    data_range = target_max - target_min
    hf_kernel_size = cfg["hf_kernel_size"]

    case_metrics = []
    mse_inv_list, mae_inv_list, rmse_inv_list, psnr_inv_list, ssim_inv_list = [], [], [], [], []
    mse_final_list, mae_final_list, rmse_final_list, psnr_final_list, ssim_final_list = [], [], [], [], []
    hf_inv_list, hf_final_list = [], []

    with torch.no_grad():
        for local_idx, (x, y) in enumerate(test_loader):
            global_idx = used_indices[local_idx]

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            inv_pred = denormalize_target(outputs["inv_pred_norm"], target_min, target_max)
            final_pred = denormalize_target(outputs["final_pred_norm"], target_min, target_max)

            mse_inv = F.mse_loss(inv_pred, y).item()
            mae_inv = F.l1_loss(inv_pred, y).item()
            rmse_inv = mse_inv ** 0.5
            psnr_inv = compute_psnr(inv_pred, y, data_range=data_range)
            ssim_inv = compute_ssim(inv_pred, y, data_range=data_range)

            mse_final = F.mse_loss(final_pred, y).item()
            mae_final = F.l1_loss(final_pred, y).item()
            rmse_final = mse_final ** 0.5
            psnr_final = compute_psnr(final_pred, y, data_range=data_range)
            ssim_final = compute_ssim(final_pred, y, data_range=data_range)

            hf_inv = F.l1_loss(highpass(inv_pred, hf_kernel_size), highpass(y, hf_kernel_size)).item()
            hf_final = F.l1_loss(highpass(final_pred, hf_kernel_size), highpass(y, hf_kernel_size)).item()

            mse_inv_list.append(mse_inv)
            mae_inv_list.append(mae_inv)
            rmse_inv_list.append(rmse_inv)
            psnr_inv_list.append(psnr_inv)
            ssim_inv_list.append(ssim_inv)

            mse_final_list.append(mse_final)
            mae_final_list.append(mae_final)
            rmse_final_list.append(rmse_final)
            psnr_final_list.append(psnr_final)
            ssim_final_list.append(ssim_final)

            hf_inv_list.append(hf_inv)
            hf_final_list.append(hf_final)

            case_metrics.append({
                "sample_index": global_idx,
                "inv": {
                    "mse": mse_inv,
                    "mae": mae_inv,
                    "rmse": rmse_inv,
                    "psnr": psnr_inv,
                    "ssim": ssim_inv,
                    "hf_l1": hf_inv,
                },
                "final": {
                    "mse": mse_final,
                    "mae": mae_final,
                    "rmse": rmse_final,
                    "psnr": psnr_final,
                    "ssim": ssim_final,
                    "hf_l1": hf_final,
                },
            })

            x_cpu = x.cpu()[0].numpy()
            y_cpu = y.cpu()[0, 0].numpy()
            inv_cpu = inv_pred.cpu()[0, 0].numpy()
            final_cpu = final_pred.cpu()[0, 0].numpy()

            hf_gt_cpu = highpass(y.cpu(), hf_kernel_size)[0, 0].numpy()
            hf_inv_cpu = highpass(inv_pred.cpu(), hf_kernel_size)[0, 0].numpy()
            hf_final_cpu = highpass(final_pred.cpu(), hf_kernel_size)[0, 0].numpy()

            save_case_figure(
                x_cpu=x_cpu,
                y_cpu=y_cpu,
                inv_cpu=inv_cpu,
                final_cpu=final_cpu,
                hf_gt=hf_gt_cpu,
                hf_inv=hf_inv_cpu,
                hf_final=hf_final_cpu,
                save_path=os.path.join(args.output_dir, f"sample_{global_idx:03d}.png"),
            )
            save_profile_figure(
                y_cpu=y_cpu,
                inv_cpu=inv_cpu,
                final_cpu=final_cpu,
                save_path=os.path.join(args.output_dir, f"sample_{global_idx:03d}_profile.png"),
            )

    summary = {
        "num_test": len(test_set),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "inv": {
            "mse_mean": float(np.mean(mse_inv_list)),
            "mse_std": float(np.std(mse_inv_list)),
            "mae_mean": float(np.mean(mae_inv_list)),
            "mae_std": float(np.std(mae_inv_list)),
            "rmse_mean": float(np.mean(rmse_inv_list)),
            "rmse_std": float(np.std(rmse_inv_list)),
            "psnr_mean": float(np.mean(psnr_inv_list)),
            "psnr_std": float(np.std(psnr_inv_list)),
            "ssim_mean": float(np.mean(ssim_inv_list)),
            "ssim_std": float(np.std(ssim_inv_list)),
            "hf_l1_mean": float(np.mean(hf_inv_list)),
            "hf_l1_std": float(np.std(hf_inv_list)),
        },
        "final": {
            "mse_mean": float(np.mean(mse_final_list)),
            "mse_std": float(np.std(mse_final_list)),
            "mae_mean": float(np.mean(mae_final_list)),
            "mae_std": float(np.std(mae_final_list)),
            "rmse_mean": float(np.mean(rmse_final_list)),
            "rmse_std": float(np.std(rmse_final_list)),
            "psnr_mean": float(np.mean(psnr_final_list)),
            "psnr_std": float(np.std(psnr_final_list)),
            "ssim_mean": float(np.mean(ssim_final_list)),
            "ssim_std": float(np.std(ssim_final_list)),
            "hf_l1_mean": float(np.mean(hf_final_list)),
            "hf_l1_std": float(np.std(hf_final_list)),
        },
    }

    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "metrics_per_case.json"), "w", encoding="utf-8") as f:
        json.dump(case_metrics, f, indent=2, ensure_ascii=False)

    print("Evaluation finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()