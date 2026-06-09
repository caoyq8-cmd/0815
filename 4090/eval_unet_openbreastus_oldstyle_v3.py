# ==============================================
# 【必须放在最顶部！任何 import 之前！】
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
# 修复 Windows 多进程崩溃
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except:
    pass
# ==============================================

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_openbreastus_oldstyle import OpenBreastUSOldStyleDataset
from train_unet_openbreastus_oldstyle_v3 import UNet, denormalize_target


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(data_range / math.sqrt(mse))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    ckpt_path = r"./results_unet_oldstyle_v3/checkpoints/best.pth"
    output_dir = r"./results_unet_oldstyle_v3/eval"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ensure_dir(output_dir)

    # 🔴 修复 weights_only 警告
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]

    test_set = OpenBreastUSOldStyleDataset(
        root_dir=cfg["data_root"],
        split="test",
        normalize_input=True,
        normalize_target=False,
    )
    # 🔴 修复 Windows 多进程崩溃：num_workers=0
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    model = UNet(in_channels=2, out_channels=1, base_ch=32).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    target_min = cfg["target_min"]
    target_max = cfg["target_max"]

    mse_list = []
    mae_list = []
    rmse_list = []
    psnr_list = []

    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            pred_norm = model(x)
            pred = denormalize_target(pred_norm, target_min, target_max)

            mse = F.mse_loss(pred, y).item()
            mae = F.l1_loss(pred, y).item()
            rmse = mse ** 0.5
            psnr = compute_psnr(pred, y, data_range=target_max - target_min)

            mse_list.append(mse)
            mae_list.append(mae)
            rmse_list.append(rmse)
            psnr_list.append(psnr)

            if idx < 12:
                x_cpu = x.cpu()[0]
                y_cpu = y.cpu()[0, 0]
                pred_cpu = pred.cpu()[0, 0]
                err = pred_cpu - y_cpu

                fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
                axes[0].imshow(x_cpu[0], cmap="gray")
                axes[0].set_title("input ch0")

                axes[1].imshow(x_cpu[1], cmap="gray")
                axes[1].set_title("input ch1")

                axes[2].imshow(y_cpu, cmap="magma")
                axes[2].set_title("gt")

                axes[3].imshow(pred_cpu, cmap="magma")
                axes[3].set_title("pred")

                im = axes[4].imshow(err, cmap="bwr")
                axes[4].set_title("error")

                for ax in axes:
                    ax.axis("off")

                fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"sample_{idx:03d}.png"), dpi=150)
                plt.close(fig)

    metrics = {
        "mse": float(np.mean(mse_list)),
        "mae": float(np.mean(mae_list)),
        "rmse": float(np.mean(rmse_list)),
        "psnr": float(np.mean(psnr_list)),
        "num_test": len(test_set),
        "checkpoint_epoch": int(ckpt["epoch"]),
    }

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Evaluation finished.")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()