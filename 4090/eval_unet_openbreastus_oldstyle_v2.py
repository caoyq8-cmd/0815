import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===================== 修复 Windows 多进程崩溃 =====================
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except:
    pass
# =================================================================

import json
import math
import os
from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_openbreastus_oldstyle import OpenBreastUSOldStyleDataset
from train_unet_openbreastus_oldstyle_v2 import SmallUNet, WrappedDataset, denormalize_y


@dataclass
class EvalConfig:
    data_root: str = r"D:\OpenBreastUS_processed_oldstyle_2x256x256_fullresize"
    checkpoint: str = "./results_openbreastus_oldstyle_unet_v2/checkpoints/best.pth"
    output_dir: str = "./eval_openbreastus_oldstyle_unet_v2"
    batch_size: int = 4
    num_workers: int = 0  # 🔴 修复多进程 pickle 错误
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    normalize_input: bool = True
    max_save: int = 20


def compute_metrics(pred_phys: torch.Tensor, target_phys: torch.Tensor) -> Dict[str, float]:
    mse = torch.mean((pred_phys - target_phys) ** 2).item()
    mae = torch.mean(torch.abs(pred_phys - target_phys)).item()
    rmse = math.sqrt(mse)
    peak = max(target_phys.max().item() - target_phys.min().item(), 1e-8)
    psnr = 20.0 * math.log10(peak / max(rmse, 1e-8))
    return {"mse": mse, "mae": mae, "rmse": rmse, "psnr": psnr}


def save_triplet(x, gt, pred, path):
    x = x.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()[0]
    pred = pred.detach().cpu().numpy()[0]
    err = pred - gt
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes[0].imshow(x[0], cmap="gray")
    axes[0].set_title("input ch0")
    axes[1].imshow(x[1], cmap="gray")
    axes[1].set_title("input ch1")
    axes[2].imshow(gt, cmap="inferno")
    axes[2].set_title("gt")
    axes[3].imshow(pred, cmap="inferno")
    axes[3].set_title("pred")
    im = axes[4].imshow(err, cmap="bwr")
    axes[4].set_title("error")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# 🔴 关键修复：把动态创建的类放到顶层，解决 PicklingError
class DummyCfg:
    pass


def main():
    cfg = EvalConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 🔴 修复 weights_only 警告
    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
    train_cfg_dict = ckpt["config"]
    with open(os.path.join(cfg.output_dir, "loaded_train_config.json"), "w", encoding="utf-8") as f:
        json.dump(train_cfg_dict, f, indent=2, ensure_ascii=False)

    # 🔴 使用顶层类
    dummy_cfg = DummyCfg()
    for k, v in train_cfg_dict.items():
        setattr(dummy_cfg, k, v)

    base = OpenBreastUSOldStyleDataset(
        root_dir=cfg.data_root,
        split="test",
        normalize_input=cfg.normalize_input,
        normalize_target=False,
    )
    ds = WrappedDataset(base, dummy_cfg)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = SmallUNet(in_channels=2, out_channels=1).to(cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total = {"mse": 0.0, "mae": 0.0, "rmse": 0.0, "psnr": 0.0}
    count = 0
    save_count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            pred = model(x)
            pred_phys = denormalize_y(pred, dummy_cfg)
            y_phys = denormalize_y(y, dummy_cfg)

            for i in range(x.shape[0]):
                m = compute_metrics(pred_phys[i:i+1], y_phys[i:i+1])
                for k in total:
                    total[k] += m[k]
                count += 1
                if save_count < cfg.max_save:
                    save_triplet(x[i], y_phys[i], pred_phys[i], os.path.join(cfg.output_dir, f"sample_{save_count:03d}.png"))
                    save_count += 1

    avg = {k: v / count for k, v in total.items()}
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(avg, f, indent=2, ensure_ascii=False)

    print("Evaluation finished.")
    for k, v in avg.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()