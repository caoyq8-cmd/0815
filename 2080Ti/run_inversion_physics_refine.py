import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import math
import argparse
from dataclasses import dataclass, asdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset_openbreastus_oldstyle import OpenBreastUSOldStyleDataset
from train_inversionnet_baseline import InversionNetBaseline, denormalize_target, normalize_target


# =========================
# 配置
# =========================
@dataclass
class RefineConfig:
    data_root: str = "/home/featurize/datasets/90024ebe-ceca-4e0d-aab7-55f496b4b5f5"
    inversion_ckpt: str = "./ablation_runs/inversionnet_b32/checkpoints/best.pth"
    output_dir: str = "./ablation_runs/inversion_physics_refine"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0

    target_min: float = 1400.0
    target_max: float = 1600.0

    num_steps: int = 30
    step_size: float = 0.05
    tv_weight: float = 0.001
    clamp_each_step: bool = True

    eval_indices: str = "1,2,3,4,5,6,7,8,9,10"
    seed: int = 42


# =========================
# 通用工具
# =========================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


# =========================
# 观测代理算子
# 说明：
# 这是“可诊断过渡实验”的 forward surrogate，
# 不等价于真实 CBS 正演，但具备：
# 1) 可微
# 2) 输入为速度图，输出为 2 通道观测风格张量
# 3) 能让你先检验基于观测一致性的细化是否有效
# =========================
class SimpleForwardSurrogate(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 128
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 64
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(size=(256, 256), mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 2, kernel_size=1, stride=1, padding=0),
        )

        # 初始化为较平稳的小权重，避免一开始梯度太炸
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# =========================
# 加载 inversion 模型
# =========================
def load_inversion_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    inv_cfg = ckpt["config"]

    model = InversionNetBaseline(
        in_channels=2,
        out_channels=1,
        base_ch=inv_cfg["base_ch"],
        bottleneck_blocks=inv_cfg["bottleneck_blocks"],
        dropout=inv_cfg["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# =========================
# 可视化
# =========================
def save_case_figure(x_cpu, gt_cpu, inv_cpu, refined_cpu, save_path):
    err_inv = inv_cpu - gt_cpu
    err_ref = refined_cpu - gt_cpu

    fig, axes = plt.subplots(1, 7, figsize=(23, 3.8))
    axes[0].imshow(x_cpu[0], cmap="gray")
    axes[0].set_title("input ch0")

    axes[1].imshow(x_cpu[1], cmap="gray")
    axes[1].set_title("input ch1")

    axes[2].imshow(gt_cpu, cmap="magma")
    axes[2].set_title("gt")

    axes[3].imshow(inv_cpu, cmap="magma")
    axes[3].set_title("inv pred")

    axes[4].imshow(err_inv, cmap="bwr")
    axes[4].set_title("inv error")

    axes[5].imshow(refined_cpu, cmap="magma")
    axes[5].set_title("refined pred")

    im = axes[6].imshow(err_ref, cmap="bwr")
    axes[6].set_title("refined error")

    for ax in axes:
        ax.axis("off")

    fig.colorbar(im, ax=axes[6], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def save_profile_figure(gt_cpu, inv_cpu, refined_cpu, save_path):
    h, w = gt_cpu.shape
    row = h // 2
    col = w // 2

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(gt_cpu[row, :], label="gt")
    plt.plot(inv_cpu[row, :], label="inv")
    plt.plot(refined_cpu[row, :], label="refined")
    plt.title("center row profile")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(gt_cpu[:, col], label="gt")
    plt.plot(inv_cpu[:, col], label="inv")
    plt.plot(refined_cpu[:, col], label="refined")
    plt.title("center col profile")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_loss_curve(loss_dict, save_path):
    plt.figure(figsize=(8, 5))
    for k, v in loss_dict.items():
        plt.plot(v, label=k)
    plt.xlabel("step")
    plt.title("Refinement losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =========================
# 单样本物理细化
# =========================
def refine_one_sample(
    inversion_model,
    forward_model,
    x_meas,
    gt,
    cfg,
    case_dir,
):
    device = cfg.device
    data_range = cfg.target_max - cfg.target_min

    with torch.no_grad():
        inv_pred_norm = inversion_model(x_meas)
        inv_pred = denormalize_target(inv_pred_norm, cfg.target_min, cfg.target_max)

    # 优化变量用 norm 域更稳
    x_var = inv_pred_norm.detach().clone().requires_grad_(True)

    optimizer = torch.optim.SGD([x_var], lr=cfg.step_size)

    history = {
        "phys_l1": [],
        "tv": [],
        "total": [],
        "mse_to_gt": [],
        "psnr_to_gt": [],
    }

    for step in range(cfg.num_steps):
        optimizer.zero_grad()

        # forward surrogate 作用在物理量纲图上
        current_phys = denormalize_target(x_var, cfg.target_min, cfg.target_max)
        pred_meas = forward_model(current_phys)

        phys_l1 = F.l1_loss(pred_meas, x_meas)
        tv = total_variation_loss(current_phys)
        total = phys_l1 + cfg.tv_weight * tv

        total.backward()
        optimizer.step()

        if cfg.clamp_each_step:
            with torch.no_grad():
                x_var.clamp_(-1.0, 1.0)

        with torch.no_grad():
            current_phys_eval = denormalize_target(x_var, cfg.target_min, cfg.target_max)
            mse = F.mse_loss(current_phys_eval, gt).item()
            psnr = compute_psnr(current_phys_eval, gt, data_range=data_range)

        history["phys_l1"].append(float(phys_l1.item()))
        history["tv"].append(float(tv.item()))
        history["total"].append(float(total.item()))
        history["mse_to_gt"].append(float(mse))
        history["psnr_to_gt"].append(float(psnr))

    with torch.no_grad():
        refined_pred = denormalize_target(x_var, cfg.target_min, cfg.target_max)

        inv_metrics = {
            "mse": F.mse_loss(inv_pred, gt).item(),
            "mae": F.l1_loss(inv_pred, gt).item(),
            "rmse": math.sqrt(F.mse_loss(inv_pred, gt).item()),
            "psnr": compute_psnr(inv_pred, gt, data_range),
            "ssim": compute_ssim(inv_pred, gt, data_range),
        }

        refined_metrics = {
            "mse": F.mse_loss(refined_pred, gt).item(),
            "mae": F.l1_loss(refined_pred, gt).item(),
            "rmse": math.sqrt(F.mse_loss(refined_pred, gt).item()),
            "psnr": compute_psnr(refined_pred, gt, data_range),
            "ssim": compute_ssim(refined_pred, gt, data_range),
        }

    # 保存图
    x_cpu = x_meas.cpu()[0].numpy()
    gt_cpu = gt.cpu()[0, 0].numpy()
    inv_cpu = inv_pred.cpu()[0, 0].numpy()
    refined_cpu = refined_pred.cpu()[0, 0].numpy()

    save_case_figure(
        x_cpu=x_cpu,
        gt_cpu=gt_cpu,
        inv_cpu=inv_cpu,
        refined_cpu=refined_cpu,
        save_path=os.path.join(case_dir, "compare.png"),
    )

    save_profile_figure(
        gt_cpu=gt_cpu,
        inv_cpu=inv_cpu,
        refined_cpu=refined_cpu,
        save_path=os.path.join(case_dir, "profile.png"),
    )

    save_loss_curve(
        history,
        save_path=os.path.join(case_dir, "loss_curve.png"),
    )

    with open(os.path.join(case_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    with open(os.path.join(case_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "inv": inv_metrics,
                "refined": refined_metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return inv_metrics, refined_metrics


# =========================
# 主函数
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--inversion_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--step_size", type=float, default=0.05)
    parser.add_argument("--tv_weight", type=float, default=0.001)

    parser.add_argument("--eval_indices", type=str, default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = RefineConfig(
        data_root=args.data_root,
        inversion_ckpt=args.inversion_ckpt,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        step_size=args.step_size,
        tv_weight=args.tv_weight,
        eval_indices=args.eval_indices,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    indices = [int(x) for x in cfg.eval_indices.split(",")]

    test_set_full = OpenBreastUSOldStyleDataset(
        root_dir=cfg.data_root,
        split="test",
        normalize_input=True,
        normalize_target=False,
    )
    test_set = Subset(test_set_full, indices)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    print(f"num_eval_cases = {len(test_set)}")
    print(f"device         = {cfg.device}")

    inversion_model = load_inversion_model(cfg.inversion_ckpt, cfg.device)
    forward_model = SimpleForwardSurrogate().to(cfg.device).eval()

    inv_list = []
    refined_list = []

    for i, (x_meas, gt) in enumerate(test_loader):
        global_idx = indices[i]
        case_dir = os.path.join(cfg.output_dir, f"sample_{global_idx:03d}")
        ensure_dir(case_dir)

        x_meas = x_meas.to(cfg.device)
        gt = gt.to(cfg.device)

        inv_metrics, refined_metrics = refine_one_sample(
            inversion_model=inversion_model,
            forward_model=forward_model,
            x_meas=x_meas,
            gt=gt,
            cfg=cfg,
            case_dir=case_dir,
        )

        inv_list.append(inv_metrics)
        refined_list.append(refined_metrics)

        print(
            f"[sample {global_idx:03d}] "
            f"inv_psnr={inv_metrics['psnr']:.4f}, "
            f"refined_psnr={refined_metrics['psnr']:.4f}, "
            f"inv_mse={inv_metrics['mse']:.4f}, "
            f"refined_mse={refined_metrics['mse']:.4f}"
        )

    def summarize(metric_list):
        keys = ["mse", "mae", "rmse", "psnr", "ssim"]
        out = {}
        for k in keys:
            vals = [d[k] for d in metric_list]
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals))
        return out

    summary = {
        "num_cases": len(indices),
        "inv": summarize(inv_list),
        "refined": summarize(refined_list),
    }

    with open(os.path.join(cfg.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()