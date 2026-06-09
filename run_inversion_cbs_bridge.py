import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import math
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.io import loadmat

from cbs_model import ConvergentBornSeries_Batch, ConvergentBornSeries_Batch_Adjoint
from utils_cbs import denormalize as cbs_denormalize
from train_inversionnet_baseline import InversionNetBaseline


# =========================================================
# 配置
# =========================================================
@dataclass
class BridgeConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    inversion_ckpt: str = "./2080Ti/ablation_runs/inversionnet_b32/checkpoints/best.pth"

    base_dir_dobs_eval: str = "Datasets_test/AI4Scup2_simulated_CBS_sparse/dobs_500k/eval"
    base_dir_speed_eval: str = "Datasets_test/AI4Scup2_simulated_CBS_sparse/speed/eval"
    aux_dir: str = "auxiliary_data"

    output_dir: str = "./2080Ti/ablation_runs/inversion_cbs_bridge"

    eval_index: int = 0
    measurement_mode: str = "sparse"
    frequency_hz: float = 500e3

    num_iters: int = 6
    step_size: float = 50.0
    grad_smooth_kernel: int = 9

    clip_min: float = 1400.0
    clip_max: float = 1605.0

    boundary_width: tuple = (300, 300)
    boundary_strength: float = 225.0
    boundary_type: str = "PML3"

    use_roi_mask: bool = True
    save_each_iter: bool = True

    seed: int = 42


# =========================================================
# 工具函数
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_psnr_torch(pred: torch.Tensor, gt: torch.Tensor, data_range: float) -> float:
    mse = F.mse_loss(pred, gt).item()
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


def calculate_ssim_torch(pred: torch.Tensor, gt: torch.Tensor, data_range: float, window_size=11, sigma=1.5) -> float:
    device = pred.device
    channels = pred.shape[1]
    window = gaussian_window(window_size, sigma, channels, device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(gt, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, padding=window_size // 2, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12
    )
    return float(ssim_map.mean().item())


def save_speed_map(tensor_480, save_path, vmin=1408.692, vmax=1595.1279, title=None):
    if tensor_480.ndim == 4:
        img = tensor_480[0, 0]
    elif tensor_480.ndim == 3:
        img = tensor_480[0]
    else:
        img = tensor_480

    plt.figure(figsize=(5, 5))
    plt.imshow(img.detach().cpu(), cmap="inferno", vmin=vmin, vmax=vmax)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_compare_figure(gt_480, inv_480, refined_480, save_path, vmin=1408.692, vmax=1595.1279):
    gt_img = gt_480[0, 0].detach().cpu()
    inv_img = inv_480[0, 0].detach().cpu()
    ref_img = refined_480[0, 0].detach().cpu()

    err_inv = inv_img - gt_img
    err_ref = ref_img - gt_img

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(gt_img, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title("GT 480")
    axes[1].imshow(inv_img, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("Inversion Init")
    axes[2].imshow(ref_img, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[2].set_title("CBS Refined")
    axes[3].imshow(err_inv, cmap="bwr")
    axes[3].set_title("Init Error")
    im4 = axes[4].imshow(err_ref, cmap="bwr")
    axes[4].set_title("Refined Error")

    for ax in axes:
        ax.axis("off")

    fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_curve(history, save_path):
    plt.figure(figsize=(8, 5))
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.xlabel("iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


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


def extract_receiver_data(u_batch, receiver_indices, receiver_mask=None):
    """
    u_batch: [N, M, H, W] complex
    receiver_indices: [L, 2]
    receiver_mask: [M, L] or None
    return: [N, M, L] complex
    """
    rec = u_batch[:, :, receiver_indices[:, 0], receiver_indices[:, 1]]
    if receiver_mask is not None:
        rec = rec * receiver_mask.to(u_batch.device)
    return rec


def build_geometry(aux_dir: str, measurement_mode: str, device: str):
    x_pos = loadmat(os.path.join(aux_dir, "x_pos.mat"))["x_pos256"]
    y_pos = loadmat(os.path.join(aux_dir, "y_pos.mat"))["y_pos256"]

    all_indices = torch.cat(
        (
            torch.tensor(x_pos.astype(np.int64)),
            torch.tensor(y_pos.astype(np.int64)),
        ),
        dim=1
    )

    if measurement_mode == "sparse":
        num_keep = 64
    elif measurement_mode == "sparse_2":
        num_keep = 32
    else:
        raise ValueError(f"Only sparse / sparse_2 are supported now, got {measurement_mode}")

    if all_indices.shape[0] == num_keep:
        transmitter_indices = all_indices.contiguous()
    else:
        subsample = all_indices.shape[0] // num_keep
        transmitter_indices = all_indices[::subsample].contiguous()

    receiver_indices = transmitter_indices.clone()

    num_src = transmitter_indices.shape[0]
    num_rec = receiver_indices.shape[0]
    receiver_mask = torch.ones((num_src, num_rec), dtype=torch.float32, device=device)

    return transmitter_indices, receiver_indices, receiver_mask


def load_eval_sample(cfg: BridgeConfig):
    """
    从来源代码风格的数据目录里读一个 eval 样本：
    dobs_500k/eval/train_6601.npy
    speed/eval/train_6601.npy
    """
    file_id = cfg.eval_index + 6601
    file_name = f"train_{file_id}.npy"

    dobs_path = os.path.join(cfg.base_dir_dobs_eval, file_name)
    speed_path = os.path.join(cfg.base_dir_speed_eval, file_name)

    if not os.path.exists(dobs_path):
        raise FileNotFoundError(f"Cannot find dobs file: {dobs_path}")
    if not os.path.exists(speed_path):
        raise FileNotFoundError(f"Cannot find speed file: {speed_path}")

    dobs_complex = np.load(dobs_path)
    speed_full = np.load(speed_path)

    return file_name, dobs_complex, speed_full


def make_inversion_input_from_dobs(dobs_complex: np.ndarray, target_size: int = 256):
    """
    当前 sparse 数据目录里读到的 dobs 已经是 [64,64] complex.
    为了对齐监督训练时的 InversionNet 输入分布，这里把 real/imag
    两个通道插值到 [2,256,256]。
    """
    inp = np.stack([dobs_complex.real, dobs_complex.imag], axis=0).astype(np.float32)  # [2,H,W]
    inp_t = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)  # [1,2,H,W]

    if inp_t.shape[-1] != target_size or inp_t.shape[-2] != target_size:
        inp_t = F.interpolate(inp_t, size=(target_size, target_size), mode="bilinear", align_corners=False)

    return inp_t


def build_observed_dobs_sparse(dobs_complex: np.ndarray, device: str):
    """
    当前目录中的 dobs 已经是 sparse/sparse_2 观测矩阵本身：
    sparse   -> [64,64]
    sparse_2 -> [32,32]
    因此这里不能再重复下采样。
    """
    dobs_obs = torch.tensor(dobs_complex, dtype=torch.complex64, device=device).unsqueeze(0)  # [1,M,L]
    return dobs_obs


# =========================================================
# 主流程
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inversion_ckpt", type=str, required=True)
    parser.add_argument("--base_dir_dobs_eval", type=str, required=True)
    parser.add_argument("--base_dir_speed_eval", type=str, required=True)
    parser.add_argument("--aux_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--eval_index", type=int, default=0)
    parser.add_argument("--measurement_mode", type=str, default="sparse", choices=["sparse", "sparse_2"])

    parser.add_argument("--num_iters", type=int, default=6)
    parser.add_argument("--step_size", type=float, default=50.0)
    parser.add_argument("--grad_smooth_kernel", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = BridgeConfig(
        inversion_ckpt=args.inversion_ckpt,
        base_dir_dobs_eval=args.base_dir_dobs_eval,
        base_dir_speed_eval=args.base_dir_speed_eval,
        aux_dir=args.aux_dir,
        output_dir=args.output_dir,
        eval_index=args.eval_index,
        measurement_mode=args.measurement_mode,
        num_iters=args.num_iters,
        step_size=args.step_size,
        grad_smooth_kernel=args.grad_smooth_kernel,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    device = cfg.device
    print("Using device:", device)

    # 1. 读取样本
    file_name, dobs_complex, speed_full = load_eval_sample(cfg)
    print("Loaded file:", file_name)
    print("dobs_complex shape:", dobs_complex.shape)
    print("speed_full shape:", speed_full.shape)

    gt_sos = torch.tensor(speed_full, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    save_speed_map(gt_sos, os.path.join(cfg.output_dir, "gt_speed_map.png"), title="Ground Truth")

    # 2. 构造 InversionNet 输入：64x64 -> 256x256
    inversion_input = make_inversion_input_from_dobs(dobs_complex, target_size=256).to(device)
    print("inversion_input shape:", inversion_input.shape)

    # 3. InversionNet 初值（256） -> CBS denormalize 到 480
    inversion_model = load_inversion_model(cfg.inversion_ckpt, device)
    with torch.no_grad():
        inv_pred_norm = inversion_model(inversion_input)      # [1,1,256,256]
        inv_sos_480 = cbs_denormalize(inv_pred_norm)          # [1,1,480,480]

    print("inv_pred_norm shape:", inv_pred_norm.shape)
    print("inv_sos_480 shape:", inv_sos_480.shape)
    save_speed_map(inv_sos_480, os.path.join(cfg.output_dir, "inversion_init_480.png"), title="Inversion Init")

    # 4. 构造 sparse 观测
    transmitter_indices, receiver_indices, receiver_mask = build_geometry(
        cfg.aux_dir, cfg.measurement_mode, device
    )
    dobs_obs = build_observed_dobs_sparse(dobs_complex, device)

    print("transmitter_indices shape:", transmitter_indices.shape)
    print("receiver_indices shape:", receiver_indices.shape)
    print("receiver_mask shape:", receiver_mask.shape)
    print("dobs_obs shape:", dobs_obs.shape)

    # 一致性检查
    num_src = transmitter_indices.shape[0]
    num_rec = receiver_indices.shape[0]
    if dobs_obs.shape[1] != num_src or dobs_obs.shape[2] != num_rec:
        raise ValueError(
            f"dobs_obs shape {tuple(dobs_obs.shape)} does not match "
            f"geometry ({num_src}, {num_rec})."
        )

    # 5. 初始化重建
    recon_sos = inv_sos_480.clone().detach()

    if cfg.use_roi_mask:
        roi_mask = torch.zeros_like(recon_sos)
        roi_mask[:, :, 90:390, 90:390] = 1.0
    else:
        roi_mask = torch.ones_like(recon_sos)

    history = {
        "rec_diff": [],
        "mse_to_gt": [],
        "psnr_to_gt": [],
        "ssim_to_gt": [],
    }

    data_range = 1595.1279 - 1408.692

    # 初值指标
    init_mse = F.mse_loss(recon_sos, gt_sos).item()
    init_psnr = calculate_psnr_torch(recon_sos, gt_sos, data_range=data_range)
    init_ssim = calculate_ssim_torch(recon_sos, gt_sos, data_range=data_range)
    print(f"[Init] mse={init_mse:.6f}, psnr={init_psnr:.4f}, ssim={init_ssim:.4f}")

    # 6. CBS + adjoint 少步细化
    for it in range(1, cfg.num_iters + 1):
        print(f"\n===== Iteration {it}/{cfg.num_iters} =====")

        model_cur = ConvergentBornSeries_Batch(
            f=cfg.frequency_hz,
            sos=recon_sos,
            boundary_width=list(cfg.boundary_width),
            boundary_strength=cfg.boundary_strength,
            boundary_type=cfg.boundary_type,
            device=device,
            src_loc_set=transmitter_indices.cpu().numpy(),
        )

        with torch.no_grad():
            u_cur = model_cur.forward()

        adj_model = ConvergentBornSeries_Batch_Adjoint(
            batch_model=model_cur,
            rec_loc=receiver_indices.to(device),
            dobs_500k_batch=dobs_obs,
            dobs_500k_mask=receiver_mask,
        )

        grad, rec_diff_value = adj_model.forward(u_cur)

        grad_smooth = F.avg_pool2d(
            grad,
            kernel_size=cfg.grad_smooth_kernel,
            stride=1,
            padding=cfg.grad_smooth_kernel // 2,
        )

        recon_sos = recon_sos - cfg.step_size * grad_smooth * roi_mask
        recon_sos = torch.clamp(recon_sos, min=cfg.clip_min, max=cfg.clip_max)

        mse_val = F.mse_loss(recon_sos, gt_sos).item()
        psnr_val = calculate_psnr_torch(recon_sos, gt_sos, data_range=data_range)
        ssim_val = calculate_ssim_torch(recon_sos, gt_sos, data_range=data_range)

        history["rec_diff"].append(float(rec_diff_value.item()))
        history["mse_to_gt"].append(float(mse_val))
        history["psnr_to_gt"].append(float(psnr_val))
        history["ssim_to_gt"].append(float(ssim_val))

        print(
            f"rec_diff={rec_diff_value.item():.6f} | "
            f"mse={mse_val:.6f} | "
            f"psnr={psnr_val:.4f} | "
            f"ssim={ssim_val:.4f}"
        )

        if cfg.save_each_iter:
            save_speed_map(
                recon_sos,
                os.path.join(cfg.output_dir, f"recon_iter_{it:03d}.png"),
                title=f"Refined Iter {it}",
            )

        del model_cur, adj_model, u_cur, grad, grad_smooth
        torch.cuda.empty_cache()

    # 7. 保存最终对比
    save_compare_figure(
        gt_480=gt_sos,
        inv_480=inv_sos_480,
        refined_480=recon_sos,
        save_path=os.path.join(cfg.output_dir, "compare_final.png"),
    )

    save_curve(history, os.path.join(cfg.output_dir, "bridge_curve.png"))

    final_metrics = {
        "init": {
            "mse": init_mse,
            "psnr": init_psnr,
            "ssim": init_ssim,
        },
        "refined": {
            "mse": F.mse_loss(recon_sos, gt_sos).item(),
            "psnr": calculate_psnr_torch(recon_sos, gt_sos, data_range=data_range),
            "ssim": calculate_ssim_torch(recon_sos, gt_sos, data_range=data_range),
        },
    }

    with open(os.path.join(cfg.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    with open(os.path.join(cfg.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    print("\nFinished.")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()