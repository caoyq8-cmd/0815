import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.io import loadmat

from utils_cbs import denormalize
from cbs_model import ConvergentBornSeries_Batch, ConvergentBornSeries_Batch_Adjoint
from image_datasets import USCT_Dataset_CBS


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_receiver_data(u_batch, receiver_indices, receiver_mask=None):
    """
    u_batch: [N, M, H, W] complex
    receiver_indices: [L, 2]
    receiver_mask: [M, L] or None
    return: [N, M, L]
    """
    rec = u_batch[:, :, receiver_indices[:, 0], receiver_indices[:, 1]]
    if receiver_mask is not None:
        rec = rec * receiver_mask.to(u_batch.device)
    return rec


def save_speed_map(tensor_480, save_path, vmin=1408.692, vmax=1595.1279, title=None):
    """
    tensor_480: [1,1,480,480] or [480,480]
    """
    if tensor_480.ndim == 4:
        img = tensor_480[0, 0]
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


def main():
    # =========================
    # 1. 基本配置
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    save_dir = "results_adjoint_descent"
    ensure_dir(save_dir)

    num_iters = 10
    step_size = 1e3
    clip_min = 1400.0
    clip_max = 1605.0

    # 图像域 mask
    base_mask = np.load("auxiliary_data/mask.npy")

    # 测量域 mask：测试数据是 64x64，先用全1
    measurement_mask = np.ones((64, 64), dtype=np.float32)

    data_dict = {
        "max_output": 1595.1279,
        "min_output": 1408.692,
        "resize_size": (256, 256),
        "measurement_mode": "sparse",
        "stage": "eval",
        "frequency": "500k",

        "base_mask": base_mask,
        "mask": np.stack([measurement_mask, measurement_mask], axis=0),

        "x_pos": loadmat("auxiliary_data/x_pos.mat")["x_pos256"],
        "y_pos": loadmat("auxiliary_data/y_pos.mat")["y_pos256"],

        "base_dir_dobs_500k_train": "Datasets_test/AI4Scup2_simulated_CBS_sparse/dobs_500k/train",
        "base_dir_dobs_500k_eval": "Datasets_test/AI4Scup2_simulated_CBS_sparse/dobs_500k/eval",
        "base_dir_speed_train": "Datasets_test/AI4Scup2_simulated_CBS_sparse/speed/train",
        "base_dir_speed_eval": "Datasets_test/AI4Scup2_simulated_CBS_sparse/speed/eval",
    }

    # =========================
    # 2. 读取一条样本
    # =========================
    print("\n[1] Loading dataset...")
    eval_dataset = USCT_Dataset_CBS(data_dict=data_dict)
    print("Dataset length:", len(eval_dataset))

    step = 0
    input_batch, output_batch = eval_dataset[step]

    print("input_batch shape:", input_batch.shape)
    print("output_batch shape:", output_batch.shape)

    # 真值声速图 [1,1,480,480]
    gt_sos = denormalize(output_batch.unsqueeze(0)).to(device)
    print("gt_sos shape:", gt_sos.shape)

    save_speed_map(gt_sos, os.path.join(save_dir, "gt_speed_map.png"), title="Ground Truth")

    # =========================
    # 3. 构造稀疏几何（64个换能器）
    # =========================
    print("\n[2] Building geometry...")
    all_indices = torch.cat(
        (
            torch.tensor(data_dict["x_pos"].astype(np.int64)),
            torch.tensor(data_dict["y_pos"].astype(np.int64)),
        ),
        dim=1
    )

    subsample = 4   # 256 -> 64
    transmitter_indices = all_indices[::subsample].contiguous()
    receiver_indices = transmitter_indices.clone()

    num_src = transmitter_indices.shape[0]
    num_rec = receiver_indices.shape[0]
    receiver_mask = torch.ones((num_src, num_rec), dtype=torch.float32, device=device)

    print("transmitter_indices shape:", transmitter_indices.shape)
    print("receiver_indices shape:", receiver_indices.shape)
    print("receiver_mask shape:", receiver_mask.shape)

    # =========================
    # 4. 用真值图生成观测 dobs_gt
    # =========================
    print("\n[3] Generating ground-truth measurements...")
    model_gt = ConvergentBornSeries_Batch(
        f=500e3,
        sos=gt_sos,
        boundary_width=[300, 300],
        boundary_strength=225,
        boundary_type="PML3",
        device=device,
        src_loc_set=transmitter_indices.cpu().numpy()
    )

    with torch.no_grad():
        u_gt = model_gt.forward()

    dobs_gt = extract_receiver_data(u_gt, receiver_indices, receiver_mask)
    print("dobs_gt shape:", dobs_gt.shape)

    plt.figure(figsize=(6, 5))
    plt.imshow(torch.real(dobs_gt[0]).detach().cpu(), cmap="gray")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gt_measurement_real.png"), dpi=200, bbox_inches="tight")
    plt.close()

    del u_gt
    torch.cuda.empty_cache()

    # =========================
    # 5. 初始化重建：均匀背景
    # =========================
    print("\n[4] Initializing reconstruction...")
    recon_sos = 1500.0 * torch.ones_like(gt_sos).to(device)
    save_speed_map(recon_sos, os.path.join(save_dir, "recon_iter_000.png"), title="Iter 0")

    # 只更新中心 ROI，避免边界/PML 区域引入强伪影
    roi_mask = torch.zeros_like(recon_sos)
    roi_mask[:, :, 90:390, 90:390] = 1.0

    # 记录曲线
    rec_diff_list = []
    mse_list = []

    # =========================
    # 6. 迭代重建
    # =========================
    print("\n[5] Starting adjoint descent...")
    for it in range(1, num_iters + 1):
        print(f"\n===== Iteration {it}/{num_iters} =====")

        # ---- 6.1 当前模型 forward ----
        model_cur = ConvergentBornSeries_Batch(
            f=500e3,
            sos=recon_sos,
            boundary_width=[300, 300],
            boundary_strength=225,
            boundary_type="PML3",
            device=device,
            src_loc_set=transmitter_indices.cpu().numpy()
        )

        with torch.no_grad():
            u_cur = model_cur.forward()

        dobs_cur = extract_receiver_data(u_cur, receiver_indices, receiver_mask)

        # ---- 6.2 Adjoint ----
        adj_model = ConvergentBornSeries_Batch_Adjoint(
            batch_model=model_cur,
            rec_loc=receiver_indices.to(device),
            dobs_500k_batch=dobs_gt,
            dobs_500k_mask=receiver_mask
        )

        grad, rec_diff_value = adj_model.forward(u_cur)

        print("grad shape:", grad.shape)
        print("rec_diff_value:", rec_diff_value.item())

        # ---- 6.3 梯度平滑 + ROI 更新 ----
        grad_smooth = F.avg_pool2d(grad, kernel_size=9, stride=1, padding=4)
        recon_sos = recon_sos - step_size * grad_smooth * roi_mask

        # 限制在合理物理范围内
        recon_sos = torch.clamp(recon_sos, min=clip_min, max=clip_max)

        # ---- 6.4 记录指标 ----
        mse_val = torch.mean((recon_sos - gt_sos) ** 2).item()
        rec_diff_list.append(rec_diff_value.item())
        mse_list.append(mse_val)

        print("MSE to GT:", mse_val)

        # ---- 6.5 保存中间结果 ----
        save_speed_map(
            recon_sos,
            os.path.join(save_dir, f"recon_iter_{it:03d}.png"),
            title=f"Iter {it}"
        )

        # 清理显存
        del u_cur, dobs_cur, grad, grad_smooth, model_cur, adj_model
        torch.cuda.empty_cache()

    # =========================
    # 7. 保存最终结果与曲线
    # =========================
    print("\n[6] Saving curves...")

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, num_iters + 1), rec_diff_list, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("rec_diff_value")
    plt.title("Measurement Mismatch")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curve_rec_diff.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, num_iters + 1), mse_list, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("MSE to GT")
    plt.title("Reconstruction Error")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curve_mse.png"), dpi=200)
    plt.close()

    print("\nDone.")
    print("Results saved to:", save_dir)
    print("rec_diff_list =", rec_diff_list)
    print("mse_list =", mse_list)


if __name__ == "__main__":
    main()