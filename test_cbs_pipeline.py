import os
import numpy as np
import torch
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


def main():
    # =========================
    # 1. 基本配置
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    save_dir = "results_test_cbs_pipeline"
    ensure_dir(save_dir)

    # 图像域 mask
    base_mask = np.load("auxiliary_data/mask.npy")

    # 测量域 mask：当前测试数据是 64x64，所以先用全1
    measurement_mask = np.ones((64, 64), dtype=np.float32)

    data_dict = {
        "max_output": 1595.1279,
        "min_output": 1408.692,
        "resize_size": (256, 256),
        "measurement_mode": "sparse",
        "stage": "eval",
        "frequency": "500k",

        # 图像域
        "base_mask": base_mask,

        # 测量域
        "mask": np.stack([measurement_mask, measurement_mask], axis=0),

        "x_pos": loadmat("auxiliary_data/x_pos.mat")["x_pos256"],
        "y_pos": loadmat("auxiliary_data/y_pos.mat")["y_pos256"],

        "base_dir_dobs_500k_train": "Datasets_test/AI4Scup2_simulated_CBS_sparse/dobs_500k/train",
        "base_dir_dobs_500k_eval": "Datasets_test/AI4Scup2_simulated_CBS_sparse/dobs_500k/eval",
        "base_dir_speed_train": "Datasets_test/AI4Scup2_simulated_CBS_sparse/speed/train",
        "base_dir_speed_eval": "Datasets_test/AI4Scup2_simulated_CBS_sparse/speed/eval",
    }

    # =========================
    # 2. 读取数据
    # =========================
    print("\n[1] Loading dataset...")
    eval_dataset = USCT_Dataset_CBS(data_dict=data_dict)

    sample_file = os.path.join(
        data_dict["base_dir_dobs_500k_eval"],
        sorted(os.listdir(data_dict["base_dir_dobs_500k_eval"]))[0]
    )
    raw_dobs = np.load(sample_file)
    print("raw dobs shape:", raw_dobs.shape, "dtype:", raw_dobs.dtype)
    print("mask shape:", data_dict["mask"].shape)
    print("Dataset length:", len(eval_dataset))

    step = 0
    input_batch, output_batch = eval_dataset[step]

    print("input_batch shape:", input_batch.shape)
    print("output_batch shape:", output_batch.shape)
    print("input_batch dtype:", input_batch.dtype)
    print("output_batch min/max:", output_batch.min().item(), output_batch.max().item())

    # output_batch: (1, 256, 256) normalized
    # denormalize后恢复到物理声速图，并放大回480x480
    output_cbs_batch = denormalize(output_batch.unsqueeze(0)).to(device)
    cbs_batch_one = 1500.0 * torch.ones_like(output_cbs_batch).to(device)

    print("output_cbs_batch shape:", output_cbs_batch.shape)
    print("cbs_batch_one shape:", cbs_batch_one.shape)

    # 保存真值图
    plt.figure(figsize=(5, 5))
    plt.imshow(
        output_cbs_batch[0, 0, 90:390, 90:390].detach().cpu(),
        cmap="inferno",
        vmin=1408.692,
        vmax=1595.1279
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gt_speed_map.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", os.path.join(save_dir, "gt_speed_map.png"))

    # =========================
    # 3. 构造几何（只取64个换能器，减小显存）
    # =========================
    print("\n[2] Building transducer geometry...")
    all_indices = torch.cat(
        (
            torch.tensor(data_dict["x_pos"].astype(np.int64)),
            torch.tensor(data_dict["y_pos"].astype(np.int64)),
        ),
        dim=1
    )

    # 256 -> 64，均匀抽样
    subsample = 4
    transmitter_indices = all_indices[::subsample].contiguous()
    receiver_indices = transmitter_indices.clone()

    print("transmitter_indices shape:", transmitter_indices.shape)
    print("receiver_indices shape:", receiver_indices.shape)

    num_src = transmitter_indices.shape[0]
    num_rec = receiver_indices.shape[0]
    downsampled_input_mask = torch.ones((num_src, num_rec), dtype=torch.float32, device=device)
    print("downsampled_input_mask shape:", downsampled_input_mask.shape)

    # =========================
    # 4. CBS forward: 真值声速图
    # =========================
    print("\n[3] Running CBS forward on ground-truth speed map...")
    model_gt = ConvergentBornSeries_Batch(
        f=500e3,
        sos=output_cbs_batch,
        boundary_width=[300, 300],
        boundary_strength=225,
        boundary_type="PML3",
        device=device,
        src_loc_set=transmitter_indices.cpu().numpy()
    )

    with torch.no_grad():
        u_gt = model_gt.forward()

    print("u_gt shape:", u_gt.shape)
    print("u_gt dtype:", u_gt.dtype)

    dobs_gt = extract_receiver_data(u_gt, receiver_indices, downsampled_input_mask)
    print("dobs_gt shape:", dobs_gt.shape)
    print("dobs_gt dtype:", dobs_gt.dtype)

    plt.figure(figsize=(6, 5))
    plt.imshow(torch.real(dobs_gt[0]).detach().cpu(), cmap="gray")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "forward_real_part.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", os.path.join(save_dir, "forward_real_part.png"))

    # 释放真值全波场，节省显存
    del u_gt
    torch.cuda.empty_cache()

    # =========================
    # 5. CBS forward: 均匀背景模型
    # =========================
    print("\n[4] Running CBS forward on homogeneous background...")
    model_bg = ConvergentBornSeries_Batch(
        f=500e3,
        sos=cbs_batch_one,
        boundary_width=[300, 300],
        boundary_strength=225,
        boundary_type="PML3",
        device=device,
        src_loc_set=transmitter_indices.cpu().numpy()
    )

    with torch.no_grad():
        u_bg = model_bg.forward()

    print("u_bg shape:", u_bg.shape)

    dobs_bg = extract_receiver_data(u_bg, receiver_indices, downsampled_input_mask)
    print("dobs_bg shape:", dobs_bg.shape)

    # =========================
    # 6. Adjoint
    # =========================
    print("\n[5] Running adjoint...")
    adj_model = ConvergentBornSeries_Batch_Adjoint(
        batch_model=model_bg,
        rec_loc=receiver_indices.to(device),
        dobs_500k_batch=dobs_gt,
        dobs_500k_mask=downsampled_input_mask
    )

    grad, rec_diff_value = adj_model.forward(u_bg)

    print("grad shape:", grad.shape)
    print("grad dtype:", grad.dtype)
    print("rec_diff_value:", rec_diff_value)

    grad_to_show = grad[0, 0, 90:390, 90:390]
    if torch.is_complex(grad_to_show):
        grad_to_show = torch.real(grad_to_show)

    plt.figure(figsize=(5, 5))
    plt.imshow(grad_to_show.detach().cpu(), cmap="viridis")
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "adjoint_gradient.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", os.path.join(save_dir, "adjoint_gradient.png"))

    print("\nAll done.")
    print("Check results in:", save_dir)


if __name__ == "__main__":
    main()