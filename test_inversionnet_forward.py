import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.io import loadmat

from image_datasets import USCT_Dataset_CBS

# 这里按常见写法导入
from InversionNet_modules.Baselines_modified import InversionNet


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def save_img(img_tensor, save_path, title=None, cmap="inferno", vmin=None, vmax=None):
    """
    img_tensor: [H,W] or [1,H,W]
    """
    if img_tensor.ndim == 3:
        img_tensor = img_tensor[0]

    plt.figure(figsize=(5, 5))
    plt.imshow(img_tensor.detach().cpu(), cmap=cmap, vmin=vmin, vmax=vmax)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    save_dir = "results_test_inversionnet_forward"
    ensure_dir(save_dir)

    # =========================
    # 1. 数据配置
    # =========================
    base_mask = np.load("auxiliary_data/mask.npy")
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

    # input_batch: [3, 2, 64, 64]
    # 这里先取第一种噪声版本
    x = input_batch[0].unsqueeze(0).to(device)   # [1, 2, 64, 64]
    y = output_batch.unsqueeze(0).to(device)     # [1, 1, 256, 256]

    print("x shape for network:", x.shape)
    print("y shape:", y.shape)

    # 保存输入的实部/虚部
    save_img(x[0, 0], os.path.join(save_dir, "input_real.png"), title="Input Real", cmap="gray")
    save_img(x[0, 1], os.path.join(save_dir, "input_imag.png"), title="Input Imag", cmap="gray")
    save_img(y[0, 0], os.path.join(save_dir, "target_speed_map_norm.png"), title="Target (normalized)")

    # =========================
    # 3. 构建网络
    # =========================
    print("\n[2] Building InversionNet...")
    model = InversionNet().to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print("Total params:", total_params)

    # =========================
    # 4. 前向测试
    # =========================
    print("\n[3] Running forward...")
    with torch.no_grad():
        pred = model(x)

    print("pred shape:", pred.shape)
    print("pred min/max:", pred.min().item(), pred.max().item())

    save_img(pred[0, 0], os.path.join(save_dir, "pred_speed_map_norm.png"), title="Prediction (normalized)")
    save_img(y[0, 0] - pred[0, 0], os.path.join(save_dir, "diff_target_minus_pred.png"), title="Target - Pred")

    # =========================
    # 5. 对齐尺寸检查
    # =========================
    if pred.shape != y.shape:
        print("\n[Warning] pred.shape != y.shape")
        print("pred.shape =", pred.shape)
        print("y.shape    =", y.shape)

        # 临时插值到 target 尺寸
        pred_resized = F.interpolate(pred, size=y.shape[-2:], mode="bilinear", align_corners=False)
        print("pred_resized shape:", pred_resized.shape)

        save_img(pred_resized[0, 0], os.path.join(save_dir, "pred_resized_norm.png"), title="Prediction Resized")
    else:
        pred_resized = pred

    mse = torch.mean((pred_resized - y) ** 2).item()
    print("Forward MSE (untrained model):", mse)

    print("\nDone.")
    print("Results saved to:", save_dir)


if __name__ == "__main__":
    main()