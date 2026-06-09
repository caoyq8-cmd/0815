import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from scipy.io import loadmat

from image_datasets import USCT_Dataset_CBS
from InversionNet_modules.Baselines_modified import InversionNet


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class ExpandedNoiseWrapper(torch.utils.data.Dataset):
    """
    原始 USCT_Dataset_CBS 返回:
        input_batch: [3, 2, 64, 64]
        output_batch: [1, 256, 256]

    这里把每个原始样本展开成 3 个样本：
        idx=0 -> 原样本0的第0种噪声
        idx=1 -> 原样本0的第1种噪声
        idx=2 -> 原样本0的第2种噪声
        idx=3 -> 原样本1的第0种噪声
        ...

    最终返回:
        x: [2, 64, 64]
        y: [1, 256, 256]
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.num_noises = 3

    def __len__(self):
        return len(self.base_dataset) * self.num_noises

    def __getitem__(self, idx):
        base_idx = idx // self.num_noises
        noise_idx = idx % self.num_noises

        x_all, y = self.base_dataset[base_idx]   # x_all: [3,2,64,64]
        x = x_all[noise_idx]                     # [2,64,64]
        return x.float(), y.float()


class WeightedMSELoss(nn.Module):
    """
    加权 MSE：
        loss = mean( weight * (pred - target)^2 )

    这里中心 ROI 权重大，外围权重小。
    """
    def __init__(self, weight_map: torch.Tensor):
        super().__init__()
        # weight_map shape: [1,1,H,W]
        self.register_buffer("weight_map", weight_map)

    def forward(self, pred, target):
        loss = self.weight_map * (pred - target) ** 2
        return loss.mean()


def build_weight_map(h=256, w=256, device="cpu", inner_weight=5.0, outer_weight=1.0):
    """
    构造 256x256 的加权图。
    中间区域加大权重，外围区域较小权重。
    """
    weight = torch.ones((1, 1, h, w), dtype=torch.float32, device=device) * outer_weight

    # 这里给中心区域更高权重
    # 你可以按需要再调
    weight[:, :, 32:224, 32:224] = 2.0
    weight[:, :, 48:208, 48:208] = 3.0
    weight[:, :, 64:192, 64:192] = inner_weight

    return weight


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)   # [B,2,64,64]
            y = y.to(device)   # [B,1,256,256]

            pred = model(x)
            loss = criterion(pred, y)

            total_loss += loss.item() * x.size(0)
            count += x.size(0)

    return total_loss / max(count, 1)


def save_predictions(model, loader, device, save_dir, epoch, max_items=3):
    model.eval()
    ensure_dir(save_dir)

    with torch.no_grad():
        saved = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)

            bs = x.size(0)
            for i in range(bs):
                if saved >= max_items:
                    return

                save_img(
                    x[i, 0],
                    os.path.join(save_dir, f"epoch_{epoch:03d}_sample_{saved}_input_real.png"),
                    title=f"Epoch {epoch} Input Real",
                    cmap="gray"
                )
                save_img(
                    x[i, 1],
                    os.path.join(save_dir, f"epoch_{epoch:03d}_sample_{saved}_input_imag.png"),
                    title=f"Epoch {epoch} Input Imag",
                    cmap="gray"
                )
                save_img(
                    y[i, 0],
                    os.path.join(save_dir, f"epoch_{epoch:03d}_sample_{saved}_target.png"),
                    title=f"Epoch {epoch} Target"
                )
                save_img(
                    pred[i, 0],
                    os.path.join(save_dir, f"epoch_{epoch:03d}_sample_{saved}_pred.png"),
                    title=f"Epoch {epoch} Prediction"
                )
                save_img(
                    y[i, 0] - pred[i, 0],
                    os.path.join(save_dir, f"epoch_{epoch:03d}_sample_{saved}_diff.png"),
                    title=f"Epoch {epoch} Target - Pred"
                )
                saved += 1


def main():
    # =========================
    # 1. 基本设置
    # =========================
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    save_root = "results_train_inversionnet_weighted"
    ckpt_dir = os.path.join(save_root, "checkpoints")
    vis_dir = os.path.join(save_root, "visuals")
    ensure_dir(save_root)
    ensure_dir(ckpt_dir)
    ensure_dir(vis_dir)

    # 训练参数
    batch_size = 2
    num_epochs = 100
    lr = 1e-4
    weight_decay = 1e-6

    # =========================
    # 2. 数据配置
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

    print("\n[1] Loading dataset...")
    base_dataset = USCT_Dataset_CBS(data_dict=data_dict)
    dataset = ExpandedNoiseWrapper(base_dataset)

    print("Base dataset size:", len(base_dataset))
    print("Expanded dataset size:", len(dataset))

    # 原始样本 10 个，展开后 30 个
    # 前 8 个原样本做 train -> 24 个展开样本
    # 后 2 个原样本做 val   -> 6 个展开样本
    train_indices = []
    val_indices = []

    for base_idx in range(10):
        expanded_ids = [base_idx * 3 + k for k in range(3)]
        if base_idx < 8:
            train_indices.extend(expanded_ids)
        else:
            val_indices.extend(expanded_ids)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print("Train size:", len(train_set))
    print("Val size:", len(val_set))

    # =========================
    # 3. 模型、损失、优化器
    # =========================
    print("\n[2] Building model...")
    model = InversionNet().to(device)

    weight_map = build_weight_map(
        h=256,
        w=256,
        device=device,
        inner_weight=5.0,
        outer_weight=1.0
    )
    criterion = WeightedMSELoss(weight_map)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total params:", total_params)

    # =========================
    # 4. 训练
    # =========================
    print("\n[3] Start training...")
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0

        for x, y in train_loader:
            x = x.to(device)   # [B,2,64,64]
            y = y.to(device)   # [B,1,256,256]

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            count += x.size(0)

        train_loss = running_loss / max(count, 1)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d}/{num_epochs} | train_loss = {train_loss:.6f} | val_loss = {val_loss:.6f}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                os.path.join(ckpt_dir, "best_inversionnet_weighted.pt"),
            )

        # 定期保存预测图
        if epoch in [1, 5, 10, 20, 40, 80, 100]:
            save_predictions(model, val_loader, device, vis_dir, epoch, max_items=2)

    # =========================
    # 5. 保存 loss 曲线
    # =========================
    print("\n[4] Saving curves...")
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted MSE Loss")
    plt.title("InversionNet Weighted Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, "loss_curve.png"), dpi=200)
    plt.close()

    print("\nDone.")
    print("Best val loss:", best_val_loss)
    print("Results saved to:", save_root)


if __name__ == "__main__":
    main()