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


class FirstNoiseWrapper(torch.utils.data.Dataset):
    """
    原始 USCT_Dataset_CBS 返回:
        input_batch: [3, 2, 64, 64]
        output_batch: [1, 256, 256]

    这里我们先只取第一种噪声版本，返回:
        x: [2, 64, 64]
        y: [1, 256, 256]
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        x = x[0]   # [2, 64, 64]
        return x.float(), y.float()


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
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            bs = x.size(0)
            for i in range(bs):
                if batch_idx * bs + i >= max_items:
                    return

                idx = batch_idx * bs + i
                save_img(x[i, 0], os.path.join(save_dir, f"epoch_{epoch:03d}_sample_{idx}_input_real.png"),
                         title=f"Epoch {epoch} Input Real", cmap="gray")
                save_img(x[i, 1], os.path.join(save_dir, f"epoch_{epoch:03d}_sample_{idx}_input_imag.png"),
                         title=f"Epoch {epoch} Input Imag", cmap="gray")
                save_img(y[i, 0], os.path.join(save_dir, f"epoch_{epoch:03d}_sample_{idx}_target.png"),
                         title=f"Epoch {epoch} Target")
                save_img(pred[i, 0], os.path.join(save_dir, f"epoch_{epoch:03d}_sample_{idx}_pred.png"),
                         title=f"Epoch {epoch} Prediction")
                save_img((y[i, 0] - pred[i, 0]),
                         os.path.join(save_dir, f"epoch_{epoch:03d}_sample_{idx}_diff.png"),
                         title=f"Epoch {epoch} Target - Pred")


def main():
    # =========================
    # 1. 基本设置
    # =========================
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    save_root = "results_train_inversionnet_small"
    ckpt_dir = os.path.join(save_root, "checkpoints")
    vis_dir = os.path.join(save_root, "visuals")
    ensure_dir(save_root)
    ensure_dir(ckpt_dir)
    ensure_dir(vis_dir)

    # 你先做小规模过拟合测试
    batch_size = 2
    num_epochs = 80
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
    dataset = FirstNoiseWrapper(base_dataset)

    print("Total dataset size:", len(dataset))

    # 小样本划分：8 train, 2 val
    all_indices = list(range(len(dataset)))
    train_indices = all_indices[:8]
    val_indices = all_indices[8:10]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # =========================
    # 3. 模型、损失、优化器
    # =========================
    print("\n[2] Building model...")
    model = InversionNet().to(device)
    criterion = nn.MSELoss()
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
                os.path.join(ckpt_dir, "best_inversionnet_small.pt"),
            )

        # 定期保存预测图
        if epoch in [1, 5, 10, 20, 40, 80]:
            save_predictions(model, val_loader, device, vis_dir, epoch, max_items=2)

    # =========================
    # 5. 保存 loss 曲线
    # =========================
    print("\n[4] Saving curves...")
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("InversionNet Small Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, "loss_curve.png"), dpi=200)
    plt.close()

    print("\nDone.")
    print("Best val loss:", best_val_loss)
    print("Results saved to:", save_root)


if __name__ == "__main__":
    main()