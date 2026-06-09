import os
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_openbreastus_oldstyle import OpenBreastUSOldStyleDataset


@dataclass
class Config:
    data_root: str = r"D:\OpenBreastUS_processed_oldstyle_2x256x256_fullresize"
    save_dir: str = "./results_openbreastus_oldstyle_unet_v1"
    batch_size: int = 2
    num_epochs: int = 20
    lr: float = 1e-3
    num_workers: int = 0
    normalize_input: bool = True
    normalize_target: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print_freq: int = 20
    save_freq: int = 5


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.up1 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up3 = Up(64, 32, 32)

        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse < eps:
        return 99.0
    max_val = max(target.max().item(), pred.max().item(), 1.0)
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, cfg):
    model.train()
    running_loss = 0.0

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % cfg.print_freq == 0 or step == len(loader):
            print(f"Epoch {epoch:03d} | Step {step:04d}/{len(loader):04d} | Loss = {loss.item():.6f}")

    return running_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        total_loss += loss.item()
        total_psnr += compute_psnr(pred, y)

    return total_loss / len(loader), total_psnr / len(loader)


def save_checkpoint(model, optimizer, epoch, save_path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )


def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"Using device: {cfg.device}")

    train_set = OpenBreastUSOldStyleDataset(
        root_dir=cfg.data_root,
        split="train",
        normalize_input=cfg.normalize_input,
        normalize_target=cfg.normalize_target,
    )
    test_set = OpenBreastUSOldStyleDataset(
        root_dir=cfg.data_root,
        split="test",
        normalize_input=cfg.normalize_input,
        normalize_target=cfg.normalize_target,
    )

    print(f"Train samples: {len(train_set)}")
    print(f"Test samples : {len(test_set)}")

    x0, y0 = train_set[0]
    print(f"Input shape  : {x0.shape}")
    print(f"Target shape : {y0.shape}")
    print(f"Input min/max: {x0.min().item():.4f}, {x0.max().item():.4f}")
    print(f"Target min/max: {y0.min().item():.4f}, {y0.max().item():.4f}")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SmallUNet(in_channels=2, out_channels=1).to(cfg.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_loss = float("inf")

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device, epoch, cfg)
        val_loss, val_psnr = evaluate(model, test_loader, criterion, cfg.device)

        print("=" * 72)
        print(f"Epoch {epoch:03d} | Train Loss = {train_loss:.6f} | Val Loss = {val_loss:.6f} | Val PSNR = {val_psnr:.3f}")
        print("=" * 72)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                os.path.join(cfg.save_dir, "checkpoint_best.pth"),
            )
            print("Saved best checkpoint.")

        if epoch % cfg.save_freq == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                os.path.join(cfg.save_dir, f"checkpoint_epoch_{epoch:03d}.pth"),
            )

    print("Training finished.")


if __name__ == "__main__":
    main()