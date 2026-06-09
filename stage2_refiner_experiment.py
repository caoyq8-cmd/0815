import os
import math
import glob
import argparse
from typing import Dict, Tuple, Optional

import numpy as np
import scipy.io as sio
import h5py
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from InversionNet_modules.Baselines import InversionNet as RealInversionNet


# ============================================================
# Utilities
# ============================================================

def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_gray_image(arr: np.ndarray, path: str, vmin: Optional[float] = None, vmax: Optional[float] = None):
    arr = arr.astype(np.float32)
    if vmin is None:
        vmin = float(arr.min())
    if vmax is None:
        vmax = float(arr.max())
    arr = np.clip((arr - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _read_mat_value_h5(obj):
    arr = np.array(obj)

    if getattr(arr.dtype, "names", None) is not None and "real" in arr.dtype.names and "imag" in arr.dtype.names:
        arr = arr["real"].astype(np.float32) + 1j * arr["imag"].astype(np.float32)

    if arr.ndim == 3:
        if arr.shape[0] not in (1, 2, 3, 4) and arr.shape[-1] in (1, 2, 3, 4):
            arr = np.transpose(arr, (2, 0, 1))

    if np.iscomplexobj(arr):
        return arr
    return arr.astype(np.float32)


def load_mat_auto(path: str, keys: Optional[list] = None):
    try:
        data = sio.loadmat(path)
        if keys is None:
            return data
        return {k: data[k] for k in keys if k in data}
    except NotImplementedError:
        out = {}
        with h5py.File(path, "r") as f:
            use_keys = keys if keys is not None else list(f.keys())
            for k in use_keys:
                if k in f:
                    out[k] = _read_mat_value_h5(f[k])
        return out


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def grad_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(sobel_edges(pred), sobel_edges(target))


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    mse = F.mse_loss(pred, target).item()
    rmse = math.sqrt(mse)

    target_min = float(target.min().item())
    target_max = float(target.max().item())
    data_range = max(target_max - target_min, 1e-8)
    psnr = 20.0 * math.log10(data_range) - 10.0 * math.log10(max(mse, 1e-12))

    mu_x = pred.mean().item()
    mu_y = target.mean().item()
    var_x = pred.var(unbiased=False).item()
    var_y = target.var(unbiased=False).item()
    cov_xy = ((pred - pred.mean()) * (target - target.mean())).mean().item()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2) + 1e-12)

    return {
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": float(ssim),
    }


# ============================================================
# Dataset
# ============================================================

class OpenBreastUSOldStyleDataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        self.root = self._resolve_split_dir(root, split)
        self.files = sorted(glob.glob(os.path.join(self.root, "*.mat")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .mat files found under {self.root}")

    @staticmethod
    def _resolve_split_dir(root: str, split: str) -> str:
        direct = os.path.join(root, split)
        if os.path.isdir(direct):
            return direct

        mats_here = glob.glob(os.path.join(root, "*.mat"))
        if os.path.isdir(root) and len(mats_here) > 0:
            return root

        if os.path.isdir(root):
            candidates = []
            for name in os.listdir(root):
                sub = os.path.join(root, name)
                if not os.path.isdir(sub):
                    continue
                sub_split = os.path.join(sub, split)
                if os.path.isdir(sub_split):
                    candidates.append(sub_split)
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                raise FileNotFoundError(
                    f"Multiple candidate split directories found for split='{split}': {candidates}. "
                    f"Please pass the exact dataset root."
                )

        raise FileNotFoundError(
            f"Could not resolve split directory for split='{split}' from root='{root}'. "
            f"Expected one of: {os.path.join(root, split)} or {os.path.join(root, '<dataset_name>', split)}"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        data = load_mat_auto(path, keys=["input_2ch", "target_256"])

        if "input_2ch" not in data:
            raise KeyError(f"input_2ch not found in {path}")
        if "target_256" not in data:
            raise KeyError(f"target_256 not found in {path}")

        x = data["input_2ch"].astype(np.float32)
        y = data["target_256"].astype(np.float32)

        if y.ndim == 2:
            y = y[None, ...]
        elif y.ndim == 3 and y.shape[0] != 1:
            if y.shape[-1] == 1:
                y = np.transpose(y, (2, 0, 1))
            else:
                raise ValueError(f"Unexpected target_256 shape {y.shape} in {path}")

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "path": path,
        }


# ============================================================
# Models
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
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

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SmallUNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base_ch: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        self.d1 = Down(base_ch, base_ch * 2)
        self.d2 = Down(base_ch * 2, base_ch * 4)
        self.d3 = Down(base_ch * 4, base_ch * 8)
        self.u1 = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.u2 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.u3 = Up(base_ch * 2, base_ch, base_ch)
        self.outc = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x = self.u1(x4, x3)
        x = self.u2(x, x2)
        x = self.u3(x, x1)
        return self.outc(x)


class ResidualRefiner(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 32, residual_scale: float = 0.1):
        super().__init__()
        self.refiner = SmallUNet(in_ch=in_ch, out_ch=1, base_ch=base_ch)
        self.residual_scale = residual_scale

    def forward(self, init_pred: torch.Tensor, aux: Optional[torch.Tensor] = None):
        if aux is None:
            inp = init_pred
        else:
            inp = torch.cat([init_pred, aux], dim=1)
        residual = self.refiner(inp)
        out = init_pred + self.residual_scale * residual
        return out, residual


def build_real_initializer(base_ch: int):
    return RealInversionNet(
        dim1=base_ch,
        dim2=base_ch * 2,
        dim3=base_ch * 4,
        dim4=base_ch * 8,
        dim5=base_ch * 16,
    )


# ============================================================
# Load / save checkpoints
# ============================================================

def smart_load_state_dict(model: nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    model_state = model.state_dict()
    cleaned = {}
    skipped = []

    for k, v in state.items():
        k2 = k[7:] if k.startswith("module.") else k
        if k2 in model_state and model_state[k2].shape == v.shape:
            cleaned[k2] = v
        else:
            skipped.append(k2)

    model.load_state_dict(cleaned, strict=False)

    total = len(model_state)
    matched = len(cleaned)
    print(f"[smart_load_state_dict] matched keys: {matched}/{total}")

    if matched < 0.8 * total:
        raise RuntimeError(
            f"Too few keys matched when loading checkpoint: {matched}/{total}. "
            f"This usually means model architecture mismatch."
        )

    return ckpt


def save_checkpoint(path: str, epoch: int, model_init: nn.Module, model_refine: nn.Module,
                    optimizer: torch.optim.Optimizer, best_val_psnr: float, args_dict: Dict):
    torch.save({
        "epoch": epoch,
        "model_init": model_init.state_dict(),
        "model_refine": model_refine.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_psnr": best_val_psnr,
        "args": args_dict,
    }, path)


# ============================================================
# Train / eval
# ============================================================

def build_aux(init_pred: torch.Tensor, aux_mode: str) -> Optional[torch.Tensor]:
    if aux_mode == "none":
        return None
    if aux_mode == "edge":
        return sobel_edges(init_pred)
    if aux_mode == "edge_hf":
        blur = F.avg_pool2d(init_pred, kernel_size=5, stride=1, padding=2)
        hf = init_pred - blur
        return torch.cat([sobel_edges(init_pred), hf], dim=1)
    raise ValueError(f"Unknown aux_mode: {aux_mode}")


def compute_total_loss(pred: torch.Tensor, target: torch.Tensor,
                       lambda_l1: float, lambda_mse: float,
                       lambda_charb: float, lambda_grad: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss = 0.0
    logs = {}
    if lambda_l1 > 0:
        l1 = F.l1_loss(pred, target)
        loss = loss + lambda_l1 * l1
        logs["l1"] = float(l1.item())
    if lambda_mse > 0:
        mse = F.mse_loss(pred, target)
        loss = loss + lambda_mse * mse
        logs["mse"] = float(mse.item())
    if lambda_charb > 0:
        charb = charbonnier_loss(pred, target)
        loss = loss + lambda_charb * charb
        logs["charb"] = float(charb.item())
    if lambda_grad > 0:
        gl = grad_loss(pred, target)
        loss = loss + lambda_grad * gl
        logs["grad"] = float(gl.item())
    logs["total"] = float(loss.item())
    return loss, logs


def evaluate_initializer_only(model_init: nn.Module, loader: DataLoader,
                              device: torch.device, max_batches: int = 5) -> Dict[str, float]:
    model_init.eval()
    meters = {"mse": 0.0, "rmse": 0.0, "psnr": 0.0, "ssim": 0.0}
    count = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model_init(x)

            bs = x.size(0)
            for i in range(bs):
                met = compute_metrics(pred[i:i + 1], y[i:i + 1])
                for k in meters:
                    meters[k] += met[k]
                count += 1

    for k in meters:
        meters[k] /= max(count, 1)
    return meters


def evaluate(model_init: nn.Module, model_refine: nn.Module, loader: DataLoader,
             device: torch.device, aux_mode: str, save_vis_dir: Optional[str] = None,
             max_vis: int = 20) -> Dict[str, float]:
    model_init.eval()
    model_refine.eval()

    meters = {"mse": 0.0, "rmse": 0.0, "psnr": 0.0, "ssim": 0.0}
    count = 0
    vis_count = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            init_pred = model_init(x)
            aux = build_aux(init_pred, aux_mode)
            pred, _ = model_refine(init_pred, aux)

            bs = x.size(0)
            for i in range(bs):
                met = compute_metrics(pred[i:i + 1], y[i:i + 1])
                for k in meters:
                    meters[k] += met[k]
                count += 1

                if save_vis_dir is not None and vis_count < max_vis:
                    ensure_dir(save_vis_dir)
                    path = os.path.basename(batch["path"][i]).replace(".mat", "")
                    gt_np = y[i, 0].detach().cpu().numpy()
                    init_np = init_pred[i, 0].detach().cpu().numpy()
                    pred_np = pred[i, 0].detach().cpu().numpy()
                    err_init = np.abs(init_np - gt_np)
                    err_ref = np.abs(pred_np - gt_np)
                    vmin = float(gt_np.min())
                    vmax = float(gt_np.max())
                    save_gray_image(gt_np, os.path.join(save_vis_dir, f"{path}_gt.png"), vmin, vmax)
                    save_gray_image(init_np, os.path.join(save_vis_dir, f"{path}_init.png"), vmin, vmax)
                    save_gray_image(pred_np, os.path.join(save_vis_dir, f"{path}_refine.png"), vmin, vmax)
                    save_gray_image(err_init, os.path.join(save_vis_dir, f"{path}_err_init.png"))
                    save_gray_image(err_ref, os.path.join(save_vis_dir, f"{path}_err_refine.png"))
                    vis_count += 1

    for k in meters:
        meters[k] /= max(count, 1)
    return meters


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ensure_dir(args.output_dir)

    train_set = OpenBreastUSOldStyleDataset(args.data_root, split="train")
    val_set = OpenBreastUSOldStyleDataset(args.data_root, split="test")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_init = build_real_initializer(args.init_base_ch).to(device)
    if args.init_ckpt:
        smart_load_state_dict(model_init, args.init_ckpt, device)
    else:
        raise ValueError("--init_ckpt is required for train mode")
    model_init.eval()
    for p in model_init.parameters():
        p.requires_grad = False

    init_metrics = evaluate_initializer_only(model_init, val_loader, device, max_batches=5)
    print(
        "[Init Only] "
        f"mse={init_metrics['mse']:.6f} | "
        f"psnr={init_metrics['psnr']:.4f} | "
        f"ssim={init_metrics['ssim']:.4f}"
    )

    aux_in_ch = 0
    if args.aux_mode == "edge":
        aux_in_ch = 1
    elif args.aux_mode == "edge_hf":
        aux_in_ch = 2

    model_refine = ResidualRefiner(in_ch=1 + aux_in_ch, base_ch=args.refine_base_ch, residual_scale=args.residual_scale).to(device)
    optimizer = torch.optim.Adam(model_refine.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_psnr = -1e9
    best_path = os.path.join(args.output_dir, "best_refiner.pth")
    log_path = os.path.join(args.output_dir, "train_log.txt")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("start training\n")

    for epoch in range(1, args.epochs + 1):
        model_refine.train()
        train_loss_sum = 0.0
        n_train = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            with torch.no_grad():
                init_pred = model_init(x)
                aux = build_aux(init_pred, args.aux_mode)

            pred, residual = model_refine(init_pred, aux)
            loss, logs = compute_total_loss(
                pred, y,
                lambda_l1=args.lambda_l1,
                lambda_mse=args.lambda_mse,
                lambda_charb=args.lambda_charb,
                lambda_grad=args.lambda_grad,
            )

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model_refine.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            train_loss_sum += float(loss.item()) * x.size(0)
            n_train += x.size(0)

        scheduler.step()
        train_loss = train_loss_sum / max(n_train, 1)

        val_metrics = evaluate(model_init, model_refine, val_loader, device, args.aux_mode)
        msg = (
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
            f"train_loss={train_loss:.6f} | "
            f"val_mse={val_metrics['mse']:.6f} | "
            f"val_psnr={val_metrics['psnr']:.4f} | "
            f"val_ssim={val_metrics['ssim']:.4f}"
        )
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        if val_metrics["psnr"] > best_val_psnr:
            best_val_psnr = val_metrics["psnr"]
            save_checkpoint(best_path, epoch, model_init, model_refine, optimizer, best_val_psnr, vars(args))
            print(f"Saved best checkpoint to: {best_path}")

    print("Training done.")
    print(f"Best val PSNR = {best_val_psnr:.4f}")


def run_eval(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ensure_dir(args.output_dir)

    test_set = OpenBreastUSOldStyleDataset(args.data_root, split="test")
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_init = build_real_initializer(args.init_base_ch).to(device)
    if not args.init_ckpt:
        raise ValueError("--init_ckpt is required for eval")
    smart_load_state_dict(model_init, args.init_ckpt, device)
    model_init.eval()

    aux_in_ch = 0
    if args.aux_mode == "edge":
        aux_in_ch = 1
    elif args.aux_mode == "edge_hf":
        aux_in_ch = 2

    model_refine = ResidualRefiner(in_ch=1 + aux_in_ch, base_ch=args.refine_base_ch, residual_scale=args.residual_scale).to(device)
    if not args.refine_ckpt:
        raise ValueError("--refine_ckpt is required for eval")

    ckpt = torch.load(args.refine_ckpt, map_location=device)
    state = ckpt["model_refine"] if "model_refine" in ckpt else ckpt
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model_refine.load_state_dict(cleaned, strict=False)
    model_refine.eval()

    vis_dir = os.path.join(args.output_dir, "visuals")
    metrics = evaluate(model_init, model_refine, test_loader, device, args.aux_mode,
                       save_vis_dir=vis_dir, max_vis=args.max_vis)

    out_txt = os.path.join(args.output_dir, "eval_metrics.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.8f}\n")

    print("===== Eval Results =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    print(f"Saved metrics to: {out_txt}")
    print(f"Saved visuals to: {vis_dir}")


# ============================================================
# Main
# ============================================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--data_root", type=str, required=True, help="Root of oldstyle dataset, containing train/ and test/")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--init_ckpt", type=str, default="", help="Checkpoint of your trained InversionNet baseline")
    parser.add_argument("--refine_ckpt", type=str, default="", help="Checkpoint of trained residual refiner for eval")

    parser.add_argument("--init_base_ch", type=int, default=32)
    parser.add_argument("--refine_base_ch", type=int, default=32)
    parser.add_argument("--aux_mode", type=str, default="none", choices=["none", "edge", "edge_hf"])
    parser.add_argument("--residual_scale", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_mse", type=float, default=0.2)
    parser.add_argument("--lambda_charb", type=float, default=0.0)
    parser.add_argument("--lambda_grad", type=float, default=0.05)

    parser.add_argument("--max_vis", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(args)
    else:
        run_eval(args)
