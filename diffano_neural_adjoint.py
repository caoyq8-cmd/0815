import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils_cbs import normalize as cbs_normalize


Tensor = torch.Tensor


@dataclass
class DobsStats:
    scale: float


def list_matched_npy_pairs(dobs_root: str, speed_root: str) -> List[Tuple[str, str]]:
    import os
    import glob

    dobs_files = sorted(glob.glob(os.path.join(dobs_root, "*.npy")))
    pairs: List[Tuple[str, str]] = []
    for dobs_path in dobs_files:
        speed_path = os.path.join(speed_root, os.path.basename(dobs_path))
        if os.path.exists(speed_path):
            pairs.append((dobs_path, speed_path))
    if not pairs:
        raise FileNotFoundError(f"No matched .npy pairs found: {dobs_root} <-> {speed_root}")
    return pairs


def estimate_dobs_scale(pairs: Sequence[Tuple[str, str]], max_files: int = 256) -> DobsStats:
    limit = min(len(pairs), max_files)
    max_abs = 0.0
    for dobs_path, _ in pairs[:limit]:
        dobs = np.load(dobs_path)
        max_abs = max(max_abs, float(np.max(np.abs(dobs.real))))
        max_abs = max(max_abs, float(np.max(np.abs(dobs.imag))))
    return DobsStats(scale=max(max_abs, 1e-8))


class NeuralAdjointDataset(Dataset):
    """
    Matched OpenBreastUS/CBS data for learning a differentiable measurement
    operator A_theta(speed) ~= dobs.

    speed: [480, 480] -> utils_cbs.normalize -> [1, 256, 256]
    dobs: complex [M, M] -> [2, M, M], divided by a global scale
    """

    def __init__(
        self,
        dobs_root: str,
        speed_root: str,
        dobs_scale: Optional[float] = None,
        max_samples: Optional[int] = None,
    ):
        self.pairs = list_matched_npy_pairs(dobs_root, speed_root)
        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]
        if dobs_scale is None:
            dobs_scale = estimate_dobs_scale(self.pairs).scale
        self.dobs_scale = float(dobs_scale)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        dobs_path, speed_path = self.pairs[idx]
        dobs = np.load(dobs_path)
        speed = np.load(speed_path)

        speed_t = torch.tensor(speed, dtype=torch.float32).view(1, 1, *speed.shape)
        speed_norm = cbs_normalize(speed_t).squeeze(0)

        dobs_2ch = np.stack([dobs.real, dobs.imag], axis=0).astype(np.float32)
        dobs_norm = torch.tensor(dobs_2ch, dtype=torch.float32) / self.dobs_scale
        return speed_norm, dobs_norm


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int = 12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        self.weight_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )
        self.weight_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    def compl_mul2d(self, x: Tensor, weights: Tensor) -> Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        mx = min(self.modes, height)
        my = min(self.modes, width // 2 + 1)
        out_ft[:, :, :mx, :my] = self.compl_mul2d(
            x_ft[:, :, :mx, :my],
            self.weight_pos[:, :, :mx, :my],
        )
        out_ft[:, :, -mx:, :my] = self.compl_mul2d(
            x_ft[:, :, -mx:, :my],
            self.weight_neg[:, :, :mx, :my],
        )
        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")


class FNOBlock(nn.Module):
    def __init__(self, channels: int, modes: int, dropout: float = 0.0):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes=modes)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        y = self.spectral(x) + self.pointwise(x)
        y = self.norm(y)
        y = F.gelu(y)
        return x + self.dropout(y)


class NeuralAdjointOperator(nn.Module):
    """
    FNO-style measurement surrogate for DIFF-ANO.

    The model is intentionally direct: it learns boundary measurements from the
    speed map and stays differentiable with respect to the speed input. The
    neural adjoint gradient is obtained by backpropagating the measurement loss.
    """

    def __init__(
        self,
        measurement_shape: Tuple[int, int] = (64, 64),
        width: int = 64,
        modes: int = 16,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.measurement_shape = measurement_shape
        self.lift = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
        )
        self.blocks = nn.ModuleList([FNOBlock(width, modes=modes, dropout=dropout) for _ in range(depth)])
        self.proj = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 2, kernel_size=1),
        )

    @staticmethod
    def coord_grid(batch_size: int, height: int, width: int, device: torch.device) -> Tensor:
        yy = torch.linspace(-1.0, 1.0, height, device=device).view(1, 1, height, 1)
        xx = torch.linspace(-1.0, 1.0, width, device=device).view(1, 1, 1, width)
        yy = yy.expand(batch_size, 1, height, width)
        xx = xx.expand(batch_size, 1, height, width)
        return torch.cat([yy, xx], dim=1)

    def forward(self, speed_norm: Tensor) -> Tensor:
        if speed_norm.ndim != 4 or speed_norm.shape[1] != 1:
            raise ValueError(f"Expected speed_norm [B,1,H,W], got {tuple(speed_norm.shape)}")
        b, _, h, w = speed_norm.shape
        x = torch.cat([speed_norm, self.coord_grid(b, h, w, speed_norm.device)], dim=1)
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        x = F.adaptive_avg_pool2d(x, self.measurement_shape)
        return self.proj(x)


def neural_adjoint_gradient(
    model: nn.Module,
    speed_norm: Tensor,
    target_dobs_norm: Tensor,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    x = speed_norm.detach().clone().requires_grad_(True)
    pred = model(x)
    residual = pred - target_dobs_norm
    if mask is not None:
        residual = residual * mask
    loss = torch.mean(residual ** 2)
    grad = torch.autograd.grad(loss, x, create_graph=False)[0]
    return grad, loss.detach()
