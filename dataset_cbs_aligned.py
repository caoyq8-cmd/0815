import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils_cbs import normalize as cbs_normalize


class CBSSparseAlignedDataset(Dataset):
    """
    输入:
        dobs_complex: [64,64] complex
        -> [2,64,64] real/imag
        -> bilinear resize -> [2,256,256]

    标签:
        speed_full: [480,480]
        -> utils_cbs.normalize -> [1,256,256]
    """

    def __init__(
        self,
        dobs_root: str,
        speed_root: str,
        resize_to: int = 256,
    ):
        self.dobs_root = dobs_root
        self.speed_root = speed_root
        self.resize_to = resize_to

        self.dobs_files = sorted(glob.glob(os.path.join(dobs_root, "*.npy")))
        if len(self.dobs_files) == 0:
            raise FileNotFoundError(f"No .npy files found in {dobs_root}")

        self.samples = []
        for dobs_path in self.dobs_files:
            name = os.path.basename(dobs_path)
            speed_path = os.path.join(speed_root, name)
            if os.path.exists(speed_path):
                self.samples.append((dobs_path, speed_path))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No matched dobs/speed pairs found.\n"
                f"dobs_root={dobs_root}\n"
                f"speed_root={speed_root}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dobs_path, speed_path = self.samples[idx]

        dobs_complex = np.load(dobs_path)   # [64,64] complex
        speed_full = np.load(speed_path)    # [480,480] float

        # input: [2,64,64] -> [2,256,256]
        x = np.stack([dobs_complex.real, dobs_complex.imag], axis=0).astype(np.float32)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1,2,64,64]
        x = F.interpolate(x, size=(self.resize_to, self.resize_to), mode="bilinear", align_corners=False)
        x = x.squeeze(0)  # [2,256,256]

        # target: [480,480] -> utils_cbs.normalize -> [1,256,256]
        y_480 = torch.tensor(speed_full, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,480,480]
        y_norm = cbs_normalize(y_480).squeeze(0)  # [1,256,256]

        return x, y_norm