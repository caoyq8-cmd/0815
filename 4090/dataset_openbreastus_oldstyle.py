import os
from typing import Tuple

import h5py
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class OpenBreastUSOldStyleDataset(Dataset):
    """
    读取预处理后的 OpenBreastUS 数据：
      - input_2ch: [2, 256, 256]
      - target_256: [256, 256]

    返回：
      x: [2, 256, 256]
      y: [1, 256, 256]
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        normalize_input: bool = True,
        normalize_target: bool = False,
    ):
        super().__init__()
        assert split in ["train", "test"]
        self.data_dir = os.path.join(root_dir, split)
        self.normalize_input = normalize_input
        self.normalize_target = normalize_target

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        prefix = "train_" if split == "train" else "test_"
        self.files = sorted(
            [
                os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                if f.startswith(prefix) and f.endswith(".mat")
            ],
            key=self._extract_index
        )

        if len(self.files) == 0:
            raise RuntimeError(f"在 {self.data_dir} 中没有找到 .mat 文件")

    @staticmethod
    def _extract_index(path: str) -> int:
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        return int(stem.split("_")[-1])

    def __len__(self) -> int:
        return len(self.files)

    def _load_mat_file(self, file_path: str):
        try:
            data = sio.loadmat(file_path)
            x = data["input_2ch"]
            y = data["target_256"]
            return x, y
        except (NotImplementedError, ValueError):
            pass

        with h5py.File(file_path, "r") as f:
            x = f["input_2ch"][()]
            y = f["target_256"][()]
        return x, y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.files[idx]
        x, y = self._load_mat_file(file_path)

        # 统一 x shape -> [2, 256, 256]
        if x.ndim != 3:
            raise ValueError(f"input_2ch 维度异常: {x.shape}, 文件: {file_path}")

        if x.shape == (2, 256, 256):
            pass
        elif x.shape == (256, 256, 2):
            x = x.transpose(2, 0, 1)
        elif x.shape == (256, 2, 256):
            x = x.transpose(1, 0, 2)
        else:
            raise ValueError(f"无法识别 input_2ch 的 shape: {x.shape}, 文件: {file_path}")

        # 统一 y shape -> [256, 256]
        if y.ndim == 2:
            pass
        elif y.ndim == 3:
            if y.shape == (256, 256, 1):
                y = y[:, :, 0]
            elif y.shape == (1, 256, 256):
                y = y[0]
            else:
                raise ValueError(f"无法识别 target_256 的 shape: {y.shape}, 文件: {file_path}")
        else:
            raise ValueError(f"target_256 维度异常: {y.shape}, 文件: {file_path}")

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float().unsqueeze(0)

        if self.normalize_input:
            x_mean = x.mean()
            x_std = x.std()
            if x_std > 1e-8:
                x = (x - x_mean) / x_std

        if self.normalize_target:
            y_min = y.min()
            y_max = y.max()
            if (y_max - y_min) > 1e-8:
                y = (y - y_min) / (y_max - y_min)

        return x, y