import os
import re
import json
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from cbs_model import ConvergentBornSeries_Batch


def numeric_sort_key(path):
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


def read_mat_v73(path):
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            arr = f[k][()]

            if hasattr(arr, "dtype") and arr.dtype.names is not None:
                names = arr.dtype.names
                if "real" in names and "imag" in names:
                    arr = arr["real"] + 1j * arr["imag"]

            arr = np.asarray(arr)

            if arr.ndim >= 2:
                arr = np.transpose(arr, axes=list(range(arr.ndim - 1, -1, -1)))

            out[k] = arr

    return out


def upsample_256_to_480(x256):
    x = torch.from_numpy(x256.astype(np.float32))[None, None]
    x480 = F.interpolate(x, size=(480, 480), mode="bilinear", align_corners=False)
    return x480[0, 0].cpu().numpy().astype(np.float32)


def build_indices(x_pos, y_pos, mode="sparse"):
    """
    x_pos, y_pos 来自 processed mat，长度 256，坐标在 480x480 物理域。
    默认 sparse: 每隔 4 个 transducer 取一个，共 64 个。
    """
    x_pos = np.asarray(x_pos).reshape(-1).astype(np.int64)
    y_pos = np.asarray(y_pos).reshape(-1).astype(np.int64)

    if mode == "full":
        idx = np.arange(256)
        src = np.stack([x_pos[idx], y_pos[idx]], axis=1)
        rec = src.copy()

    elif mode == "sparse":
        idx = np.arange(0, 256, 4)
        src = np.stack([x_pos[idx], y_pos[idx]], axis=1)
        rec = src.copy()

    elif mode == "sparse_2":
        idx = np.arange(0, 256, 8)
        src = np.stack([x_pos[idx], y_pos[idx]], axis=1)
        rec = src.copy()

    elif mode == "partial":
        # 兼容旧代码中的 partial 构造方式
        rows = np.arange(0, 64)
        cols = np.arange(128, 128 + 64)
        src = np.stack([x_pos[rows], y_pos[cols]], axis=1)
        rec = src.copy()

    elif mode == "partial_2":
        rows = np.arange(0, 64, 2)
        cols = np.arange(128, 128 + 64, 2)
        src = np.stack([x_pos[rows], y_pos[cols]], axis=1)
        rec = src.copy()

    else:
        raise ValueError(f"Unknown measurement mode: {mode}")

    return src.astype(np.int64), rec.astype(np.int64)


@torch.no_grad()
def run_cbs_forward(target_480, src_indices, rec_indices, args, device):
    """
    target_480: [480,480], numpy float32
    返回:
        dobs: [num_src, num_rec] complex64
        wavefields: [num_src, 480,480] complex64，可选
    """
    sos = torch.from_numpy(target_480.astype(np.float32))[None, None].to(device)

    model = ConvergentBornSeries_Batch(
        f=args.frequency,
        sos=sos,
        boundary_width=[args.boundary_width, args.boundary_width],
        boundary_strength=args.boundary_strength,
        boundary_type=args.boundary_type,
        src_loc_set=src_indices,
        device=device,
    )

    u = model(max_iters=args.cbs_iters)  # [1, num_src, 480,480]

    rec_t = torch.from_numpy(rec_indices.astype(np.int64)).to(device)
    dobs = u[0, :, rec_t[:, 0], rec_t[:, 1]]  # [num_src, num_rec]

    dobs_np = dobs.detach().cpu().numpy().astype(np.complex64)

    if args.save_wavefields:
        wave_np = u[0].detach().cpu().numpy().astype(np.complex64)
    else:
        wave_np = None

    del model, u, sos
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dobs_np, wave_np


def process_split(args, split, device):
    in_dir = Path(args.data_root) / split
    out_dir = Path(args.output_root) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(in_dir.glob(f"{split}_*.mat")), key=numeric_sort_key)

    if split == "train":
        max_n = args.max_train
    else:
        max_n = args.max_test

    if max_n > 0:
        files = files[:max_n]

    print(f"[{split}] num input files = {len(files)}")
    print(f"[{split}] output dir = {out_dir}")

    for i, path in enumerate(files, start=1):
        print("=" * 80)
        print(f"[{split}] {i}/{len(files)} | {path.name}")
        print("=" * 80)

        data = read_mat_v73(str(path))

        target_256 = data["target_256"].astype(np.float32)
        x_pos = data["x_pos"]
        y_pos = data["y_pos"]

        target_480 = upsample_256_to_480(target_256)
        target_480 = np.clip(target_480, args.speed_min, args.speed_max).astype(np.float32)

        src_indices, rec_indices = build_indices(x_pos, y_pos, mode=args.measurement_mode)

        print("target_256 min/max =", float(target_256.min()), float(target_256.max()))
        print("target_480 min/max =", float(target_480.min()), float(target_480.max()))
        print("num_src/num_rec    =", src_indices.shape[0], rec_indices.shape[0])
        print("cbs_iters          =", args.cbs_iters)

        dobs, wavefields = run_cbs_forward(
            target_480,
            src_indices,
            rec_indices,
            args,
            device,
        )

        sample_name = path.stem
        save_path = out_dir / f"{sample_name}.npz"

        save_dict = {
            "target_256": target_256.astype(np.float32),
            "target_480": target_480.astype(np.float32),
            "dobs_complex": dobs.astype(np.complex64),
            "src_indices": src_indices.astype(np.int64),
            "rec_indices": rec_indices.astype(np.int64),
            "source_file": np.array([str(path)]),
            "measurement_mode": np.array([args.measurement_mode]),
            "frequency": np.array([args.frequency], dtype=np.float64),
            "cbs_iters": np.array([args.cbs_iters], dtype=np.int32),
        }

        if wavefields is not None:
            save_dict["wavefields"] = wavefields

        np.savez_compressed(save_path, **save_dict)

        print("saved:", save_path)
        print("dobs shape =", dobs.shape)
        print("dobs abs min/max/mean/std =",
              float(np.abs(dobs).min()),
              float(np.abs(dobs).max()),
              float(np.abs(dobs).mean()),
              float(np.abs(dobs).std()))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--measurement_mode", type=str, default="sparse",
                        choices=["full", "sparse", "sparse_2", "partial", "partial_2"])

    parser.add_argument("--max_train", type=int, default=5)
    parser.add_argument("--max_test", type=int, default=2)

    parser.add_argument("--frequency", type=float, default=500e3)
    parser.add_argument("--cbs_iters", type=int, default=80)

    parser.add_argument("--boundary_width", type=int, default=300)
    parser.add_argument("--boundary_strength", type=float, default=225.0)
    parser.add_argument("--boundary_type", type=str, default="PML3")

    parser.add_argument("--speed_min", type=float, default=1400.0)
    parser.add_argument("--speed_max", type=float, default=1605.0)

    parser.add_argument("--save_wavefields", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    os.makedirs(args.output_root, exist_ok=True)

    with open(os.path.join(args.output_root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    process_split(args, "train", device)
    process_split(args, "test", device)

    print("[Done] self-consistent CBS dataset generated.")


if __name__ == "__main__":
    main()