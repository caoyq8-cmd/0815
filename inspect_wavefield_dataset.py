import os
import re
import argparse
from pathlib import Path
import numpy as np


def numeric_key(path):
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_files", type=int, default=5)
    args = parser.parse_args()

    files = sorted(
        list((Path(args.data_root) / args.split).glob(f"{args.split}_*.npz")),
        key=numeric_key,
    )

    if args.max_files > 0:
        files = files[:args.max_files]

    print("num files inspected =", len(files))

    all_abs = []
    all_real = []
    all_imag = []

    for p in files:
        print("=" * 80)
        print(p)
        d = np.load(p)

        w = d["wavefields"]
        s = d["target_480"]
        y = d["dobs_complex"]

        print("target_480:", s.shape, s.dtype, float(s.min()), float(s.max()))
        print("wavefields:", w.shape, w.dtype)
        print("dobs:", y.shape, y.dtype)

        absw = np.abs(w)
        print("wave abs min/max/mean/std:",
              float(absw.min()), float(absw.max()), float(absw.mean()), float(absw.std()))
        print("wave real min/max/mean/std:",
              float(w.real.min()), float(w.real.max()), float(w.real.mean()), float(w.real.std()))
        print("wave imag min/max/mean/std:",
              float(w.imag.min()), float(w.imag.max()), float(w.imag.mean()), float(w.imag.std()))

        all_abs.append(absw.reshape(-1))
        all_real.append(w.real.reshape(-1))
        all_imag.append(w.imag.reshape(-1))

    all_abs = np.concatenate(all_abs)
    all_real = np.concatenate(all_real)
    all_imag = np.concatenate(all_imag)

    print("=" * 80)
    print("[Global approximate stats]")
    print("=" * 80)
    for name, arr in [
        ("abs", all_abs),
        ("real", all_real),
        ("imag", all_imag),
    ]:
        print(
            f"{name}: "
            f"min={arr.min():.8e}, "
            f"p50={np.percentile(arr, 50):.8e}, "
            f"p90={np.percentile(arr, 90):.8e}, "
            f"p95={np.percentile(arr, 95):.8e}, "
            f"p99={np.percentile(arr, 99):.8e}, "
            f"max={arr.max():.8e}, "
            f"mean={arr.mean():.8e}, "
            f"std={arr.std():.8e}"
        )


if __name__ == "__main__":
    main()