import os
import argparse
import numpy as np
from scipy.io import loadmat


def describe_array(name, arr):
    arr_np = np.asarray(arr)
    print(f"  {name}:")
    print(f"    shape = {arr_np.shape}")
    print(f"    dtype = {arr_np.dtype}")
    if np.iscomplexobj(arr_np):
        print(f"    real min/max = {np.nanmin(arr_np.real):.6g} / {np.nanmax(arr_np.real):.6g}")
        print(f"    imag min/max = {np.nanmin(arr_np.imag):.6g} / {np.nanmax(arr_np.imag):.6g}")
        print(f"    abs  min/max = {np.nanmin(np.abs(arr_np)):.6g} / {np.nanmax(np.abs(arr_np)):.6g}")
    else:
        print(f"    min/max = {np.nanmin(arr_np):.6g} / {np.nanmax(arr_np):.6g}")
        print(f"    mean/std = {np.nanmean(arr_np):.6g} / {np.nanstd(arr_np):.6g}")


def inspect_one(path):
    print("=" * 100)
    print(f"[Inspect] {path}")
    print("=" * 100)

    data = loadmat(path)
    keys = [k for k in data.keys() if not k.startswith("__")]
    print("keys =", keys)

    for k in keys:
        try:
            describe_array(k, data[k])
        except Exception as e:
            print(f"  {k}: failed to describe, error = {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="/home/featurize/datasets/90024ebe-ceca-4e0d-aab7-55f496b4b5f5"
    )
    parser.add_argument("--train_index", type=int, default=1)
    parser.add_argument("--test_index", type=int, default=1)
    args = parser.parse_args()

    train_path = os.path.join(args.data_root, "train", f"train_{args.train_index}.mat")
    test_path = os.path.join(args.data_root, "test", f"test_{args.test_index}.mat")
    bad_path = os.path.join(args.data_root, "bad_samples.mat")

    print("\n[Directory check]")
    print("data_root exists:", os.path.isdir(args.data_root))
    print("train dir exists:", os.path.isdir(os.path.join(args.data_root, "train")))
    print("test dir exists :", os.path.isdir(os.path.join(args.data_root, "test")))

    train_files = sorted([f for f in os.listdir(os.path.join(args.data_root, "train")) if f.endswith(".mat")])
    test_files = sorted([f for f in os.listdir(os.path.join(args.data_root, "test")) if f.endswith(".mat")])

    print("num train files:", len(train_files))
    print("num test files :", len(test_files))
    print("first train files:", train_files[:10])
    print("first test files :", test_files[:10])

    if os.path.exists(train_path):
        inspect_one(train_path)
    else:
        print(f"[Warning] train sample not found: {train_path}")

    if os.path.exists(test_path):
        inspect_one(test_path)
    else:
        print(f"[Warning] test sample not found: {test_path}")

    if os.path.exists(bad_path):
        inspect_one(bad_path)
    else:
        print(f"[Info] bad_samples.mat not found: {bad_path}")


if __name__ == "__main__":
    main()