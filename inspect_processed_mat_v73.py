import os
import argparse
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from scipy.io import loadmat


def matlab_h5_to_numpy(node):
    """
    读取 MATLAB v7.3 HDF5 数据。
    MATLAB v7.3 用 HDF5 存储，维度顺序通常需要反转：
        raw shape (256,256,2) -> numpy shape (2,256,256)
        raw shape (1,256)     -> numpy shape (256,1)
    """
    if isinstance(node, h5py.Dataset):
        arr = node[()]

        # MATLAB complex 有时以 compound dtype 存储
        if hasattr(arr, "dtype") and arr.dtype.names is not None:
            names = arr.dtype.names
            if "real" in names and "imag" in names:
                arr = arr["real"] + 1j * arr["imag"]

        arr = np.asarray(arr)

        # 字符串或 object 不做数值处理
        if arr.dtype.kind in ["S", "U", "O"]:
            return arr

        # MATLAB/HDF5 维度反转
        if arr.ndim >= 2:
            arr = np.transpose(arr, axes=list(range(arr.ndim - 1, -1, -1)))

        return arr

    elif isinstance(node, h5py.Group):
        keys = list(node.keys())

        # MATLAB complex 有时以 group: real/imag 存储
        if "real" in keys and "imag" in keys:
            real = matlab_h5_to_numpy(node["real"])
            imag = matlab_h5_to_numpy(node["imag"])
            return real + 1j * imag

        return {"__group_keys__": keys}

    else:
        return None


def describe_array(name, arr):
    print(f"  {name}:")

    if isinstance(arr, dict):
        print(f"    group keys = {arr.get('__group_keys__')}")
        return

    arr_np = np.asarray(arr)
    print(f"    shape = {arr_np.shape}")
    print(f"    dtype = {arr_np.dtype}")

    if arr_np.size == 0:
        print("    empty array")
        return

    if arr_np.dtype.kind in ["S", "U", "O"]:
        print("    non-numeric array")
        return

    if np.iscomplexobj(arr_np):
        print(f"    real min/max = {np.nanmin(arr_np.real):.6g} / {np.nanmax(arr_np.real):.6g}")
        print(f"    imag min/max = {np.nanmin(arr_np.imag):.6g} / {np.nanmax(arr_np.imag):.6g}")
        print(f"    abs  min/max = {np.nanmin(np.abs(arr_np)):.6g} / {np.nanmax(np.abs(arr_np)):.6g}")
        print(f"    abs mean/std = {np.nanmean(np.abs(arr_np)):.6g} / {np.nanstd(np.abs(arr_np)):.6g}")
    else:
        print(f"    min/max = {np.nanmin(arr_np):.6g} / {np.nanmax(arr_np):.6g}")
        print(f"    mean/std = {np.nanmean(arr_np):.6g} / {np.nanstd(arr_np):.6g}")


def inspect_with_scipy(path):
    data = loadmat(path)
    keys = [k for k in data.keys() if not k.startswith("__")]
    print("format = MATLAB old mat, loaded by scipy.io.loadmat")
    print("keys =", keys)

    for k in keys:
        describe_array(k, data[k])


def inspect_with_h5py(path):
    if h5py is None:
        raise ImportError("当前环境没有 h5py，请先安装：pip install h5py")

    with h5py.File(path, "r") as f:
        keys = [k for k in f.keys() if not k.startswith("#")]
        print("format = MATLAB v7.3 / HDF5, loaded by h5py")
        print("keys =", keys)

        for k in keys:
            try:
                arr = matlab_h5_to_numpy(f[k])
                describe_array(k, arr)
            except Exception as e:
                print(f"  {k}: failed to read, error = {repr(e)}")


def inspect_one(path):
    print("=" * 100)
    print(f"[Inspect] {path}")
    print("=" * 100)

    try:
        inspect_with_scipy(path)
    except NotImplementedError as e:
        print("[Info] scipy.io.loadmat failed because this is MATLAB v7.3.")
        print("[Info] switching to h5py...")
        inspect_with_h5py(path)
    except Exception as e:
        print("[Warning] scipy.io.loadmat failed:")
        print(repr(e))
        print("[Info] trying h5py...")
        inspect_with_h5py(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_index", type=int, default=1)
    parser.add_argument("--test_index", type=int, default=1)
    args = parser.parse_args()

    train_dir = os.path.join(args.data_root, "train")
    test_dir = os.path.join(args.data_root, "test")

    train_path = os.path.join(train_dir, f"train_{args.train_index}.mat")
    test_path = os.path.join(test_dir, f"test_{args.test_index}.mat")
    bad_path = os.path.join(args.data_root, "bad_samples.mat")

    print("\n[Directory check]")
    print("data_root exists:", os.path.isdir(args.data_root))
    print("train dir exists:", os.path.isdir(train_dir))
    print("test dir exists :", os.path.isdir(test_dir))

    train_files = sorted([f for f in os.listdir(train_dir) if f.endswith(".mat")])
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".mat")])

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