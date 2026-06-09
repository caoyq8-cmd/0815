import os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt


def read_mat_v73(path):
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            node = f[k]
            arr = node[()]

            if hasattr(arr, "dtype") and arr.dtype.names is not None:
                names = arr.dtype.names
                if "real" in names and "imag" in names:
                    arr = arr["real"] + 1j * arr["imag"]

            arr = np.asarray(arr)

            if arr.ndim >= 2:
                arr = np.transpose(arr, axes=list(range(arr.ndim - 1, -1, -1)))

            out[k] = arr

    return out


def save_img(x, path, title=None, cmap="viridis"):
    plt.figure(figsize=(4, 4))
    plt.imshow(x, cmap=cmap)
    plt.colorbar(fraction=0.046, pad=0.04)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./check_processed_outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mat_path = os.path.join(args.data_root, args.split, f"{args.split}_{args.index}.mat")
    print("reading:", mat_path)

    data = read_mat_v73(mat_path)

    dobs = data["dobs_complex"]
    input_2ch = data["input_2ch"]
    target = data["target_256"]

    print("dobs shape      =", dobs.shape, dobs.dtype)
    print("input_2ch shape =", input_2ch.shape, input_2ch.dtype)
    print("target shape    =", target.shape, target.dtype)

    real_diff = np.max(np.abs(input_2ch[0] - dobs.real))
    imag_diff = np.max(np.abs(input_2ch[1] - dobs.imag))

    print("max |input_2ch[0] - real(dobs)| =", real_diff)
    print("max |input_2ch[1] - imag(dobs)| =", imag_diff)

    print("target min/max/mean/std =",
          float(target.min()),
          float(target.max()),
          float(target.mean()),
          float(target.std()))

    save_img(target, os.path.join(args.output_dir, f"{args.split}_{args.index}_target_256.png"),
             title="target_256", cmap="inferno")

    save_img(dobs.real, os.path.join(args.output_dir, f"{args.split}_{args.index}_dobs_real.png"),
             title="real(dobs_complex)", cmap="seismic")

    save_img(dobs.imag, os.path.join(args.output_dir, f"{args.split}_{args.index}_dobs_imag.png"),
             title="imag(dobs_complex)", cmap="seismic")

    save_img(np.log1p(np.abs(dobs)), os.path.join(args.output_dir, f"{args.split}_{args.index}_log_abs_dobs.png"),
             title="log(1 + abs(dobs_complex))", cmap="viridis")

    print("saved to:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()