import os
import argparse
import numpy as np


def mse(a, b):
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def mae(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def transforms(x):
    return {
        "identity": x,
        "transpose": x.T,
        "flipud": np.flipud(x),
        "fliplr": np.fliplr(x),
        "rot90": np.rot90(x, 1),
        "rot180": np.rot90(x, 2),
        "rot270": np.rot90(x, 3),
        "transpose_flipud": np.flipud(x.T),
        "transpose_fliplr": np.fliplr(x.T),
    }


def best_alignment(a, b):
    """
    找 b 经过哪种变换后最接近 a。
    """
    results = []
    for name, bt in transforms(b).items():
        results.append((name, mse(a, bt), mae(a, bt)))
    results.sort(key=lambda x: x[1])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self_sample", type=str, required=True)
    parser.add_argument("--cond_sample", type=str, required=True)
    args = parser.parse_args()

    sc = np.load(args.self_sample)
    cc = np.load(args.cond_sample)

    self_target_256 = sc["target_256"].astype(np.float32)
    self_target_480 = sc["target_480"].astype(np.float32)

    cond_target_speed = cc["target_speed"].astype(np.float32)
    cond_condition_speed = cc["condition_speed"].astype(np.float32)

    if cond_target_speed.ndim == 3:
        cond_target_256 = cond_target_speed[0]
    else:
        cond_target_256 = cond_target_speed

    if cond_condition_speed.ndim == 3:
        cond_condition_256 = cond_condition_speed[0]
    else:
        cond_condition_256 = cond_condition_speed

    print("=" * 100)
    print("[Files]")
    print("=" * 100)
    print("self_sample =", args.self_sample)
    print("cond_sample =", args.cond_sample)

    print("=" * 100)
    print("[Shapes]")
    print("=" * 100)
    print("self_target_256      =", self_target_256.shape, self_target_256.dtype)
    print("self_target_480      =", self_target_480.shape, self_target_480.dtype)
    print("cond_target_256      =", cond_target_256.shape, cond_target_256.dtype)
    print("cond_condition_256   =", cond_condition_256.shape, cond_condition_256.dtype)

    print("=" * 100)
    print("[Ranges]")
    print("=" * 100)
    for name, x in [
        ("self_target_256", self_target_256),
        ("cond_target_256", cond_target_256),
        ("cond_condition_256", cond_condition_256),
    ]:
        print(
            f"{name}: "
            f"min={x.min():.6f}, max={x.max():.6f}, "
            f"mean={x.mean():.6f}, std={x.std():.6f}"
        )

    print("=" * 100)
    print("[Direct comparisons]")
    print("=" * 100)
    print("MSE self_target_256 vs cond_target_256    =", mse(self_target_256, cond_target_256))
    print("MAE self_target_256 vs cond_target_256    =", mae(self_target_256, cond_target_256))
    print("MSE cond_condition_256 vs cond_target_256 =", mse(cond_condition_256, cond_target_256))
    print("MAE cond_condition_256 vs cond_target_256 =", mae(cond_condition_256, cond_target_256))
    print("MSE cond_condition_256 vs self_target_256 =", mse(cond_condition_256, self_target_256))
    print("MAE cond_condition_256 vs self_target_256 =", mae(cond_condition_256, self_target_256))

    print("=" * 100)
    print("[Best transform: cond_target_256 -> self_target_256]")
    print("=" * 100)
    for name, m, a in best_alignment(self_target_256, cond_target_256):
        print(f"{name:20s} mse={m:.8f}, mae={a:.8f}")

    print("=" * 100)
    print("[Best transform: cond_condition_256 -> self_target_256]")
    print("=" * 100)
    for name, m, a in best_alignment(self_target_256, cond_condition_256):
        print(f"{name:20s} mse={m:.8f}, mae={a:.8f}")


if __name__ == "__main__":
    main()