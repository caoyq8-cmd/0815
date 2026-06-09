import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_path", type=str, required=True)
    parser.add_argument("--low", type=float, default=1400.0)
    parser.add_argument("--high", type=float, default=1605.0)
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()

    x = np.load(args.sample_path)

    print("shape =", x.shape)
    print("min/max =", float(x.min()), float(x.max()))
    print("mean/std =", float(x.mean()), float(x.std()))

    low_ratio = np.mean(x <= args.low + args.eps)
    high_ratio = np.mean(x >= args.high - args.eps)

    print("low clamp ratio  =", float(low_ratio))
    print("high clamp ratio =", float(high_ratio))

    # 每张图的统计
    if x.ndim == 4:
        for i in range(x.shape[0]):
            xi = x[i]
            print(
                f"[{i:02d}] "
                f"min={xi.min():.3f}, max={xi.max():.3f}, "
                f"mean={xi.mean():.3f}, std={xi.std():.3f}, "
                f"low_ratio={(xi <= args.low + args.eps).mean():.4f}, "
                f"high_ratio={(xi >= args.high - args.eps).mean():.4f}"
            )


if __name__ == "__main__":
    main()