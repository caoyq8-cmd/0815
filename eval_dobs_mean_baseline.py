import re
import json
import argparse
from pathlib import Path
import numpy as np


def numeric_key(path):
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


def complex_rrmse(pred, target, eps=1e-12):
    num = np.sqrt(np.mean(np.abs(pred - target) ** 2))
    den = np.sqrt(np.mean(np.abs(target) ** 2)) + eps
    return float(num / den)


def complex_mae(pred, target):
    return float(np.mean(np.abs(pred - target)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_train_files", type=int, default=-1)
    parser.add_argument("--max_test_files", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files = sorted(
        list((Path(args.data_root) / "train").glob("train_*.npz")),
        key=numeric_key,
    )
    test_files = sorted(
        list((Path(args.data_root) / "test").glob("test_*.npz")),
        key=numeric_key,
    )

    if args.max_train_files > 0:
        train_files = train_files[:args.max_train_files]
    if args.max_test_files > 0:
        test_files = test_files[:args.max_test_files]

    print("num train files =", len(train_files))
    print("num test files  =", len(test_files))

    train_dobs = []
    for p in train_files:
        d = np.load(p)
        train_dobs.append(d["dobs_complex"].astype(np.complex64))

    train_dobs = np.stack(train_dobs, axis=0)
    mean_dobs = train_dobs.mean(axis=0).astype(np.complex64)

    np.savez_compressed(
        out_dir / "mean_dobs_train.npz",
        mean_dobs=mean_dobs,
    )

    rows = []
    for p in test_files:
        d = np.load(p)
        y = d["dobs_complex"].astype(np.complex64)

        rows.append({
            "sample": p.stem,
            "rrmse": complex_rrmse(mean_dobs, y),
            "mae": complex_mae(mean_dobs, y),
        })

    rrmse = np.array([r["rrmse"] for r in rows], dtype=np.float64)
    mae = np.array([r["mae"] for r in rows], dtype=np.float64)

    result = {
        "num_train": len(train_files),
        "num_test": len(test_files),
        "mean_baseline_dobs_rrmse": float(rrmse.mean()),
        "mean_baseline_dobs_rrmse_std": float(rrmse.std()),
        "mean_baseline_dobs_mae": float(mae.mean()),
        "mean_baseline_dobs_mae_std": float(mae.std()),
        "rows": rows,
    }

    with open(out_dir / "mean_baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()