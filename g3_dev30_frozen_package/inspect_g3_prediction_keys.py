#!/usr/bin/env python3
"""Print NPZ paths, keys and shapes when the harmonized evaluator cannot auto-detect G3."""

from pathlib import Path
import argparse
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path,
                   default=Path("dev30_test21_50/results/g3_ensemble_cbs_guard_frozen"))
    p.add_argument("--sample_id", type=int, default=21)
    args = p.parse_args()
    paths = sorted(args.root.rglob(f"*{args.sample_id}*.npz"))
    print(f"root={args.root.resolve()}")
    print(f"matching npz={len(paths)}")
    for path in paths:
        print(f"\nFILE: {path}")
        try:
            with np.load(path, allow_pickle=False) as z:
                for key in z.files:
                    a = np.asarray(z[key])
                    print(f"  {key:32s} shape={str(a.shape):18s} dtype={a.dtype}")
        except Exception as exc:
            print("  ERROR:", repr(exc))


if __name__ == "__main__":
    main()
