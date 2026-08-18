#!/usr/bin/env python3
"""
Final blind-holdout guard.

Verifies that candidate samples from the 2700/300 test subset are absent by exact
speed-map content from BOTH the historical 897-train and 100-test datasets used
in development. Then writes a frozen 30-sample manifest.

Supports classic MAT and MATLAB v7.3/HDF5 MAT files.
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def numeric_id(path: Path):
    m = re.search(r"(\d+)(?!.*\d)", path.stem)
    return int(m.group(1)) if m else -1


def load_arrays(path):
    # Classic MAT
    try:
        import scipy.io as sio
        m = sio.loadmat(path)
        out = {
            k: np.asarray(v)
            for k, v in m.items()
            if not k.startswith("__")
            and isinstance(v, np.ndarray)
            and np.issubdtype(v.dtype, np.number)
        }
        if out:
            return out, "scipy"
    except Exception:
        pass

    # MATLAB v7.3
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset) and not name.startswith("#refs#"):
                try:
                    a = np.asarray(obj)
                except Exception:
                    return
                if np.issubdtype(a.dtype, np.number):
                    out[name] = a
        f.visititems(collect)
    if not out:
        raise RuntimeError(f"No numeric arrays found in {path}")
    return out, "h5py"


def score(name, a):
    n = name.lower()
    s = 0
    for token, bonus in [
        ("slice", 1000), ("speed", 900), ("sos", 850),
        ("target", 700), ("sound", 500)
    ]:
        if token in n:
            s += bonus
    a = np.squeeze(a)
    if a.ndim == 2:
        s += 500
    if a.shape == (480, 480):
        s += 800
    elif a.shape == (256, 256):
        s += 500
    elif a.shape == (240, 240):
        s += 400
    return s


def load_speed(path):
    arrays, fmt = load_arrays(path)
    cand = []
    for k, v in arrays.items():
        a = np.squeeze(np.asarray(v))
        if a.ndim == 2 and np.issubdtype(a.dtype, np.number):
            cand.append((score(k, a), k, a))
    if not cand:
        raise RuntimeError(
            f"No 2-D speed-like array in {path}; "
            f"available={[(k, np.asarray(v).shape, str(np.asarray(v).dtype)) for k,v in arrays.items()]}"
        )
    _, name, a = max(cand, key=lambda x: x[0])
    return np.asarray(a, np.float32), name, fmt


def arr_hash(a):
    a = np.ascontiguousarray(np.asarray(a, np.float32))
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def canonical_hash(a):
    # Be robust to MATLAB v7.3 transpose conventions for square images.
    return min(arr_hash(a), arr_hash(a.T))


def file_list(root, prefix):
    root = Path(root)
    return sorted(root.glob(f"{prefix}_*.mat"), key=numeric_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--old_train_root",
        default="/home/featurize/datasets/90024ebe-ceca-4e0d-aab7-55f496b4b5f5/train",
    )
    ap.add_argument(
        "--old_test_root",
        default="/home/featurize/datasets/90024ebe-ceca-4e0d-aab7-55f496b4b5f5/test",
    )
    ap.add_argument(
        "--new_test_root",
        default=(
            "/home/featurize/datasets/3bbeb1cd-4200-4953-9c5d-ca396f7f0c32/"
            "breast_speed_subset_2700_300/test"
        ),
    )
    ap.add_argument(
        "--mapping_csv",
        default=(
            "/home/featurize/datasets/3bbeb1cd-4200-4953-9c5d-ca396f7f0c32/"
            "breast_speed_subset_2700_300/test_mapping.csv"
        ),
    )
    ap.add_argument("--output_dir", default="./final_holdout_guard")
    ap.add_argument("--prefer_start_id", type=int, default=101)
    ap.add_argument("--holdout_n", type=int, default=30)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    old_train = file_list(args.old_train_root, "train")
    old_test = file_list(args.old_test_root, "test")
    new_test = file_list(args.new_test_root, "test")

    print(f"old train files = {len(old_train)}")
    print(f"old test files  = {len(old_test)}")
    print(f"new test files  = {len(new_test)}")

    # Build historical-development hash set.
    dev_hashes = {}
    for split, files in [("old_train", old_train), ("old_test", old_test)]:
        for i, p in enumerate(files, 1):
            a, var, fmt = load_speed(p)
            h = canonical_hash(a)
            dev_hashes.setdefault(h, []).append({
                "split": split,
                "id": numeric_id(p),
                "path": str(p),
                "variable": var,
                "format": fmt,
            })
            if i % 100 == 0 or i == len(files):
                print(f"[{split}] hashed {i}/{len(files)}")

    mapping = pd.read_csv(args.mapping_csv, header=None)
    mapping = mapping.iloc[:, :2].copy()
    mapping.columns = ["subset_test_id", "original_test_id"]
    mapping["subset_test_id"] = mapping["subset_test_id"].astype(int)
    mapping["original_test_id"] = mapping["original_test_id"].astype(int)

    rows = []
    for i, p in enumerate(new_test, 1):
        a, var, fmt = load_speed(p)
        h = canonical_hash(a)
        sid = numeric_id(p)
        overlaps = dev_hashes.get(h, [])

        mr = mapping[mapping["subset_test_id"] == sid]
        oid = int(mr.iloc[0]["original_test_id"]) if len(mr) else -1

        rows.append({
            "subset_test_id": sid,
            "original_test_id": oid,
            "source_file": str(p),
            "speed_variable": var,
            "mat_format": fmt,
            "content_hash": h,
            "overlaps_historical_development": bool(overlaps),
            "overlap_locations": ";".join(
                f"{x['split']}:{x['id']}" for x in overlaps
            ),
        })
        if i % 50 == 0 or i == len(new_test):
            print(f"[new_test] checked {i}/{len(new_test)}")

    all_df = pd.DataFrame(rows)
    all_df.to_csv(out / "new300_vs_historical_development.csv", index=False)

    clean = all_df[~all_df["overlaps_historical_development"]].copy()
    preferred = clean[clean["subset_test_id"] >= args.prefer_start_id]
    earlier = clean[clean["subset_test_id"] < args.prefer_start_id]
    selected = pd.concat([preferred, earlier], ignore_index=True).head(args.holdout_n)

    if len(selected) < args.holdout_n:
        raise RuntimeError(
            f"Only {len(selected)} development-unseen candidates remain; "
            f"requested {args.holdout_n}"
        )

    selected = selected.copy()
    selected["frozen_final_holdout"] = True
    selected.to_csv(out / "final_holdout_manifest_frozen.csv", index=False)

    selected_ids = selected["subset_test_id"].astype(int).tolist()
    expected_101_130 = list(range(101, 131))
    report = {
        "old_train_file_count": len(old_train),
        "old_test_file_count": len(old_test),
        "new_test_file_count": len(new_test),
        "new_test_overlap_with_historical_development_count": int(
            all_df["overlaps_historical_development"].sum()
        ),
        "new_test_fully_unseen_count": int(len(clean)),
        "selected_holdout_n": int(len(selected)),
        "selected_subset_test_ids": selected_ids,
        "selected_original_test_ids": selected["original_test_id"].astype(int).tolist(),
        "selected_is_exactly_101_to_130": selected_ids == expected_101_130,
        "selected_overlap_count": int(
            selected["overlaps_historical_development"].sum()
        ),
    }

    with open(out / "final_holdout_guard_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("saved:", out.resolve())


if __name__ == "__main__":
    main()
