#!/usr/bin/env python3
import argparse, csv, hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd

def numeric_id(path: Path):
    m = re.search(r"(\d+)(?!.*\d)", path.stem)
    return int(m.group(1)) if m else -1

def _load_classic_mat(path):
    import scipy.io as sio
    m = sio.loadmat(path)
    out = {}
    for k, v in m.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            out[k] = np.asarray(v)
    return out

def _load_hdf5_mat(path):
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        def collect(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if name.startswith("#refs#"):
                return
            try:
                a = np.asarray(obj)
            except Exception:
                return
            if np.issubdtype(a.dtype, np.number):
                out[name] = a
        f.visititems(collect)
    return out

def load_numeric_arrays(path):
    err1 = None
    try:
        a = _load_classic_mat(path)
        if a:
            return a, "scipy"
    except Exception as e:
        err1 = repr(e)
    try:
        a = _load_hdf5_mat(path)
        if a:
            return a, "h5py"
    except Exception as e:
        raise RuntimeError(
            f"Could not read {path}\nscipy error: {err1}\nh5py error: {repr(e)}"
        )
    raise RuntimeError(f"No numeric arrays found in {path}")

def candidate_score(name, arr):
    name_l = name.lower()
    score = 0.0
    for token, bonus in [("slice",1000),("speed",900),("sos",850),("target",700),("sound",500)]:
        if token in name_l:
            score += bonus
    if arr.ndim == 2:
        score += 500
    shape = tuple(int(x) for x in arr.shape)
    if shape == (480,480):
        score += 800
    elif shape == (256,256):
        score += 500
    elif shape == (240,240):
        score += 400
    score += min(arr.size/1000.0, 250)
    return score

def choose_speed_array(path):
    arrays, fmt = load_numeric_arrays(path)
    candidates = []
    for name, arr in arrays.items():
        a2 = np.squeeze(np.asarray(arr))
        if a2.ndim != 2:
            continue
        if not np.issubdtype(a2.dtype, np.number):
            continue
        candidates.append((candidate_score(name, a2), name, a2))
    if not candidates:
        raise RuntimeError(
            f"Could not identify a 2-D speed field in {path}; "
            f"available={[ (k, np.asarray(v).shape, str(np.asarray(v).dtype)) for k,v in arrays.items() ]}"
        )
    _, name, a = max(candidates, key=lambda x: x[0])
    a = np.asarray(a, dtype=np.float32)
    meta = {k: {"shape": list(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype)} for k,v in arrays.items()}
    return a, name, fmt, meta

def sha256_array(a):
    x = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    h = hashlib.sha256()
    h.update(str(x.shape).encode("utf-8"))
    h.update(x.tobytes(order="C"))
    return h.hexdigest()

def canonical_image_hash(a):
    a = np.asarray(a, dtype=np.float32)
    return min(sha256_array(a), sha256_array(a.T))

def read_mapping(path):
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 2:
        raise RuntimeError(f"Expected >=2 columns in mapping: {path}, got {df.shape}")
    df = df.iloc[:, :2].copy()
    df.columns = ["subset_test_id", "original_test_id"]
    df["subset_test_id"] = pd.to_numeric(df["subset_test_id"], errors="raise").astype(int)
    df["original_test_id"] = pd.to_numeric(df["original_test_id"], errors="raise").astype(int)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_test_root", default="/home/featurize/datasets/90024ebe-ceca-4e0d-aab7-55f496b4b5f5/test")
    ap.add_argument("--new_test_root", default="/home/featurize/datasets/3bbeb1cd-4200-4953-9c5d-ca396f7f0c32/breast_speed_subset_2700_300/test")
    ap.add_argument("--mapping_csv", default="/home/featurize/datasets/3bbeb1cd-4200-4953-9c5d-ca396f7f0c32/breast_speed_subset_2700_300/test_mapping.csv")
    ap.add_argument("--output_dir", default="./final_holdout_selection")
    ap.add_argument("--holdout_n", type=int, default=30)
    ap.add_argument("--prefer_start_id", type=int, default=101)
    args = ap.parse_args()

    old_root = Path(args.old_test_root)
    new_root = Path(args.new_test_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    old_files = sorted(old_root.glob("test_*.mat"), key=numeric_id)
    new_files = sorted(new_root.glob("test_*.mat"), key=numeric_id)
    print(f"old files={len(old_files)} | new files={len(new_files)}")

    for label, p in [("OLD test_1", old_files[0]), ("NEW test_1", new_files[0])]:
        a, name, fmt, meta = choose_speed_array(p)
        print(f"{label}: fmt={fmt} chosen={name} shape={a.shape} dtype={a.dtype} min={a.min():.3f} max={a.max():.3f}")
        print("  arrays:", meta)

    new_by_hash = {}
    new_meta = {}
    for i, p in enumerate(new_files, 1):
        a, name, fmt, _ = choose_speed_array(p)
        h = canonical_image_hash(a)
        sid = numeric_id(p)
        new_by_hash.setdefault(h, []).append(sid)
        new_meta[sid] = {"path": str(p), "chosen_variable": name, "format": fmt, "shape": list(a.shape), "hash": h}
        if i % 50 == 0 or i == len(new_files):
            print(f"[new] hashed {i}/{len(new_files)}")

    matches = []
    old_hashes = set()
    for i, p in enumerate(old_files, 1):
        a, name, fmt, _ = choose_speed_array(p)
        h = canonical_image_hash(a)
        old_hashes.add(h)
        ids = new_by_hash.get(h, [])
        matches.append({
            "old_subset_test_id": numeric_id(p),
            "old_file": str(p),
            "old_chosen_variable": name,
            "old_format": fmt,
            "matched_new_subset_ids": ";".join(str(x) for x in ids),
            "num_exact_matches_in_new300": len(ids),
        })
        if i % 25 == 0 or i == len(old_files):
            print(f"[old] matched {i}/{len(old_files)}")

    with open(out/"old100_to_new300_matches.csv","w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(matches[0].keys()))
        w.writeheader(); w.writerows(matches)

    mapping = read_mapping(args.mapping_csv)
    mapping.to_csv(out/"test_mapping_parsed.csv", index=False)

    candidates = []
    for p in new_files:
        sid = numeric_id(p)
        h = new_meta[sid]["hash"]
        if h in old_hashes:
            continue
        row = mapping.loc[mapping["subset_test_id"] == sid]
        original_id = int(row.iloc[0]["original_test_id"]) if len(row) else -1
        candidates.append({
            "subset_test_id": sid,
            "original_test_id": original_id,
            "source_file": str(p),
            "exactly_absent_from_old100": True,
            "content_hash": h,
        })

    preferred = [r for r in candidates if r["subset_test_id"] >= args.prefer_start_id]
    fallback = [r for r in candidates if r["subset_test_id"] < args.prefer_start_id]
    selected = (preferred + fallback)[:args.holdout_n]
    if len(selected) < args.holdout_n:
        raise RuntimeError(f"Only {len(selected)} unseen candidates available.")

    with open(out/"final_holdout_manifest.csv","w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
        w.writeheader(); w.writerows(selected)

    n_unique = sum(r["num_exact_matches_in_new300"] == 1 for r in matches)
    n_none = sum(r["num_exact_matches_in_new300"] == 0 for r in matches)
    n_multi = sum(r["num_exact_matches_in_new300"] > 1 for r in matches)
    first100_identity = all(
        r["matched_new_subset_ids"] == str(r["old_subset_test_id"])
        for r in matches[:100]
    ) if len(matches) >= 100 else False

    report = {
        "old_test_count": len(old_files),
        "new_test_count": len(new_files),
        "mapping_rows": int(len(mapping)),
        "old_samples_with_unique_exact_match": int(n_unique),
        "old_samples_with_no_exact_match": int(n_none),
        "old_samples_with_multiple_exact_matches": int(n_multi),
        "old100_equals_new300_first100_by_exact_content": bool(first100_identity),
        "num_new300_samples_exactly_absent_from_old100": int(len(candidates)),
        "requested_holdout_n": int(args.holdout_n),
        "selected_holdout_subset_ids": [int(r["subset_test_id"]) for r in selected],
        "selected_holdout_original_ids": [int(r["original_test_id"]) for r in selected],
    }
    with open(out/"dataset_overlap_report.json","w",encoding="utf-8") as f:
        json.dump(report,f,indent=2,ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("saved:", out.resolve())

if __name__ == "__main__":
    main()
