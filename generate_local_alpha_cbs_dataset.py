import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def numeric_key(path):
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


def to_2d(x):
    x = np.asarray(x)
    while x.ndim > 2:
        x = x[0]
    return x.astype(np.float32)


def resize_np(x, size):
    x_t = torch.from_numpy(x.astype(np.float32))[None, None]
    y_t = F.interpolate(
        x_t,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    return y_t[0, 0].cpu().numpy().astype(np.float32)


@torch.no_grad()
def forward_cbs_dobs(sos_480, src_indices, rec_indices, args, device):
    from cbs_model import ConvergentBornSeries_Batch

    sos = torch.from_numpy(sos_480.astype(np.float32))[None, None].to(device)

    model = ConvergentBornSeries_Batch(
        f=args.frequency,
        sos=sos,
        boundary_width=[args.boundary_width, args.boundary_width],
        boundary_strength=args.boundary_strength,
        boundary_type=args.boundary_type,
        src_loc_set=src_indices.astype(np.int64),
        device=device,
    )

    u = model(max_iters=args.cbs_iters)

    rec_t = torch.from_numpy(rec_indices.astype(np.int64)).long().to(device)
    dobs = u[0, :, rec_t[:, 0], rec_t[:, 1]]
    dobs_np = dobs.detach().cpu().numpy().astype(np.complex64)

    del model, u, sos
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dobs_np


def load_condition_and_gt(self_npz_path, cond_npz_path):
    sc = np.load(self_npz_path)
    cc = np.load(cond_npz_path)

    condition_256_img = to_2d(cc["condition_speed"])

    # 注意：condition cache 是 image coordinate
    # self-consistent CBS 数据是 physics coordinate
    condition_256_phys = condition_256_img.T.astype(np.float32)

    if "target_256" in sc.files:
        gt_256_phys = sc["target_256"].astype(np.float32)
    elif "target_speed" in cc.files:
        gt_256_img = to_2d(cc["target_speed"])
        gt_256_phys = gt_256_img.T.astype(np.float32)
    else:
        raise RuntimeError(
            f"Cannot find GT target_256 in {self_npz_path} or target_speed in {cond_npz_path}"
        )

    return condition_256_phys, gt_256_phys, sc


def process_split(args, split, max_base, device):
    self_split_dir = Path(args.self_cbs_root) / split
    cond_split_dir = Path(args.condition_root) / split
    out_split_dir = Path(args.output_root) / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    self_files = sorted(
        list(self_split_dir.glob(f"{split}_*.npz")),
        key=numeric_key,
    )

    if max_base > 0:
        self_files = self_files[:max_base]

    print("=" * 100)
    print(f"[{split}] base samples = {len(self_files)}")
    print("=" * 100)

    meta_rows = []
    out_counter = 0

    for base_idx, self_path in enumerate(self_files, start=1):
        stem = self_path.stem
        cond_path = cond_split_dir / f"{stem}.npz"

        if not cond_path.exists():
            print(f"[Skip] missing condition file: {cond_path}")
            continue

        condition_256_phys, gt_256_phys, sc = load_condition_and_gt(
            self_path,
            cond_path,
        )

        src_indices = sc["src_indices"].astype(np.int64)
        rec_indices = sc["rec_indices"].astype(np.int64)

        gt_480 = sc["target_480"].astype(np.float32)
        gt_dobs = sc["dobs_complex"].astype(np.complex64)

        condition_480 = resize_np(condition_256_phys, 480)
        condition_480 = np.clip(
            condition_480,
            args.speed_min,
            args.speed_max,
        ).astype(np.float32)

        print("=" * 100)
        print(f"[{split}] base {base_idx}/{len(self_files)} | {stem}")
        print(
            f"condition_256 min/max = {condition_256_phys.min():.4f} / {condition_256_phys.max():.4f}"
        )
        print(
            f"gt_256        min/max = {gt_256_phys.min():.4f} / {gt_256_phys.max():.4f}"
        )
        print("=" * 100)

        for alpha in args.alphas:
            out_counter += 1

            local_256 = (
                (1.0 - alpha) * condition_256_phys
                + alpha * gt_256_phys
            ).astype(np.float32)

            if abs(alpha - 1.0) < 1e-8:
                local_480 = gt_480.copy()
                dobs = gt_dobs.copy()
                reused_gt = True
            elif abs(alpha - 0.0) < 1e-8:
                local_480 = condition_480.copy()
                dobs = forward_cbs_dobs(
                    local_480,
                    src_indices,
                    rec_indices,
                    args,
                    device,
                )
                reused_gt = False
            else:
                local_480 = resize_np(local_256, 480)
                local_480 = np.clip(
                    local_480,
                    args.speed_min,
                    args.speed_max,
                ).astype(np.float32)

                dobs = forward_cbs_dobs(
                    local_480,
                    src_indices,
                    rec_indices,
                    args,
                    device,
                )
                reused_gt = False

            out_name = f"{split}_{out_counter:06d}.npz"
            out_path = out_split_dir / out_name

            np.savez_compressed(
                out_path,
                target_480=local_480.astype(np.float32),
                target_256=local_256.astype(np.float32),
                dobs_complex=dobs.astype(np.complex64),
                alpha=np.array(alpha, dtype=np.float32),
                base_sample=np.array(stem),
                base_index=np.array(base_idx, dtype=np.int32),
                src_indices=src_indices.astype(np.int64),
                rec_indices=rec_indices.astype(np.int64),
                condition_256=condition_256_phys.astype(np.float32),
                gt_256=gt_256_phys.astype(np.float32),
                reused_gt=np.array(reused_gt),
            )

            row = {
                "split": split,
                "out_file": str(out_path),
                "base_sample": stem,
                "base_index": base_idx,
                "alpha": float(alpha),
                "reused_gt": bool(reused_gt),
                "speed_min": float(local_480.min()),
                "speed_max": float(local_480.max()),
                "dobs_abs_min": float(np.abs(dobs).min()),
                "dobs_abs_max": float(np.abs(dobs).max()),
                "dobs_abs_mean": float(np.abs(dobs).mean()),
                "dobs_abs_std": float(np.abs(dobs).std()),
            }
            meta_rows.append(row)

            print(
                f"[{split}] saved {out_name} | "
                f"base={stem} | alpha={alpha:.2f} | "
                f"reused_gt={reused_gt} | "
                f"dobs_abs_mean={row['dobs_abs_mean']:.6f}"
            )

            with open(Path(args.output_root) / f"meta_{split}.json", "w", encoding="utf-8") as f:
                json.dump(meta_rows, f, indent=2, ensure_ascii=False)

    return meta_rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--self_cbs_root", type=str, required=True)
    parser.add_argument("--condition_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--max_train_base", type=int, default=100)
    parser.add_argument("--max_test_base", type=int, default=20)

    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )

    parser.add_argument("--frequency", type=float, default=500000.0)
    parser.add_argument("--cbs_iters", type=int, default=80)
    parser.add_argument("--boundary_width", type=int, default=300)
    parser.add_argument("--boundary_strength", type=float, default=225.0)
    parser.add_argument("--boundary_type", type=str, default="PML3")

    parser.add_argument("--speed_min", type=float, default=1400.0)
    parser.add_argument("--speed_max", type=float, default=1605.0)

    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    with open(Path(args.output_root) / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device =", device)
    print("alphas =", args.alphas)

    train_meta = process_split(
        args,
        split="train",
        max_base=args.max_train_base,
        device=device,
    )

    test_meta = process_split(
        args,
        split="test",
        max_base=args.max_test_base,
        device=device,
    )

    aggregate = {
        "num_train_files": len(train_meta),
        "num_test_files": len(test_meta),
        "num_alphas": len(args.alphas),
        "alphas": args.alphas,
        "output_root": args.output_root,
    }

    with open(Path(args.output_root) / "aggregate.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("[Done] local-alpha CBS dataset generated")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()