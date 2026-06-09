import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from cbs_model import ConvergentBornSeries_Batch


def numeric_sort_key(path):
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


def upsample_256_to_480_np(x256):
    x = torch.from_numpy(x256.astype(np.float32))[None, None]
    x480 = F.interpolate(x, size=(480, 480), mode="bilinear", align_corners=False)
    return x480[0, 0].cpu().numpy().astype(np.float32)


@torch.no_grad()
def run_cbs_forward(sos_480, src_indices, rec_indices, args, device):
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
    rec_t = torch.from_numpy(rec_indices.astype(np.int64)).to(device)
    dobs = u[0, :, rec_t[:, 0], rec_t[:, 1]]
    dobs_np = dobs.detach().cpu().numpy().astype(np.complex64)

    del model, u, sos
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dobs_np


def rel_l1(a, b):
    return float(np.mean(np.abs(a - b)) / (np.mean(np.abs(b)) + 1e-12))


def rel_l2(a, b):
    return float(np.sqrt(np.mean(np.abs(a - b) ** 2)) / (np.sqrt(np.mean(np.abs(b) ** 2)) + 1e-12))


def image_metrics(pred, target):
    diff = pred - target
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    return mse, mae, rmse


def save_vis(target_480, cond_480, dobs_gt, dobs_cond, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    err_img = cond_480 - target_480
    resid = dobs_cond - dobs_gt

    items = [
        ("target_480", target_480, "inferno"),
        ("condition_480", cond_480, "inferno"),
        ("condition_minus_target", err_img, "bwr"),
        ("dobs_gt_abs", np.abs(dobs_gt), "viridis"),
        ("dobs_cond_abs", np.abs(dobs_cond), "viridis"),
        ("dobs_residual_abs", np.abs(resid), "viridis"),
    ]

    for name, img, cmap in items:
        plt.figure(figsize=(5, 4))
        if cmap == "bwr":
            m = max(abs(float(img.min())), abs(float(img.max())), 1.0)
            plt.imshow(img, cmap=cmap, vmin=-m, vmax=m)
        else:
            plt.imshow(img, cmap=cmap)

        plt.colorbar()
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_{name}.png"), dpi=150)
        plt.close()


def infer_condition_path(condition_root, split, sample_file_name):
    """
    self-consistent 样本名是 train_1.npz / test_1.npz，
    condition cache 也是 train_1.npz / test_1.npz。
    """
    return Path(condition_root) / split / sample_file_name


def process_one(sample_path, condition_root, split, args, device):
    sample_path = Path(sample_path)
    cond_path = infer_condition_path(condition_root, split, sample_path.name)

    if not cond_path.exists():
        raise FileNotFoundError(f"condition file not found: {cond_path}")

    sc = np.load(sample_path)
    cc = np.load(cond_path)

    target_480 = sc["target_480"].astype(np.float32)
    dobs_gt = sc["dobs_complex"].astype(np.complex64)
    src_indices = sc["src_indices"].astype(np.int64)
    rec_indices = sc["rec_indices"].astype(np.int64)

    condition_256_img = cc["condition_speed"][0].astype(np.float32)
    target_256_img = cc["target_speed"][0].astype(np.float32)

    # 关键修正：
    # condition_cache 是图像坐标；
    # self_consistent_cbs 是 CBS 物理坐标；
    # 两者相差一个 transpose。
    condition_256 = condition_256_img.T
    target_256 = target_256_img.T

    condition_480 = upsample_256_to_480_np(condition_256)
    condition_480 = np.clip(condition_480, args.speed_min, args.speed_max).astype(np.float32)

    # 重新 forward condition
    dobs_cond = run_cbs_forward(
        condition_480,
        src_indices,
        rec_indices,
        args,
        device,
    )

    # 同时检查 GT dobs 是否自洽
    dobs_gt_rerun = run_cbs_forward(
        target_480,
        src_indices,
        rec_indices,
        args,
        device,
    )

    gt_rel_l1 = rel_l1(dobs_gt_rerun, dobs_gt)
    gt_rel_l2 = rel_l2(dobs_gt_rerun, dobs_gt)

    cond_rel_l1 = rel_l1(dobs_cond, dobs_gt)
    cond_rel_l2 = rel_l2(dobs_cond, dobs_gt)

    img_mse_480, img_mae_480, img_rmse_480 = image_metrics(condition_480, target_480)
    img_mse_256, img_mae_256, img_rmse_256 = image_metrics(condition_256, target_256)

    metrics = {
        "sample": str(sample_path),
        "condition_file": str(cond_path),

        "gt_forward_rel_l1": gt_rel_l1,
        "gt_forward_rel_l2": gt_rel_l2,

        "condition_dobs_rel_l1": cond_rel_l1,
        "condition_dobs_rel_l2": cond_rel_l2,

        "image_mse_256": img_mse_256,
        "image_mae_256": img_mae_256,
        "image_rmse_256": img_rmse_256,

        "image_mse_480": img_mse_480,
        "image_mae_480": img_mae_480,
        "image_rmse_480": img_rmse_480,

        "dobs_gt_abs_mean": float(np.abs(dobs_gt).mean()),
        "dobs_cond_abs_mean": float(np.abs(dobs_cond).mean()),
        "dobs_residual_abs_mean": float(np.abs(dobs_cond - dobs_gt).mean()),
    }

    if args.save_vis:
        save_vis(
            target_480,
            condition_480,
            dobs_gt,
            dobs_cond,
            args.output_dir,
            sample_path.stem,
        )

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--self_cbs_root", type=str, required=True)
    parser.add_argument("--condition_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--max_samples", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="./condition_physics_residual_check")

    parser.add_argument("--frequency", type=float, default=500e3)
    parser.add_argument("--cbs_iters", type=int, default=80)

    parser.add_argument("--boundary_width", type=int, default=300)
    parser.add_argument("--boundary_strength", type=float, default=225.0)
    parser.add_argument("--boundary_type", type=str, default="PML3")

    parser.add_argument("--speed_min", type=float, default=1400.0)
    parser.add_argument("--speed_max", type=float, default=1605.0)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_vis", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    split_dir = Path(args.self_cbs_root) / args.split
    files = sorted(list(split_dir.glob(f"{args.split}_*.npz")), key=numeric_sort_key)

    if args.max_samples > 0:
        files = files[:args.max_samples]

    print(f"num samples = {len(files)}")
    all_metrics = []

    for i, p in enumerate(files, start=1):
        print("=" * 80)
        print(f"[{i}/{len(files)}] {p}")
        print("=" * 80)

        m = process_one(p, args.condition_root, args.split, args, device)
        all_metrics.append(m)

        for k, v in m.items():
            if isinstance(v, float):
                print(f"{k}: {v:.8e}")
            else:
                print(f"{k}: {v}")

    # 汇总
    numeric_keys = [k for k in all_metrics[0].keys() if isinstance(all_metrics[0][k], float)]
    summary = {}

    for k in numeric_keys:
        vals = np.array([m[k] for m in all_metrics], dtype=np.float64)
        summary[k + "_mean"] = float(vals.mean())
        summary[k + "_std"] = float(vals.std())

    print("=" * 80)
    print("[Summary]")
    print("=" * 80)
    for k, v in summary.items():
        print(f"{k}: {v:.8e}")

    with open(os.path.join(args.output_dir, f"{args.split}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "all_metrics": all_metrics,
                "summary": summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("saved metrics to:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()