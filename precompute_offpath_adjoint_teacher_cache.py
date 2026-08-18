#!/usr/bin/env python3
"""
Precompute CBS adjoint teachers on OFF-PATH states collected from accepted
adjoint-aware hybrid trajectories.

Input layout (created by run_adjoint_aware_multistep_correction_v2.py):
    trajectory_root/
      train_1/trajectory_states.npz
      train_2/trajectory_states.npz
      ...

Each trajectory file contains:
    speeds_240            [K,H,W], state 0 = initialization, later states = accepted updates
    iter_indices          [K]
    target_gt_dobs
    src_indices, rec_indices
    gt_speed_240, gt_speed_480
    base_sample

For each base, this script selects up to --max_states_per_base accepted off-path
states, spread approximately evenly over the trajectory, computes the true CBS
adjoint gradient, smooths it on the 480 grid, and performs the exact bilinear
transpose-Jacobian pullback to the 240 parameter grid.

The output cache is compatible with train_adjoint_aware_surrogate_finetune_v2.py:
    *_adjoint_teacher.npz
"""

import json
import argparse
from pathlib import Path

import numpy as np
import torch

from precompute_cbs_adjoint_teacher_cache_v2 import (
    resize_np,
    pullback_bilinear_grad,
    compute_cbs_teacher,
)


def numeric_base_key(p):
    name = p.parent.name
    try:
        return int(name.split("_")[-1])
    except Exception:
        return 10**9


def choose_state_indices(num_states, max_states):
    """
    State 0 is initialization and is excluded.
    Choose up to max_states among states 1..K-1, approximately evenly spaced
    and including the deepest available state.
    """
    n_off = num_states - 1
    if n_off <= 0:
        return []
    m = min(int(max_states), n_off)
    if m <= 0:
        return []
    raw = np.linspace(1, n_off, num=m)
    idx = np.unique(np.rint(raw).astype(int)).tolist()
    # Rounding can rarely reduce count. Fill missing slots from unused states.
    if len(idx) < m:
        for j in range(1, n_off + 1):
            if j not in idx:
                idx.append(j)
            if len(idx) == m:
                break
    return sorted(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory_root", type=str, required=True)
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--max_states_per_base", type=int, default=4)

    ap.add_argument("--frequency", type=float, default=500000.0)
    ap.add_argument("--forward_iters", type=int, default=80)
    ap.add_argument("--adjoint_iters", type=int, default=80)
    ap.add_argument("--boundary_width", type=int, default=300)
    ap.add_argument("--boundary_strength", type=float, default=225.0)
    ap.add_argument("--boundary_type", type=str, default="PML3")
    ap.add_argument("--smooth_kernel", type=int, default=9)

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    traj_files = sorted(
        Path(args.trajectory_root).glob("*/trajectory_states.npz"),
        key=numeric_base_key,
    )
    if not traj_files:
        raise RuntimeError(
            f"No */trajectory_states.npz found under {args.trajectory_root}"
        )

    manifest = []
    n_generated = 0
    n_skipped = 0

    print("=" * 105)
    print("Precompute OFF-PATH CBS adjoint teachers")
    print("=" * 105)
    print("trajectory_root =", args.trajectory_root)
    print("num bases       =", len(traj_files))
    print("max states/base =", args.max_states_per_base)
    print("device          =", device)
    print("=" * 105)

    for bi, tp in enumerate(traj_files, start=1):
        z = np.load(tp, allow_pickle=True)

        speeds = z["speeds_240"].astype(np.float32)
        iter_indices = z["iter_indices"].astype(np.int32)
        target_gt_dobs = z["target_gt_dobs"].astype(np.complex64)
        src_indices = z["src_indices"].astype(np.int64)
        rec_indices = z["rec_indices"].astype(np.int64)
        base = str(z["base_sample"])

        image_size = int(speeds.shape[-1])
        selected = choose_state_indices(len(speeds), args.max_states_per_base)

        if not selected:
            print(f"[{bi:03d}/{len(traj_files):03d}] {base}: no accepted off-path states")
            continue

        for state_pos in selected:
            c240 = speeds[state_pos]
            traj_iter = int(iter_indices[state_pos])

            save_path = out / (
                f"{base}_offpath_iter{traj_iter:02d}_state{state_pos:02d}"
                "_adjoint_teacher.npz"
            )

            if args.resume and save_path.exists():
                n_skipped += 1
                continue

            c480 = resize_np(c240, 480)

            g_raw_480, g_smooth_480, pred_projected, reported_loss, true_mse = (
                compute_cbs_teacher(
                    c480,
                    target_gt_dobs,
                    src_indices,
                    rec_indices,
                    args,
                    device,
                )
            )

            g_raw_240 = pullback_bilinear_grad(g_raw_480, image_size, device)
            g_smooth_240 = pullback_bilinear_grad(
                g_smooth_480, image_size, device
            )

            # For an off-path state there is no original pre-generated measurement.
            # Use the true projected CBS forward at this exact state as the point-wise
            # forward target. This keeps the cache compatible with later mixed
            # forward+gradient recovery training.
            point_dobs_original = pred_projected

            np.savez_compressed(
                save_path,
                speed_240=c240.astype(np.float32),
                point_dobs_original=point_dobs_original.astype(np.complex64),
                target_gt_dobs=target_gt_dobs.astype(np.complex64),
                projected_cbs_dobs=pred_projected.astype(np.complex64),
                teacher_grad_raw_240=g_raw_240.astype(np.float32),
                teacher_grad_smooth_240=g_smooth_240.astype(np.float32),
                src_indices=src_indices.astype(np.int64),
                rec_indices=rec_indices.astype(np.int64),
                alpha=np.array(-1.0, dtype=np.float32),
                base_sample=np.array(base),
                trajectory_iter=np.array(traj_iter, dtype=np.int32),
                trajectory_state_position=np.array(state_pos, dtype=np.int32),
                source_file=np.array(str(tp)),
                cbs_reported_loss=np.array(reported_loss, dtype=np.float64),
                true_cbs_mse_to_gt=np.array(true_mse, dtype=np.float64),
            )

            row = {
                "base_sample": base,
                "trajectory_iter": traj_iter,
                "trajectory_state_position": int(state_pos),
                "cache_file": str(save_path),
                "true_cbs_mse_to_gt": float(true_mse),
                "raw_grad_rms_240": float(
                    np.sqrt(np.mean(g_raw_240.astype(np.float64) ** 2))
                ),
                "smooth_grad_rms_240": float(
                    np.sqrt(np.mean(g_smooth_240.astype(np.float64) ** 2))
                ),
            }
            manifest.append(row)
            n_generated += 1

            print(
                f"[{bi:03d}/{len(traj_files):03d}] {base} "
                f"iter={traj_iter:02d} | "
                f"MSE={true_mse:.6e} | "
                f"smoothRMS={row['smooth_grad_rms_240']:.4e}"
            )

    summary = {
        "num_bases_found": len(traj_files),
        "max_states_per_base": int(args.max_states_per_base),
        "num_generated": int(n_generated),
        "num_skipped_existing": int(n_skipped),
        "num_manifest_rows": len(manifest),
    }

    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[DONE]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("saved to:", out.resolve())


if __name__ == "__main__":
    main()
