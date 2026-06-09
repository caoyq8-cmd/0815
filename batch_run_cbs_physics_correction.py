import os
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np


def numeric_key(path):
    import re
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--self_cbs_root", type=str, required=True)
    parser.add_argument("--condition_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--max_samples", type=int, default=10)

    parser.add_argument("--num_iters", type=int, default=8)
    parser.add_argument("--step_size_mps", type=float, default=1.0)
    parser.add_argument("--prior_tether", type=float, default=0.1)

    parser.add_argument("--frequency", type=float, default=500000)
    parser.add_argument("--forward_iters", type=int, default=80)
    parser.add_argument("--adjoint_iters", type=int, default=80)
    parser.add_argument("--boundary_width", type=int, default=300)
    parser.add_argument("--boundary_strength", type=float, default=225)
    parser.add_argument("--boundary_type", type=str, default="PML3")
    parser.add_argument("--smooth_kernel", type=int, default=9)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--save_every", type=int, default=1000000)

    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    sample_dir = Path(args.self_cbs_root) / args.split
    files = sorted(sample_dir.glob(f"{args.split}_*.npz"), key=numeric_key)

    if args.max_samples > 0:
        files = files[:args.max_samples]

    print(f"Found {len(files)} samples")

    all_rows = []

    for i, sample_path in enumerate(files, start=1):
        condition_path = Path(args.condition_root) / args.split / sample_path.name

        if not condition_path.exists():
            raise FileNotFoundError(f"Missing condition file: {condition_path}")

        sample_name = sample_path.stem
        out_dir = Path(args.output_root) / sample_name

        cmd = [
            "python", "run_cbs_physics_correction.py",
            "--sample_path", str(sample_path),
            "--condition_path", str(condition_path),
            "--output_dir", str(out_dir),
            "--num_iters", str(args.num_iters),
            "--step_size_mps", str(args.step_size_mps),
            "--prior_tether", str(args.prior_tether),
            "--try_both_signs",
            "--frequency", str(args.frequency),
            "--forward_iters", str(args.forward_iters),
            "--adjoint_iters", str(args.adjoint_iters),
            "--boundary_width", str(args.boundary_width),
            "--boundary_strength", str(args.boundary_strength),
            "--boundary_type", args.boundary_type,
            "--smooth_kernel", str(args.smooth_kernel),
            "--device", args.device,
            "--save_every", str(args.save_every),
        ]

        print("=" * 100)
        print(f"[{i}/{len(files)}] Running {sample_name}")
        print(" ".join(cmd))
        print("=" * 100)

        subprocess.run(cmd, check=True)

        history_path = out_dir / "history.json"
        with open(history_path, "r", encoding="utf-8") as f:
            hist = json.load(f)

        init = hist[0]
        final = hist[-1]
        best_img = min(hist, key=lambda x: x["image_mse_256"])
        best_meas = min(hist, key=lambda x: x["dobs_abs_loss"])

        row = {
            "sample": sample_name,

            "init_mse_256": init["image_mse_256"],
            "final_mse_256": final["image_mse_256"],
            "best_mse_256": best_img["image_mse_256"],
            "best_mse_256_iter": best_img["iter"],

            "init_mse_480": init["image_mse_480"],
            "final_mse_480": final["image_mse_480"],
            "best_mse_480": best_img["image_mse_480"],

            "init_dobs_abs_loss": init["dobs_abs_loss"],
            "final_dobs_abs_loss": final["dobs_abs_loss"],
            "best_dobs_abs_loss": best_meas["dobs_abs_loss"],
            "best_dobs_iter": best_meas["iter"],

            "init_dobs_rel_l1": init["dobs_rel_l1"],
            "final_dobs_rel_l1": final["dobs_rel_l1"],

            "mse256_improve_ratio": (init["image_mse_256"] - final["image_mse_256"]) / init["image_mse_256"],
            "mse480_improve_ratio": (init["image_mse_480"] - final["image_mse_480"]) / init["image_mse_480"],
            "dobs_loss_improve_ratio": (init["dobs_abs_loss"] - final["dobs_abs_loss"]) / init["dobs_abs_loss"],
        }

        all_rows.append(row)

        with open(Path(args.output_root) / "summary.json", "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2, ensure_ascii=False)

    keys = [
        "init_mse_256", "final_mse_256", "best_mse_256",
        "init_mse_480", "final_mse_480", "best_mse_480",
        "init_dobs_abs_loss", "final_dobs_abs_loss", "best_dobs_abs_loss",
        "mse256_improve_ratio", "mse480_improve_ratio", "dobs_loss_improve_ratio",
    ]

    aggregate = {}
    for k in keys:
        vals = np.array([r[k] for r in all_rows], dtype=np.float64)
        aggregate[k + "_mean"] = float(vals.mean())
        aggregate[k + "_std"] = float(vals.std())

    aggregate["num_samples"] = len(all_rows)
    aggregate["num_mse256_improved"] = int(sum(r["final_mse_256"] < r["init_mse_256"] for r in all_rows))
    aggregate["num_mse480_improved"] = int(sum(r["final_mse_480"] < r["init_mse_480"] for r in all_rows))
    aggregate["num_dobs_loss_improved"] = int(sum(r["final_dobs_abs_loss"] < r["init_dobs_abs_loss"] for r in all_rows))

    with open(Path(args.output_root) / "aggregate.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("[Aggregate]")
    print("=" * 100)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()