import os
import re
import json
import argparse
import subprocess
from pathlib import Path

import numpy as np


def numeric_key(path):
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


def load_final_metrics(history_path):
    with open(history_path, "r", encoding="utf-8") as f:
        hist = json.load(f)

    first = hist[0]
    last = hist[-1]

    return {
        "init_mse_240": first["image_mse_240"],
        "final_mse_240": last["image_mse_240"],
        "init_mse_480": first["image_mse_480_up"],
        "final_mse_480": last["image_mse_480_up"],
        "init_true_cbs_abs_loss": first["true_cbs_abs_loss"],
        "final_true_cbs_abs_loss": last["true_cbs_abs_loss"],
        "init_true_cbs_rel_l1": first["true_cbs_rel_l1"],
        "final_true_cbs_rel_l1": last["true_cbs_rel_l1"],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--self_cbs_root", type=str, required=True)
    parser.add_argument("--condition_root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--mean_dobs_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=10)

    parser.add_argument("--num_iters", type=int, default=6)
    parser.add_argument("--step_size_mps", type=float, default=0.5)
    parser.add_argument("--prior_tether", type=float, default=0.0)
    parser.add_argument("--step_factors", type=str, default="1.0 0.5 0.25 0.1 0.05 0.02")

    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_mse", type=float, default=1.0)

    parser.add_argument("--frequency", type=float, default=500000.0)
    parser.add_argument("--cbs_iters", type=int, default=80)
    parser.add_argument("--boundary_width", type=int, default=300)
    parser.add_argument("--boundary_strength", type=float, default=225.0)
    parser.add_argument("--boundary_type", type=str, default="PML3")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    sample_files = sorted(
        list((Path(args.self_cbs_root) / args.split).glob(f"{args.split}_*.npz")),
        key=numeric_key,
    )

    if args.max_samples > 0:
        sample_files = sample_files[:args.max_samples]

    print("num samples =", len(sample_files))

    rows = []

    for idx, sample_path in enumerate(sample_files, start=1):
        stem = sample_path.stem
        cond_path = Path(args.condition_root) / args.split / f"{stem}.npz"
        sample_out = out_root / stem

        if not cond_path.exists():
            print("[Skip] missing condition:", cond_path)
            continue

        cmd = [
            "python", "run_neural_proposal_cbs_validated_correction.py",
            "--sample_path", str(sample_path),
            "--condition_path", str(cond_path),
            "--ckpt_path", args.ckpt_path,
            "--mean_dobs_path", args.mean_dobs_path,
            "--output_dir", str(sample_out),
            "--num_iters", str(args.num_iters),
            "--step_size_mps", str(args.step_size_mps),
            "--prior_tether", str(args.prior_tether),
            "--lambda_l1", str(args.lambda_l1),
            "--lambda_mse", str(args.lambda_mse),
            "--frequency", str(args.frequency),
            "--cbs_iters", str(args.cbs_iters),
            "--boundary_width", str(args.boundary_width),
            "--boundary_strength", str(args.boundary_strength),
            "--boundary_type", args.boundary_type,
            "--device", args.device,
        ]

        # append step factors
        cmd.append("--step_factors")
        cmd.extend(args.step_factors.split())

        print("=" * 100)
        print(f"[{idx}/{len(sample_files)}] {stem}")
        print(" ".join(cmd))
        print("=" * 100)

        subprocess.run(cmd, check=True)

        hist_path = sample_out / "history.json"
        m = load_final_metrics(hist_path)
        m["sample"] = stem

        m["mse240_improved"] = int(m["final_mse_240"] < m["init_mse_240"])
        m["mse480_improved"] = int(m["final_mse_480"] < m["init_mse_480"])
        m["dobs_improved"] = int(m["final_true_cbs_abs_loss"] < m["init_true_cbs_abs_loss"])

        m["mse240_improve_ratio"] = (
            m["init_mse_240"] - m["final_mse_240"]
        ) / (m["init_mse_240"] + 1e-12)

        m["mse480_improve_ratio"] = (
            m["init_mse_480"] - m["final_mse_480"]
        ) / (m["init_mse_480"] + 1e-12)

        m["dobs_improve_ratio"] = (
            m["init_true_cbs_abs_loss"] - m["final_true_cbs_abs_loss"]
        ) / (m["init_true_cbs_abs_loss"] + 1e-12)

        rows.append(m)

        with open(out_root / "summary.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    def arr(key):
        return np.array([r[key] for r in rows], dtype=np.float64)

    aggregate = {
        "num_samples": len(rows),

        "init_mse_240_mean": float(arr("init_mse_240").mean()),
        "final_mse_240_mean": float(arr("final_mse_240").mean()),

        "init_mse_480_mean": float(arr("init_mse_480").mean()),
        "final_mse_480_mean": float(arr("final_mse_480").mean()),

        "init_true_cbs_abs_loss_mean": float(arr("init_true_cbs_abs_loss").mean()),
        "final_true_cbs_abs_loss_mean": float(arr("final_true_cbs_abs_loss").mean()),

        "mse240_improve_ratio_mean": float(arr("mse240_improve_ratio").mean()),
        "mse480_improve_ratio_mean": float(arr("mse480_improve_ratio").mean()),
        "dobs_improve_ratio_mean": float(arr("dobs_improve_ratio").mean()),

        "num_mse240_improved": int(arr("mse240_improved").sum()),
        "num_mse480_improved": int(arr("mse480_improved").sum()),
        "num_dobs_improved": int(arr("dobs_improved").sum()),
    }

    with open(out_root / "aggregate.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("[Aggregate]")
    print("=" * 100)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()