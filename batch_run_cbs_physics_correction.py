import os
import re
import sys
import csv
import json
import time
import argparse
import subprocess
from pathlib import Path

import numpy as np


LOWER_IS_BETTER = [
    "image_mse_256", "image_mae_256", "image_rmse_256",
    "image_mse_480", "image_mae_480", "image_rmse_480",
    "dobs_abs_loss", "dobs_rel_l1", "dobs_rel_l2",
]
HIGHER_IS_BETTER = [
    "image_psnr_256", "image_ssim_256",
    "image_psnr_480", "image_ssim_480",
]
ALL_METRICS = LOWER_IS_BETTER + HIGHER_IS_BETTER


def numeric_key(path):
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


def safe_ratio(before, after):
    return float((before - after) / (abs(before) + 1e-12))


def maybe_read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_preset(args):
    """论文消融预设，避免手工组合参数时出错。"""
    if args.ablation_preset == "custom":
        return

    # 所有正式 preset 从 full 配置开始。
    args.prior_tether = 0.1
    args.step_factors = [1.0, 0.5, 0.25, 0.1]
    args.direction_mode = "both"
    args.candidate_validation = "cbs"
    args.include_no_update = True
    args.smooth_kernel = 9

    if args.ablation_preset == "full":
        pass
    elif args.ablation_preset == "no_prior":
        args.prior_tether = 0.0
    elif args.ablation_preset == "single_scale":
        args.step_factors = [1.0]
    elif args.ablation_preset == "single_direction":
        args.direction_mode = "minus"
    elif args.ablation_preset == "no_verification":
        # 该组定义为普通固定步长梯度下降：不做多候选真实 CBS 选择。
        args.step_factors = [1.0]
        args.direction_mode = "minus"
        args.candidate_validation = "none"
        args.include_no_update = False
    elif args.ablation_preset == "no_smoothing":
        args.smooth_kernel = 1
    else:
        raise ValueError(f"Unknown preset: {args.ablation_preset}")


def build_row(sample_name, hist, run_summary=None):
    if len(hist) == 0:
        raise ValueError(f"Empty history for {sample_name}")
    init = hist[0]
    final = hist[-1]

    row = {
        "sample": sample_name,
        "init_iter": init.get("iter", 0),
        "final_iter": final.get("iter", len(hist) - 1),
    }

    for metric in ALL_METRICS:
        if metric in init:
            row[f"init_{metric}"] = float(init[metric])
        if metric in final:
            row[f"final_{metric}"] = float(final[metric])

    for metric in LOWER_IS_BETTER:
        ik = f"init_{metric}"
        fk = f"final_{metric}"
        if ik in row and fk in row:
            row[f"{metric}_improve_ratio"] = safe_ratio(row[ik], row[fk])
            row[f"{metric}_improved"] = bool(row[fk] < row[ik])

    for metric in HIGHER_IS_BETTER:
        ik = f"init_{metric}"
        fk = f"final_{metric}"
        if ik in row and fk in row:
            row[f"{metric}_delta"] = float(row[fk] - row[ik])
            row[f"{metric}_improved"] = bool(row[fk] > row[ik])

    # 与旧版 summary 兼容的常用字段。
    if "init_image_mse_256" in row:
        row["init_mse_256"] = row["init_image_mse_256"]
        row["final_mse_256"] = row["final_image_mse_256"]
        row["mse256_improve_ratio"] = row["image_mse_256_improve_ratio"]
    if "init_image_mse_480" in row:
        row["init_mse_480"] = row["init_image_mse_480"]
        row["final_mse_480"] = row["final_image_mse_480"]
        row["mse480_improve_ratio"] = row["image_mse_480_improve_ratio"]
    if "init_dobs_abs_loss" in row:
        row["dobs_loss_improve_ratio"] = row["dobs_abs_loss_improve_ratio"]

    best_img = min(hist, key=lambda x: x.get("image_mse_256", float("inf")))
    best_meas = min(hist, key=lambda x: x.get("dobs_abs_loss", float("inf")))
    row["best_mse_256"] = float(best_img.get("image_mse_256", np.nan))
    row["best_mse_256_iter"] = int(best_img.get("iter", -1))
    row["best_dobs_abs_loss"] = float(best_meas.get("dobs_abs_loss", np.nan))
    row["best_dobs_iter"] = int(best_meas.get("iter", -1))

    if run_summary is not None:
        for key in [
            "runtime_seconds",
            "initial_forward_evals",
            "adjoint_step_forward_evals",
            "candidate_selection_forward_evals",
            "post_update_eval_forward_evals",
            "total_forward_like_evals_excluding_adjoint_solve",
        ]:
            if key in run_summary:
                row[key] = run_summary[key]

    return row


def aggregate_rows(rows):
    if len(rows) == 0:
        raise RuntimeError("No successful rows to aggregate.")

    aggregate = {"num_samples": len(rows)}

    numeric_keys = sorted({
        k
        for row in rows
        for k, v in row.items()
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
    })
    skip_keys = {"init_iter", "final_iter", "best_mse_256_iter", "best_dobs_iter"}

    for key in numeric_keys:
        if key in skip_keys:
            continue
        vals = []
        for row in rows:
            if key in row and np.isfinite(float(row[key])):
                vals.append(float(row[key]))
        if len(vals) == 0:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        aggregate[f"{key}_mean"] = float(arr.mean())
        aggregate[f"{key}_std"] = float(arr.std())

    for metric in LOWER_IS_BETTER + HIGHER_IS_BETTER:
        flag = f"{metric}_improved"
        if all(flag in r for r in rows):
            aggregate[f"num_{metric}_improved"] = int(sum(bool(r[flag]) for r in rows))

    # 与旧版 aggregate 命名兼容。
    if all("image_mse_256_improved" in r for r in rows):
        aggregate["num_mse256_improved"] = int(sum(r["image_mse_256_improved"] for r in rows))
    if all("image_mse_480_improved" in r for r in rows):
        aggregate["num_mse480_improved"] = int(sum(r["image_mse_480_improved"] for r in rows))
    if all("dobs_abs_loss_improved" in r for r in rows):
        aggregate["num_dobs_loss_improved"] = int(sum(r["dobs_abs_loss_improved"] for r in rows))

    return aggregate


def save_csv(rows, path):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch runner for stable CBS correction and paper ablations."
    )
    parser.add_argument("--self_cbs_root", type=str, required=True)
    parser.add_argument("--condition_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument(
        "--runner_path",
        type=str,
        default=str(Path(__file__).resolve().parent / "run_cbs_physics_correction.py"),
    )

    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--num_iters", type=int, default=8)
    parser.add_argument("--step_size_mps", type=float, default=1.0)
    parser.add_argument("--prior_tether", type=float, default=0.1)
    parser.add_argument("--step_factors", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1])
    parser.add_argument("--direction_mode", choices=["both", "minus", "plus"], default="both")
    parser.add_argument("--candidate_validation", choices=["cbs", "none"], default="cbs")

    no_update_group = parser.add_mutually_exclusive_group()
    no_update_group.add_argument("--include_no_update", dest="include_no_update", action="store_true")
    no_update_group.add_argument("--no_include_no_update", dest="include_no_update", action="store_false")
    parser.set_defaults(include_no_update=True)

    parser.add_argument(
        "--ablation_preset",
        choices=["custom", "full", "no_prior", "single_scale", "single_direction", "no_verification", "no_smoothing"],
        default="custom",
        help="推荐正式消融直接使用 preset。preset 会覆盖对应稳定化参数。",
    )

    parser.add_argument("--frequency", type=float, default=500000)
    parser.add_argument("--forward_iters", type=int, default=80)
    parser.add_argument("--adjoint_iters", type=int, default=80)
    parser.add_argument("--boundary_width", type=int, default=300)
    parser.add_argument("--boundary_strength", type=float, default=225)
    parser.add_argument("--boundary_type", type=str, default="PML3")
    parser.add_argument("--smooth_kernel", type=int, default=9)
    parser.add_argument("--speed_min", type=float, default=1400.0)
    parser.add_argument("--speed_max", type=float, default=1605.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_every", type=int, default=1000000)
    parser.add_argument("--skip_initial_vis", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    apply_preset(args)
    batch_t0 = time.perf_counter()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(output_root / "batch_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    sample_dir = Path(args.self_cbs_root) / args.split
    files = sorted(sample_dir.glob(f"{args.split}_*.npz"), key=numeric_key)
    if args.max_samples > 0:
        files = files[: args.max_samples]
    if not files:
        raise FileNotFoundError(f"No samples found in {sample_dir}")

    runner_path = Path(args.runner_path)
    if not runner_path.exists():
        raise FileNotFoundError(f"Missing runner: {runner_path}")

    print(f"Found {len(files)} samples")
    print("Preset/config:")
    print(json.dumps({
        "ablation_preset": args.ablation_preset,
        "prior_tether": args.prior_tether,
        "step_factors": args.step_factors,
        "direction_mode": args.direction_mode,
        "candidate_validation": args.candidate_validation,
        "include_no_update": args.include_no_update,
        "smooth_kernel": args.smooth_kernel,
    }, indent=2, ensure_ascii=False))

    all_rows = []
    failures = []

    for i, sample_path in enumerate(files, start=1):
        condition_path = Path(args.condition_root) / args.split / sample_path.name
        if not condition_path.exists():
            msg = f"Missing condition file: {condition_path}"
            if args.continue_on_error:
                failures.append({"sample": sample_path.stem, "error": msg})
                print("[ERROR]", msg)
                continue
            raise FileNotFoundError(msg)

        sample_name = sample_path.stem
        out_dir = output_root / sample_name
        history_path = out_dir / "history.json"
        run_summary_path = out_dir / "run_summary.json"

        should_run = not (args.resume and history_path.exists())
        if should_run:
            cmd = [
                sys.executable,
                str(runner_path),
                "--sample_path", str(sample_path),
                "--condition_path", str(condition_path),
                "--output_dir", str(out_dir),
                "--num_iters", str(args.num_iters),
                "--step_size_mps", str(args.step_size_mps),
                "--prior_tether", str(args.prior_tether),
                "--step_factors", *[str(x) for x in args.step_factors],
                "--direction_mode", args.direction_mode,
                "--candidate_validation", args.candidate_validation,
                "--frequency", str(args.frequency),
                "--forward_iters", str(args.forward_iters),
                "--adjoint_iters", str(args.adjoint_iters),
                "--boundary_width", str(args.boundary_width),
                "--boundary_strength", str(args.boundary_strength),
                "--boundary_type", args.boundary_type,
                "--smooth_kernel", str(args.smooth_kernel),
                "--speed_min", str(args.speed_min),
                "--speed_max", str(args.speed_max),
                "--device", args.device,
                "--save_every", str(args.save_every),
            ]
            cmd.append("--include_no_update" if args.include_no_update else "--no_include_no_update")
            if args.skip_initial_vis:
                cmd.append("--skip_initial_vis")

            print("=" * 100)
            print(f"[{i}/{len(files)}] Running {sample_name}")
            print(" ".join(cmd))
            print("=" * 100)
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                failures.append({"sample": sample_name, "returncode": e.returncode, "error": str(e)})
                with open(output_root / "failures.json", "w", encoding="utf-8") as f:
                    json.dump(failures, f, indent=2, ensure_ascii=False)
                if args.continue_on_error:
                    print(f"[ERROR] {sample_name} failed; continue_on_error=True")
                    continue
                raise
        else:
            print(f"[{i}/{len(files)}] Resume: using existing {history_path}")

        hist = maybe_read_json(history_path)
        if hist is None:
            failures.append({"sample": sample_name, "error": f"Missing {history_path}"})
            if args.continue_on_error:
                continue
            raise FileNotFoundError(history_path)

        run_summary = maybe_read_json(run_summary_path)
        row = build_row(sample_name, hist, run_summary)
        all_rows.append(row)

        with open(output_root / "summary.json", "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2, ensure_ascii=False)
        save_csv(all_rows, output_root / "summary.csv")

    if failures:
        with open(output_root / "failures.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)

    aggregate = aggregate_rows(all_rows)
    aggregate["ablation_preset"] = args.ablation_preset
    aggregate["batch_runtime_seconds"] = float(time.perf_counter() - batch_t0)
    aggregate["num_failures"] = len(failures)
    aggregate["effective_config"] = {
        "num_iters": args.num_iters,
        "step_size_mps": args.step_size_mps,
        "prior_tether": args.prior_tether,
        "step_factors": args.step_factors,
        "direction_mode": args.direction_mode,
        "candidate_validation": args.candidate_validation,
        "include_no_update": args.include_no_update,
        "smooth_kernel": args.smooth_kernel,
        "frequency": args.frequency,
        "forward_iters": args.forward_iters,
        "adjoint_iters": args.adjoint_iters,
        "boundary_width": args.boundary_width,
        "boundary_strength": args.boundary_strength,
        "boundary_type": args.boundary_type,
    }

    with open(output_root / "aggregate.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("[Aggregate]")
    print("=" * 100)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
