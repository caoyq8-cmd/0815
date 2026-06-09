import os
import json
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, workdir=None):
    print("=" * 120)
    print("RUN:", " ".join(cmd))
    print("=" * 120)
    result = subprocess.run(cmd, cwd=workdir)
    if result.returncode != 0:
        print(f"[WARN] command failed with code {result.returncode}")
    return result.returncode


def load_metrics(metrics_path):
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_exec", type=str, default="python")
    parser.add_argument("--bridge_script", type=str, default="run_inversion_cbs_bridge.py")
    parser.add_argument("--workdir", type=str, default=".")
    parser.add_argument("--inversion_ckpt", type=str, required=True)
    parser.add_argument("--base_dir_dobs_eval", type=str, required=True)
    parser.add_argument("--base_dir_speed_eval", type=str, required=True)
    parser.add_argument("--aux_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--sample_indices", type=str, default="0,1,2")
    parser.add_argument("--step_sizes", type=str, default="5,10,20,50")
    parser.add_argument("--num_iters", type=int, default=4)
    parser.add_argument("--measurement_mode", type=str, default="sparse")

    args = parser.parse_args()

    sample_indices = [int(x) for x in args.sample_indices.split(",")]
    step_sizes = [float(x) for x in args.step_sizes.split(",")]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_results = []

    for sample_idx in sample_indices:
        for step_size in step_sizes:
            exp_name = f"sample_{sample_idx:03d}_step_{str(step_size).replace('.', 'p')}"
            exp_dir = output_root / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                args.python_exec,
                args.bridge_script,
                "--inversion_ckpt", args.inversion_ckpt,
                "--base_dir_dobs_eval", args.base_dir_dobs_eval,
                "--base_dir_speed_eval", args.base_dir_speed_eval,
                "--aux_dir", args.aux_dir,
                "--output_dir", str(exp_dir),
                "--eval_index", str(sample_idx),
                "--measurement_mode", args.measurement_mode,
                "--num_iters", str(args.num_iters),
                "--step_size", str(step_size),
            ]

            code = run_command(cmd, workdir=args.workdir)

            metrics_path = exp_dir / "metrics.json"
            history_path = exp_dir / "history.json"

            metrics = load_metrics(metrics_path)
            history = load_metrics(history_path)

            row = {
                "sample_index": sample_idx,
                "step_size": step_size,
                "return_code": code,
                "metrics_path": str(metrics_path),
                "history_path": str(history_path),
            }

            if metrics is not None:
                init_mse = metrics["init"]["mse"]
                ref_mse = metrics["refined"]["mse"]
                init_psnr = metrics["init"]["psnr"]
                ref_psnr = metrics["refined"]["psnr"]
                init_ssim = metrics["init"]["ssim"]
                ref_ssim = metrics["refined"]["ssim"]

                row.update({
                    "init_mse": init_mse,
                    "refined_mse": ref_mse,
                    "delta_mse": ref_mse - init_mse,
                    "init_psnr": init_psnr,
                    "refined_psnr": ref_psnr,
                    "delta_psnr": ref_psnr - init_psnr,
                    "init_ssim": init_ssim,
                    "refined_ssim": ref_ssim,
                    "delta_ssim": ref_ssim - init_ssim,
                })

            if history is not None and "rec_diff" in history and len(history["rec_diff"]) > 0:
                row["rec_diff_start"] = history["rec_diff"][0]
                row["rec_diff_end"] = history["rec_diff"][-1]
                row["delta_rec_diff"] = history["rec_diff"][-1] - history["rec_diff"][0]

            all_results.append(row)

    summary_path = output_root / "sweep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\nSaved summary to:", summary_path)

    # 简单打印一个人眼可读摘要
    print("\n===== Sweep Summary =====")
    for row in all_results:
        print(
            f"sample={row['sample_index']:03d} | step={row['step_size']:>5} | "
            f"delta_mse={row.get('delta_mse', None)} | "
            f"delta_psnr={row.get('delta_psnr', None)} | "
            f"delta_ssim={row.get('delta_ssim', None)} | "
            f"delta_rec_diff={row.get('delta_rec_diff', None)}"
        )


if __name__ == "__main__":
    main()