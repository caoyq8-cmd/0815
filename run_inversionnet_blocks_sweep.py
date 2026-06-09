import os
import json
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, workdir=None):
    print("=" * 120)
    print("RUN:", " ".join(cmd))
    print("=" * 120)
    result = subprocess.run(cmd, cwd=workdir)
    return result.returncode


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_exec", type=str, default="python")
    parser.add_argument("--workdir", type=str, default=".")
    parser.add_argument("--train_script", type=str, default="train_inversionnet_ablation.py")
    parser.add_argument("--eval_script", type=str, default="eval_inversionnet_ablation.py")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--block_list", type=str, default="2,4,6")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_mse", type=float, default=0.2)
    parser.add_argument("--lambda_grad", type=float, default=0.1)

    parser.add_argument("--eval_indices", type=str,
                        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")

    parser.add_argument("--use_early_stopping", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    block_list = [int(x) for x in args.block_list.split(",") if x.strip()]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_results = []

    for blocks in block_list:
        exp_name = f"inversionnet_b{args.base_ch}_blocks{blocks}"
        exp_dir = output_root / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            args.python_exec, args.train_script,
            "--data_root", args.data_root,
            "--output_dir", str(exp_dir),
            "--base_ch", str(args.base_ch),
            "--bottleneck_blocks", str(blocks),
            "--batch_size", str(args.batch_size),
            "--num_epochs", str(args.num_epochs),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--num_workers", str(args.num_workers),
            "--lambda_l1", str(args.lambda_l1),
            "--lambda_mse", str(args.lambda_mse),
            "--lambda_grad", str(args.lambda_grad),
            "--seed", str(args.seed),
        ]
        if args.use_early_stopping:
            train_cmd.append("--use_early_stopping")

        train_return = run_command(train_cmd, workdir=args.workdir)

        ckpt_path = exp_dir / "checkpoints" / "best.pth"
        eval_dir = exp_dir / "eval_fixed20"

        eval_return = None
        metrics = None

        if ckpt_path.exists():
            eval_cmd = [
                args.python_exec, args.eval_script,
                "--ckpt_path", str(ckpt_path),
                "--output_dir", str(eval_dir),
                "--eval_indices", args.eval_indices,
            ]
            eval_return = run_command(eval_cmd, workdir=args.workdir)

            metrics_path = eval_dir / "metrics_summary.json"
            metrics = load_json(metrics_path)

        row = {
            "base_ch": args.base_ch,
            "bottleneck_blocks": blocks,
            "train_return_code": train_return,
            "eval_return_code": eval_return,
            "exp_dir": str(exp_dir),
        }

        if metrics is not None:
            row.update({
                "checkpoint_epoch": metrics.get("checkpoint_epoch"),
                "mse_mean": metrics.get("mse_mean"),
                "mae_mean": metrics.get("mae_mean"),
                "rmse_mean": metrics.get("rmse_mean"),
                "psnr_mean": metrics.get("psnr_mean"),
                "ssim_mean": metrics.get("ssim_mean"),
            })

        all_results.append(row)

    summary_path = output_root / "blocks_sweep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\nSaved summary to:", summary_path)
    print("\n===== Blocks Sweep Summary =====")
    for row in all_results:
        print(
            f"base_ch={row['base_ch']} | "
            f"blocks={row['bottleneck_blocks']} | "
            f"mse={row.get('mse_mean', None)} | "
            f"psnr={row.get('psnr_mean', None)} | "
            f"ssim={row.get('ssim_mean', None)}"
        )


if __name__ == "__main__":
    main()