import os
import subprocess
import sys

PYTHON = sys.executable

DATA_ROOT = r"/home/featurize/datasets/90024ebe-ceca-4e0d-aab7-55f496b4b5f5"
ROOT_OUT = "./ablation_runs"

FIXED_EVAL_INDICES = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"


def run_command(cmd):
    print("=" * 120)
    print("RUN:", " ".join(cmd))
    print("=" * 120)
    subprocess.run(cmd, check=True)


def run_one_exp(name, extra_train_args):
    out_dir = os.path.join(ROOT_OUT, name)
    ckpt_path = os.path.join(out_dir, "checkpoints", "best.pth")
    eval_dir = os.path.join(out_dir, "eval_fixed20")

    train_cmd = [
        PYTHON, "train_unet_ablation.py",
        "--data_root", DATA_ROOT,
        "--output_dir", out_dir,
        "--num_epochs", "50",
        "--batch_size", "8",
    ] + extra_train_args

    eval_cmd = [
        PYTHON, "eval_unet_ablation.py",
        "--ckpt_path", ckpt_path,
        "--output_dir", eval_dir,
        "--eval_indices", FIXED_EVAL_INDICES,
    ]

    run_command(train_cmd)
    run_command(eval_cmd)


def main():
    # 1) base_ch 消融
    for base_ch in [32, 48, 64]:
        run_one_exp(
            name=f"basech_{base_ch}",
            extra_train_args=[
                "--base_ch", str(base_ch),
                "--lambda_l1", "1.0",
                "--lambda_mse", "0.2",
                "--lambda_grad", "0.1",
                "--use_early_stopping",
            ],
        )

    # 2) 梯度损失消融
    for lambda_grad in [0.0, 0.05, 0.1]:
        run_one_exp(
            name=f"gradloss_{str(lambda_grad).replace('.', 'p')}",
            extra_train_args=[
                "--base_ch", "32",
                "--lambda_l1", "1.0",
                "--lambda_mse", "0.2",
                "--lambda_grad", str(lambda_grad),
                "--use_early_stopping",
            ],
        )

    # 3) early stopping 对照
    run_one_exp(
        name="earlystop_on",
        extra_train_args=[
            "--base_ch", "32",
            "--lambda_l1", "1.0",
            "--lambda_mse", "0.2",
            "--lambda_grad", "0.1",
            "--use_early_stopping",
        ],
    )

    run_one_exp(
        name="earlystop_off",
        extra_train_args=[
            "--base_ch", "32",
            "--lambda_l1", "1.0",
            "--lambda_mse", "0.2",
            "--lambda_grad", "0.1",
        ],
    )


if __name__ == "__main__":
    main()