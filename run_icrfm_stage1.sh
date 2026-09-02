#!/usr/bin/env bash
set -euo pipefail

# Run from the USCT_download repository root.
REPO_ROOT="$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"

# This cache must have been created by the frozen epoch-67 InversionNet using
# precompute_inversionnet_conditions_v73.py.
CONDITION_ROOT="${CONDITION_ROOT:-${REPO_ROOT}/condition_cache/inversionnet_b32_blocks2_epoch67}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/generative_runs/icrfm_v1}"
SCRIPT="${REPO_ROOT}/train_icrfm_v73.py"

MODE="${1:-help}"

common=(
  --condition_root "${CONDITION_ROOT}"
  --device "${DEVICE}"
  --seed 20260902
  --speed_min 1400
  --speed_max 1605
  --base_ch 32
  --time_dim 128
  --steps 4
  --solver heun
)

case "${MODE}" in
  smoke)
    # Interface/VRAM check only. Never use this checkpoint in a paper table.
    "${PYTHON_BIN}" "${SCRIPT}" \
      --mode train \
      "${common[@]}" \
      --output_dir "${RUN_ROOT}_smoke" \
      --max_train 8 \
      --max_val 4 \
      --epochs 2 \
      --batch_size 2 \
      --eval_batch_size 2 \
      --early_stop_patience 0 \
      --use_amp
    ;;

  train)
    # Frozen model-development protocol:
    #   train cache: all 897 training samples
    #   checkpoint selection: old test1-20 only
    "${PYTHON_BIN}" "${SCRIPT}" \
      --mode train \
      "${common[@]}" \
      --output_dir "${RUN_ROOT}" \
      --train_split train \
      --val_split test \
      --val_start 1 \
      --val_end 20 \
      --epochs 100 \
      --batch_size 4 \
      --eval_batch_size 4 \
      --lr 2e-4 \
      --start_sigma_max 0.15 \
      --zero_start_prob 0.25 \
      --lambda_endpoint_l1 0.10 \
      --lambda_endpoint_grad 0.05 \
      --ssim_guard 0.001 \
      --early_stop_patience 20 \
      --augment \
      --use_amp
    ;;

  resume)
    "${PYTHON_BIN}" "${SCRIPT}" \
      --mode train \
      "${common[@]}" \
      --output_dir "${RUN_ROOT}" \
      --train_split train \
      --val_split test \
      --val_start 1 \
      --val_end 20 \
      --epochs 100 \
      --batch_size 4 \
      --eval_batch_size 4 \
      --lr 2e-4 \
      --start_sigma_max 0.15 \
      --zero_start_prob 0.25 \
      --lambda_endpoint_l1 0.10 \
      --lambda_endpoint_grad 0.05 \
      --ssim_guard 0.001 \
      --early_stop_patience 20 \
      --augment \
      --use_amp \
      --resume
    ;;

  dev30)
    # Run once after training/checkpoint selection is complete.  This remains a
    # development-set report, not an independent final holdout.
    "${PYTHON_BIN}" "${SCRIPT}" \
      --mode eval \
      "${common[@]}" \
      --ckpt_path "${RUN_ROOT}/checkpoints/best.pth" \
      --output_dir "${RUN_ROOT}/eval_dev30_test21_50" \
      --eval_split test \
      --eval_start 21 \
      --eval_end 50 \
      --eval_batch_size 4
    ;;

  nfe)
    # Few-step ablation; model checkpoint is unchanged.
    for nfe in 1 2 4 8; do
      "${PYTHON_BIN}" "${SCRIPT}" \
        --mode eval \
        "${common[@]}" \
        --steps "${nfe}" \
        --ckpt_path "${RUN_ROOT}/checkpoints/best.pth" \
        --output_dir "${RUN_ROOT}/eval_dev30_nfe${nfe}" \
        --eval_split test \
        --eval_start 21 \
        --eval_end 50 \
        --eval_batch_size 4
    done
    ;;

  *)
    echo "Usage: bash run_icrfm_stage1.sh {smoke|train|resume|dev30|nfe}"
    echo "Optional environment variables: CONDITION_ROOT, RUN_ROOT, DEVICE, PYTHON_BIN"
    exit 2
    ;;
esac

