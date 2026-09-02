#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"

DATA_ROOT="${DATA_ROOT:-/home/featurize/datasets/90024ebe-ceca-4e0d-aab7-55f496b4b5f5}"
ORIGINAL_CONDITION_ROOT="${ORIGINAL_CONDITION_ROOT:-${REPO_ROOT}/condition_cache/inversionnet_b32_blocks2_epoch67}"
OOF_ROOT="${OOF_ROOT:-${REPO_ROOT}/condition_cache/inversionnet_oof5_b32_blocks2_e67}"
FLOW_ROOT="${FLOW_ROOT:-${REPO_ROOT}/generative_runs/icrfm_oof_v2}"

OOF_SCRIPT="${REPO_ROOT}/build_inversionnet_oof5_conditions.py"
FLOW_SCRIPT="${REPO_ROOT}/train_icrfm_oof_v2.py"
MODE="${1:-help}"

oof_common=(
  --data_root "${DATA_ROOT}"
  --original_condition_root "${ORIGINAL_CONDITION_ROOT}"
  --output_root "${OOF_ROOT}"
  --num_folds 5
  --seed 20260902
  --device "${DEVICE}"
  --epochs 67
  --batch_size 8
  --eval_batch_size 8
  --base_ch 32
  --bottleneck_blocks 2
  --lr 2e-4
  --lambda_l1 1.0
  --lambda_mse 0.2
  --lambda_grad 0.1
  --target_min 1400
  --target_max 1600
  --speed_min 1400
  --speed_max 1605
  --use_amp
)

flow_common=(
  --condition_root "${OOF_ROOT}"
  --device "${DEVICE}"
  --seed 20260902
  --speed_min 1400
  --speed_max 1605
  --base_ch 32
  --time_dim 128
  --steps 4
  --solver heun
  --start_sigma_max 0.0
  --zero_start_prob 1.0
  --lambda_endpoint_l1 1.0
  --lambda_endpoint_grad 0.20
)

run_fold() {
  local fold_id="$1"
  "${PYTHON_BIN}" "${OOF_SCRIPT}" \
    --mode fold \
    --fold "${fold_id}" \
    "${oof_common[@]}"
}

case "${MODE}" in
  fold)
    if [[ $# -ne 2 ]]; then
      echo "Usage: bash run_oof5_icrfm_v2.sh fold {0|1|2|3|4}"
      exit 2
    fi
    run_fold "$2"
    ;;

  resume_fold)
    if [[ $# -ne 2 ]]; then
      echo "Usage: bash run_oof5_icrfm_v2.sh resume_fold {0|1|2|3|4}"
      exit 2
    fi
    "${PYTHON_BIN}" "${OOF_SCRIPT}" \
      --mode fold \
      --fold "$2" \
      "${oof_common[@]}" \
      --resume
    ;;

  folds_all)
    for fold_id in 0 1 2 3 4; do
      run_fold "${fold_id}"
    done
    ;;

  finalize)
    "${PYTHON_BIN}" "${OOF_SCRIPT}" \
      --mode finalize \
      "${oof_common[@]}"
    ;;

  flow_smoke)
    "${PYTHON_BIN}" "${FLOW_SCRIPT}" \
      --mode train \
      "${flow_common[@]}" \
      --output_dir "${FLOW_ROOT}_smoke" \
      --max_train 8 \
      --max_val 4 \
      --epochs 2 \
      --batch_size 2 \
      --eval_batch_size 2 \
      --early_stop_patience 0 \
      --use_amp
    ;;

  flow_train)
    "${PYTHON_BIN}" "${FLOW_SCRIPT}" \
      --mode train \
      "${flow_common[@]}" \
      --output_dir "${FLOW_ROOT}" \
      --train_split train \
      --val_split test \
      --val_start 1 \
      --val_end 20 \
      --epochs 100 \
      --batch_size 4 \
      --eval_batch_size 4 \
      --lr 2e-4 \
      --ssim_guard 0.001 \
      --early_stop_patience 30 \
      --augment \
      --use_amp
    ;;

  flow_resume)
    "${PYTHON_BIN}" "${FLOW_SCRIPT}" \
      --mode train \
      "${flow_common[@]}" \
      --output_dir "${FLOW_ROOT}" \
      --train_split train \
      --val_split test \
      --val_start 1 \
      --val_end 20 \
      --epochs 100 \
      --batch_size 4 \
      --eval_batch_size 4 \
      --lr 2e-4 \
      --ssim_guard 0.001 \
      --early_stop_patience 30 \
      --augment \
      --use_amp \
      --resume
    ;;

  dev30)
    "${PYTHON_BIN}" "${FLOW_SCRIPT}" \
      --mode eval \
      "${flow_common[@]}" \
      --ckpt_path "${FLOW_ROOT}/checkpoints/best.pth" \
      --output_dir "${FLOW_ROOT}/eval_dev30_test21_50" \
      --eval_split test \
      --eval_start 21 \
      --eval_end 50 \
      --eval_batch_size 4
    ;;

  nfe)
    for nfe in 1 2 4 8; do
      "${PYTHON_BIN}" "${FLOW_SCRIPT}" \
        --mode eval \
        "${flow_common[@]}" \
        --steps "${nfe}" \
        --ckpt_path "${FLOW_ROOT}/checkpoints/best.pth" \
        --output_dir "${FLOW_ROOT}/eval_dev30_nfe${nfe}" \
        --eval_split test \
        --eval_start 21 \
        --eval_end 50 \
        --eval_batch_size 4
    done
    ;;

  *)
    echo "Usage: bash run_oof5_icrfm_v2.sh {fold N|resume_fold N|folds_all|finalize|flow_smoke|flow_train|flow_resume|dev30|nfe}"
    exit 2
    ;;
esac

