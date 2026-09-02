#!/usr/bin/env bash
set -euo pipefail

# Run this file from the latest 0815 / USCT experiment repository root.
# Override any path without editing the script, for example:
#   DATA_ROOT=/actual/path CKPT_PATH=/actual/model.pth bash run_structure_preserving_dev30.sh preflight

DATA_ROOT="${DATA_ROOT:-./dev30_test21_50/local_alpha01}"
CKPT_PATH="${CKPT_PATH:-./adjoint_aware_runs/clean_reverse_train20_val10_lf1_lg3/checkpoints/epoch_001.pth}"
MEAN_DOBS_PATH="${MEAN_DOBS_PATH:-./dobs_mean_baseline/local_alpha_train100_test20/mean_dobs_train.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-./dev30_test21_50/results/structure_preserving_e35}"
DEVICE="${DEVICE:-cuda:0}"
MODE="${1:-screen10}"

COMMON=(
  --repo_root .
  --data_root "$DATA_ROOT"
  --ckpt_path "$CKPT_PATH"
  --mean_dobs_path "$MEAN_DOBS_PATH"
  --output_dir "$OUTPUT_DIR"
  --baseline_csv ./dev30_test21_50/results/local_aa/sample_results.csv
  --device "$DEVICE"
  --num_steps 8
  --base_step_mps 0.5
  --step_factors 1.0 0.5 0.25 0.1
  --frequency 500000
  --forward_iters 80
  --boundary_width 300
  --boundary_strength 225
  --boundary_type PML3
)

case "$MODE" in
  preflight)
    python run_structure_preserving_dev30.py "${COMMON[@]}" --dry_run
    ;;
  smoke3)
    python run_structure_preserving_dev30.py "${COMMON[@]}" \
      --sample_id_start 21 --sample_id_end 23 --allow_non_dev30 \
      --rhos 1.0 0.5 --gammas 0.5
    ;;
  screen10)
    python run_structure_preserving_dev30.py "${COMMON[@]}" \
      --sample_id_start 21 --sample_id_end 30 --allow_non_dev30 \
      --rhos 1.0 0.75 0.5 0.25 --gammas 0.5
    ;;
  dev30)
    # Run only the configuration frozen after screen10.  The current first-step
    # evidence points to rho=0.5, gamma=0.5; do not change it after DEV30 starts.
    python run_structure_preserving_dev30.py "${COMMON[@]}" \
      --sample_id_start 21 --sample_id_end 50 \
      --rhos 0.5 --gammas 0.5
    ;;
  regression)
    # rho=1 should agree with the existing Local-AA selection, up to exact ties.
    python run_structure_preserving_dev30.py "${COMMON[@]}" \
      --sample_id_start 21 --sample_id_end 50 \
      --rhos 1.0 --gammas 0.5
    ;;
  *)
    echo "Usage: bash run_structure_preserving_dev30.sh {preflight|smoke3|screen10|dev30|regression}" >&2
    exit 2
    ;;
esac
