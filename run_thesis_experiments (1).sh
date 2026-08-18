#!/usr/bin/env bash
set -euo pipefail

# Run this file from the 0815 repository root.
# Edit only these paths first.
SAMPLE_ROOT="./self_consistent_cbs/sparse_64_dobs_train300_test50/test"
COND_ROOT="./condition_cache/inversionnet_b32_blocks2_epoch67/test"
STABLE_RESULT_ROOT="./final_milestones/cbs_correction_sparse64_test50_backtracking_tether0p1"
RESIDUAL_CKPT="PATH_TO_LOCAL_ALPHA_RESIDUAL_OPERATOR_BEST_CKPT.pth"
MEAN_DOBS="PATH_TO_MEAN_DOBS.npz"
DEVICE="cuda:0"

# 1) P0: offline unified metrics. No CBS rerun.
python thesis_eval_saved_cbs_results.py \
  --result_root "$STABLE_RESULT_ROOT" \
  --output_dir ./final_thesis_results/unified_metrics_test50

# 2) P1: Stable-CBS ablation. Start on 10-20 samples; scale to 50 after sanity check.
python run_stable_cbs_ablation_batch.py \
  --sample_root "$SAMPLE_ROOT" \
  --condition_root "$COND_ROOT" \
  --output_root ./final_thesis_results/stable_cbs_ablation_test20 \
  --max_samples 20 \
  --num_iters 8 \
  --step_size_mps 1.0 \
  --prior_tether 0.1 \
  --device "$DEVICE"

# 3) P1: neural-vs-CBS gradient direction. Use pure MSE objective for closer alignment.
python eval_gradient_consistency.py \
  --sample_root "$SAMPLE_ROOT" \
  --condition_root "$COND_ROOT" \
  --ckpt_path "$RESIDUAL_CKPT" \
  --mean_dobs_path "$MEAN_DOBS" \
  --output_dir ./final_thesis_results/gradient_consistency_test20 \
  --max_samples 20 \
  --lambda_l1 0 \
  --lambda_mse 1 \
  --smooth_kernel 9 \
  --device "$DEVICE"

# 4) Optional stronger diagnostic on 10 samples: verify +/- 0.1 m/s normalized directions with true CBS.
# This adds extra CBS calls; run only after step 3 works.
# python eval_gradient_consistency.py \
#   --sample_root "$SAMPLE_ROOT" \
#   --condition_root "$COND_ROOT" \
#   --ckpt_path "$RESIDUAL_CKPT" \
#   --mean_dobs_path "$MEAN_DOBS" \
#   --output_dir ./final_thesis_results/gradient_consistency_true_step_test10 \
#   --max_samples 10 \
#   --lambda_l1 0 --lambda_mse 1 \
#   --verify_step_mps 0.1 \
#   --device "$DEVICE"
