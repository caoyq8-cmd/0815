#!/usr/bin/env bash
set -euo pipefail

python evaluate_dev30_all_methods_240.py \
  --condition_root ./condition_cache/inversionnet_oof5_b32_blocks2_e67 \
  --local_alpha_root ./dev30_test21_50/local_alpha01 \
  --g3_output_root ./dev30_test21_50/results/g3_ensemble_cbs_guard_frozen \
  --g3_per_sample_csv ./dev30_test21_50/results/g3_ensemble_cbs_guard_frozen/dev30_per_sample.csv \
  --original_csv ./dev30_test21_50/results/original/sample_results.csv \
  --local_aa_csv ./dev30_test21_50/results/local_aa/sample_results.csv \
  --full_cbs_csv ./dev30_test21_50/results/full_cbs/sample_results.csv \
  --e35_csv ./dev30_test21_50/results/structure_preserving_e35/rho0p5_gamma0p5/sample_results.csv \
  --output_dir ./dev30_test21_50/results/all_methods_harmonized_240
