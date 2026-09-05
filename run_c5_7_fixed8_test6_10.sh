#!/usr/bin/env bash
set -euo pipefail

ROOT="$PWD"

CKPT="$(cat "$ROOT/c5_wavefield_runs/c5_7_fixed8_checkpoint.txt")"

SAMPLES="$ROOT/c5_wavefield_data/smoke8src_train1_val10/val"
GRADS="$ROOT/c5_wavefield_runs/c5_7_gradient_test6_10_rho08"

BG64="$ROOT/c5_wavefield_runs/c5_4a_mgnoI_source64_test1_rho08.npz"

OUT="$ROOT/c5_wavefield_runs/c5_7_fixed8_gradient_test6_10_rho08"

mkdir -p "$OUT"

echo "Fixed8 checkpoint:"
echo "$CKPT"

for i in 6 7 8 9 10
do
    echo
    echo "================================================================================"
    echo "C5.7 FIXED8 TEST $i"
    echo "================================================================================"

    CUDA_VISIBLE_DEVICES=0 \
    python c5_ano_failure_decompose.py \
      --sample "$SAMPLES/test_${i}.npz" \
      --gradient_npz "$GRADS/test_${i}.npz" \
      --background64_npz "$BG64" \
      --mgno_ckpt "$CKPT" \
      --speed_mean 1488.39 \
      --speed_std 27.53 \
      --wave_scale 3.72290883e-02 \
      --chunk_size 8 \
      --device cuda:0 \
      --output "$OUT/decompose_test${i}.json" \
      2>&1 | tee "$OUT/decompose_test${i}.log"
done
