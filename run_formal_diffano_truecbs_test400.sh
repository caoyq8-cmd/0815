#!/usr/bin/env bash
set -u

ROOT=/home/featurize/work/USCT_repro/USCT_download

PHYS=$ROOT/formal_test400_physics/cbs8

COND=$ROOT/condition_cache/inversionnet_formal_3600_400_400_e89/test

OUT=$ROOT/formal_diffano_truecbs_test400

mkdir -p "$OUT"

for i in $(seq 1 400); do

    SAMPLE=$PHYS/test_${i}.npz
    CONDITION=$COND/test_${i}.npz
    RUN=$OUT/test_${i}

    if [ -f "$RUN/run_summary.json" ]; then
        echo "[SKIP] test_${i}"
        continue
    fi

    mkdir -p "$RUN"

    echo
    echo "================================================================================"
    echo "FORMAL DIFF-ANO TRUE-CBS TEST ${i}/400"
    echo "================================================================================"

    CUDA_VISIBLE_DEVICES=0 \
    python "$ROOT/run_formal_diffano_truecbs.py" \
      --sample_path "$SAMPLE" \
      --condition_path "$CONDITION" \
      --output_dir "$RUN" \
      --num_iters 1 \
      --step_size_mps 1.5 \
      --prior_tether 0 \
      --step_factors 1.0 \
      --direction_mode minus \
      --candidate_validation none \
      --no_include_no_update \
      --smooth_kernel 1 \
      --save_every 0 \
      --skip_initial_vis \
      --device cuda:0 \
      > "$RUN/run.log" 2>&1

    STATUS=$?

    if [ $STATUS -ne 0 ]; then
        echo "[FAIL] test_${i}"
        tail -80 "$RUN/run.log"
        exit $STATUS
    fi

    tail -8 "$RUN/run.log"

done

echo
echo "DONE"
