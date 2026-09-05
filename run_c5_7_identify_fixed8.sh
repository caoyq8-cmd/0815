#!/usr/bin/env bash
set -euo pipefail

ROOT="$PWD"
SAMPLE="$ROOT/c5_wavefield_data/smoke8src_train20_val5/val/test_1.npz"
GRAD="$ROOT/c5_wavefield_runs/c5_3c_val5_rho08/test_1.npz"
BG64="$ROOT/c5_wavefield_runs/c5_4a_mgnoI_source64_test1_rho08.npz"
OUT="$ROOT/c5_wavefield_runs/c5_7_fixed8_identify"

mkdir -p "$OUT"
rm -f "$OUT/candidates.tsv"

CANDS=(
  "$ROOT/c5_wavefield_runs/mgnoI_v2_residual_truebg_e200/best.pt"
  "$ROOT/c5_wavefield_runs/mgnoII_v2_residual_truebg_e200/best.pt"
  "$ROOT/c5_wavefield_runs/mgnoI_truebg_train20_val5_smoke2/best.pt"
  "$ROOT/c5_wavefield_runs/mgnoII_v2_residual_truebg_smoke2/best.pt"
)

for CKPT in "${CANDS[@]}"
do
    if [ ! -f "$CKPT" ]; then
        continue
    fi

    TAG="$(basename "$(dirname "$CKPT")")"

    echo
    echo "================================================================================"
    echo "CANDIDATE: $TAG"
    echo "================================================================================"

    if CUDA_VISIBLE_DEVICES=0 \
       python c5_ano_failure_decompose.py \
         --sample "$SAMPLE" \
         --gradient_npz "$GRAD" \
         --background64_npz "$BG64" \
         --mgno_ckpt "$CKPT" \
         --speed_mean 1488.39 \
         --speed_std 27.53 \
         --wave_scale 3.72290883e-02 \
         --chunk_size 8 \
         --device cuda:0 \
         --output "$OUT/${TAG}.json" \
         > "$OUT/${TAG}.log" 2>&1
    then
        printf "%s\t%s\t%s\n" \
          "$TAG" \
          "$CKPT" \
          "$OUT/${TAG}.npz" \
          >> "$OUT/candidates.tsv"
    else
        echo "[SKIP] failed: $CKPT"
    fi
done

python - <<'PY'
import numpy as np
from pathlib import Path

root = Path("c5_wavefield_runs/c5_7_fixed8_identify")
target = 0.604650

def cosine(a,b):
    a=np.asarray(a,dtype=np.float64).ravel()
    b=np.asarray(b,dtype=np.float64).ravel()
    return float(
        np.dot(a,b) /
        (np.linalg.norm(a)*np.linalg.norm(b)+1e-30)
    )

rows=[]

for line in (root/"candidates.tsv").read_text().splitlines():
    tag, ckpt, npz = line.split("\t")
    z=np.load(npz, allow_pickle=True)

    c=cosine(
        z["full_ano_gradient"],
        z["cbs_gradient"]
    )

    rows.append(
        (abs(c-target), c, tag, ckpt)
    )

rows.sort()

print("="*100)
print("C5.7 FIXED8 CHECKPOINT IDENTIFICATION")
print("="*100)

for err,c,tag,ckpt in rows:
    print(
        f"{tag:42s} "
        f"cos={c:+.6f} "
        f"|target diff|={err:.6e}"
    )

best=rows[0]

print()
print("SELECTED:")
print("  tag  =", best[2])
print("  ckpt =", best[3])
print("  cos  =", f"{best[1]:+.6f}")
print("  target= +0.604650")

Path(
    "c5_wavefield_runs/c5_7_fixed8_checkpoint.txt"
).write_text(
    best[3] + "\n"
)

if best[0] > 5e-3:
    raise RuntimeError(
        "No checkpoint reproduces Fixed8 cosine closely enough."
    )

print()
print("[PASS] Fixed8 checkpoint identified.")
PY
