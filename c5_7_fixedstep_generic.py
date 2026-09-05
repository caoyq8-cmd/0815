import argparse
import csv
from pathlib import Path

import numpy as np

from c5_directional_fd_audit import (
    measurement_loss,
    normalize_direction,
    scalar,
)

from c5_paper_ano_val5 import cosine


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--sample_dir", required=True)
    ap.add_argument("--decompose_root", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--first", type=int, default=6)
    ap.add_argument("--last", type=int, default=10)
    ap.add_argument("--step", type=float, default=1.5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)

    args = ap.parse_args()

    sample_root = Path(args.sample_dir)
    decomp_root = Path(args.decompose_root)

    rows = []

    print("=" * 115)
    print(f"C5.7 FIXED-STEP TRUE-CBS: {args.label}")
    print("=" * 115)
    print(f"frozen RMS step = {args.step:.3f} m/s")

    for i in range(args.first, args.last + 1):

        sample_path = sample_root / f"test_{i}.npz"
        decomp_path = decomp_root / f"decompose_test{i}.npz"

        if not sample_path.exists():
            raise FileNotFoundError(sample_path)

        if not decomp_path.exists():
            raise FileNotFoundError(decomp_path)

        z = np.load(
            sample_path,
            allow_pickle=True,
        )

        d = np.load(
            decomp_path,
            allow_pickle=True,
        )

        candidate = d[
            "candidate_speed"
        ].astype(np.float32)

        g = d[
            "full_ano_gradient"
        ].astype(np.float64)

        ref = d[
            "cbs_gradient"
        ].astype(np.float64)

        dobs = z[
            "dobs_complex"
        ].astype(np.complex64)

        src = z[
            "src_indices"
        ].astype(np.int64)

        rec = z[
            "rec_indices"
        ].astype(np.int64)

        frequency = float(
            scalar(z["frequency"])
        )

        cbs_iters = int(
            scalar(z["cbs_iters"])
        )

        boundary_width = int(
            scalar(z["boundary_width"])
        )

        boundary_strength = float(
            scalar(z["boundary_strength"])
        )

        boundary_type = str(
            scalar(z["boundary_type"])
        )

        # --------------------------------------------------
        # Baseline true CBS objective
        # --------------------------------------------------

        J0, rr0 = measurement_loss(
            candidate,
            dobs,
            src,
            rec,
            frequency,
            cbs_iters,
            boundary_width,
            boundary_strength,
            boundary_type,
            args.device,
        )

        # RMS-normalized frozen step
        direction = normalize_direction(g)

        trial = (
            candidate
            - args.step * direction
        ).astype(np.float32)

        J1, rr1 = measurement_loss(
            trial,
            dobs,
            src,
            rec,
            frequency,
            cbs_iters,
            boundary_width,
            boundary_strength,
            boundary_type,
            args.device,
        )

        c = cosine(
            g,
            ref,
        )

        relative_drop = (
            (J0 - J1)
            / max(abs(J0), 1e-30)
        )

        descent = bool(
            J1 < J0
        )

        row = {
            "sample": i,
            "method": args.label,
            "step_rms_mps": args.step,
            "cosine_to_cbs": c,
            "J0": J0,
            "J1": J1,
            "relative_drop": relative_drop,
            "rr0": rr0,
            "rr1": rr1,
            "descent": descent,
        }

        rows.append(row)

        print(
            f"test{i:02d} "
            f"cos={c:+.6f} "
            f"J0={J0:.8e} "
            f"J1={J1:.8e} "
            f"drop={100*relative_drop:+8.3f}% "
            f"descent={descent}"
        )

    cosines = np.asarray(
        [x["cosine_to_cbs"] for x in rows],
        dtype=np.float64,
    )

    drops = np.asarray(
        [x["relative_drop"] for x in rows],
        dtype=np.float64,
    )

    descents = np.asarray(
        [x["descent"] for x in rows],
        dtype=bool,
    )

    print()
    print("=" * 115)
    print("AGGREGATE")
    print("=" * 115)

    print(
        f"descent     = "
        f"{descents.sum()}/{len(descents)}"
    )

    print(
        f"cos mean    = "
        f"{cosines.mean():+.6f}"
    )

    print(
        f"cos median  = "
        f"{np.median(cosines):+.6f}"
    )

    print(
        f"cos min     = "
        f"{cosines.min():+.6f}"
    )

    print(
        f"drop mean   = "
        f"{100*drops.mean():+.3f}%"
    )

    print(
        f"drop std    = "
        f"{100*drops.std():.3f}%"
    )

    print(
        f"drop median = "
        f"{100*np.median(drops):+.3f}%"
    )

    print(
        f"drop worst  = "
        f"{100*drops.min():+.3f}%"
    )

    out = Path(args.output)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        out,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("saved =", out)


if __name__ == "__main__":
    main()
