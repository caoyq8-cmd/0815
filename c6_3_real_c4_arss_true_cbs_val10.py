import argparse
import csv
import json
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

    ap.add_argument(
        "--sample_dir",
        required=True,
    )

    ap.add_argument(
        "--decompose_root",
        required=True,
    )

    ap.add_argument(
        "--step",
        type=float,
        default=1.5,
    )

    ap.add_argument(
        "--device",
        default="cuda:0",
    )

    ap.add_argument(
        "--output_dir",
        required=True,
    )

    args = ap.parse_args()

    sample_dir = Path(args.sample_dir)
    decomp_root = Path(args.decompose_root)
    out = Path(args.output_dir)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print("=" * 115)
    print("C5.7 FIXED 1.5 m/s TRUE-CBS ROBUSTNESS — TEST6-10")
    print("=" * 115)

    print(
        "frozen RMS step =",
        args.step,
        "m/s"
    )

    for i in range(6, 11):

        print()
        print("#" * 115)
        print(f"TEST {i}")
        print("#" * 115)

        sample_path = (
            sample_dir
            / f"test_{i}.npz"
        )

        decomp_path = (
            decomp_root
            / f"decompose_test{i}.npz"
        )

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

        g_exact = d[
            "cbs_gradient"
        ].astype(np.float64)

        g_strat = d[
            "full_ano_gradient"
        ].astype(np.float64)

        dobs = z[
            "dobs_complex"
        ].astype(np.complex64)

        src_indices = z[
            "src_indices"
        ].astype(np.int64)

        rec_indices = z[
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

        # ------------------------------------------------------------
        # True physical baseline
        # ------------------------------------------------------------

        J0, rr0 = measurement_loss(
            candidate,
            dobs,
            src_indices,
            rec_indices,
            frequency,
            cbs_iters,
            boundary_width,
            boundary_strength,
            boundary_type,
            args.device,
        )

        cos = cosine(
            g_strat,
            g_exact,
        )

        print(
            f"cos(Strat,CBS) = {cos:+.6f}"
        )

        print(
            f"J0             = {J0:.10e}"
        )

        print(
            f"RR0            = {rr0:.8f}"
        )

        methods = {
            "Exact-CBS":
                g_exact,

            "Strat4+4":
                g_strat,
        }

        for name, grad in methods.items():

            direction = normalize_direction(
                grad
            )

            trial = (
                candidate
                - args.step * direction
            ).astype(np.float32)

            J1, rr1 = measurement_loss(
                trial,
                dobs,
                src_indices,
                rec_indices,
                frequency,
                cbs_iters,
                boundary_width,
                boundary_strength,
                boundary_type,
                args.device,
            )

            rel_change = (
                (J1 - J0)
                /
                max(abs(J0), 1e-30)
            )

            rel_drop = -rel_change

            descent = bool(
                J1 < J0
            )

            method_cos = cosine(
                grad,
                g_exact,
            )

            rows.append({
                "sample":
                    i,

                "method":
                    name,

                "step_rms_mps":
                    args.step,

                "cosine_to_cbs":
                    method_cos,

                "J0":
                    J0,

                "J1":
                    J1,

                "relative_change":
                    rel_change,

                "relative_drop":
                    rel_drop,

                "rr0":
                    rr0,

                "rr1":
                    rr1,

                "descent":
                    descent,

                "trial_min":
                    float(trial.min()),

                "trial_max":
                    float(trial.max()),
            })

            print(
                f"{name:12s} | "
                f"J1={J1:.10e} | "
                f"drop={100*rel_drop:+8.3f}% | "
                f"RR={rr1:.8f} | "
                f"descent={descent}"
            )

    # ================================================================
    # Aggregate new 5 cases
    # ================================================================

    print()
    print("=" * 115)
    print("C5.7 TEST6-10 AGGREGATE")
    print("=" * 115)

    aggregate = {}

    for method in [
        "Exact-CBS",
        "Strat4+4",
    ]:

        sub = [
            r
            for r in rows
            if r["method"] == method
        ]

        drops = np.asarray(
            [
                r["relative_drop"]
                for r in sub
            ],
            dtype=np.float64,
        )

        cosines = np.asarray(
            [
                r["cosine_to_cbs"]
                for r in sub
            ],
            dtype=np.float64,
        )

        descent = np.asarray(
            [
                r["descent"]
                for r in sub
            ],
            dtype=bool,
        )

        aggregate[method] = {
            "descent_cases":
                int(descent.sum()),

            "total_cases":
                len(sub),

            "mean_cosine":
                float(cosines.mean()),

            "min_cosine":
                float(cosines.min()),

            "mean_drop":
                float(drops.mean()),

            "median_drop":
                float(np.median(drops)),

            "min_drop":
                float(drops.min()),

            "max_drop":
                float(drops.max()),
        }

        print(
            f"{method:12s} | "
            f"descent={descent.sum()}/{len(sub)} | "
            f"cos={cosines.mean():+.6f} | "
            f"drop={100*drops.mean():+.3f}% "
            f"± {100*drops.std():.3f}% | "
            f"worst={100*drops.min():+.3f}%"
        )

    # ================================================================
    # Decision for new held-out 5
    # ================================================================

    n = aggregate[
        "Strat4+4"
    ][
        "descent_cases"
    ]

    print()
    print("=" * 115)
    print("C5.7 NEW-CASE DECISION")
    print("=" * 115)

    if n == 5:

        print(
            "[STRONG PASS] "
            "Frozen Strat4+4 step gives "
            "true-CBS descent on all 5 new cases."
        )

    elif n >= 4:

        print(
            "[PASS] "
            "Frozen Strat4+4 step gives "
            "true-CBS descent on >=4/5 new cases."
        )

    else:

        print(
            "[FAIL] "
            "Frozen 1.5 m/s step does not "
            "generalize robustly enough."
        )

    csv_path = (
        out / "test6_10_fixed15.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    json_path = (
        out / "summary_test6_10.json"
    )

    json_path.write_text(
        json.dumps(
            {
                "frozen_step":
                    args.step,

                "aggregate":
                    aggregate,
            },
            indent=2,
        )
    )

    print()
    print("saved =", csv_path)
    print("saved =", json_path)


if __name__ == "__main__":
    main()
