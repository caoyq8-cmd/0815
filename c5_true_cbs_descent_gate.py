import argparse
import csv
import json
from pathlib import Path

import numpy as np

from c5_directional_fd_audit import (
    measurement_loss,
    normalize_direction,
    rms,
    scalar,
)

from c5_paper_ano_val5 import (
    ano_gradient,
    cosine,
)


def load_neural_gradient(
    sample,
    decompose_path,
):
    """
    Reconstruct fully-neural ANO gradient from an existing
    C5.4d/C5.5 decomposition NPZ.

    No MgNO inference is performed here.
    """

    d = np.load(
        decompose_path,
        allow_pickle=True,
    )

    candidate = d[
        "candidate_speed"
    ].astype(np.float32)

    pred64 = d[
        "pred64"
    ].astype(np.complex64)

    g_cbs = d[
        "cbs_gradient"
    ].astype(np.float64)

    source_positions = sample[
        "source_positions"
    ].astype(np.int64)

    rec_indices = sample[
        "rec_indices"
    ].astype(np.int64)

    dobs = sample[
        "dobs_complex"
    ].astype(np.complex64)

    frequency = float(
        scalar(
            sample["frequency"]
        )
    )

    neural_tx = pred64[
        source_positions
    ]

    rr = rec_indices[:, 0]
    cc = rec_indices[:, 1]

    neural_measurement = neural_tx[
        :,
        rr,
        cc,
    ]

    neural_residual = (
        neural_measurement
        - dobs
    )

    g_ano = ano_gradient(
        candidate=candidate,
        tx_waves=neural_tx,
        basis64=pred64,
        residual=neural_residual,
        frequency=frequency,
    )

    return (
        candidate,
        g_cbs,
        g_ano,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--sample_dir",
        required=True,
    )

    ap.add_argument(
        "--fixed_root",
        required=True,
    )

    ap.add_argument(
        "--uniform_root",
        required=True,
    )

    ap.add_argument(
        "--strat_root",
        required=True,
    )

    ap.add_argument(
        "--steps",
        default="0.125,0.25,0.5,1.0",
        help=(
            "RMS speed perturbation sizes "
            "in m/s."
        ),
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

    sample_dir = Path(
        args.sample_dir
    )

    fixed_root = Path(
        args.fixed_root
    )

    uniform_root = Path(
        args.uniform_root
    )

    strat_root = Path(
        args.strat_root
    )

    out = Path(
        args.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    steps = [
        float(x)
        for x in args.steps.split(",")
        if x.strip()
    ]

    if not steps:
        raise ValueError(
            "Empty step grid."
        )

    all_rows = []
    sample_summary = []

    print("=" * 120)
    print(
        "C5.6 TRUE-CBS ONE-STEP DESCENT GATE"
    )
    print("=" * 120)

    print(
        "RMS step grid =",
        steps,
        "m/s",
    )

    for i in range(1, 6):

        print()
        print("#" * 120)
        print(
            f"TEST {i}"
        )
        print("#" * 120)

        sample_path = (
            sample_dir
            / f"test_{i}.npz"
        )

        # ------------------------------------------------------------------
        # These are the paths created by the previous experiments.
        # ------------------------------------------------------------------

        fixed_path = (
            fixed_root
            / f"c5_4d_test{i}_rho08.npz"
        )

        uniform_path = (
            uniform_root
            / f"decompose_test{i}.npz"
        )

        strat_path = (
            strat_root
            / f"decompose_test{i}.npz"
        )

        for p in [
            sample_path,
            fixed_path,
            uniform_path,
            strat_path,
        ]:
            if not p.exists():
                raise FileNotFoundError(
                    p
                )

        z = np.load(
            sample_path,
            allow_pickle=True,
        )

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
            scalar(
                z["frequency"]
            )
        )

        cbs_iters = int(
            scalar(
                z["cbs_iters"]
            )
        )

        boundary_width = int(
            scalar(
                z["boundary_width"]
            )
        )

        boundary_strength = float(
            scalar(
                z["boundary_strength"]
            )
        )

        boundary_type = str(
            scalar(
                z["boundary_type"]
            )
        )

        (
            candidate_fixed,
            g_cbs_fixed,
            g_fixed,
        ) = load_neural_gradient(
            z,
            fixed_path,
        )

        (
            candidate_uniform,
            g_cbs_uniform,
            g_uniform,
        ) = load_neural_gradient(
            z,
            uniform_path,
        )

        (
            candidate_strat,
            g_cbs_strat,
            g_strat,
        ) = load_neural_gradient(
            z,
            strat_path,
        )

        # ------------------------------------------------------------------
        # Candidate and exact-gradient consistency.
        # All three methods must be evaluated at exactly the same X.
        # ------------------------------------------------------------------

        max_candidate_diff = max(
            float(
                np.max(
                    np.abs(
                        candidate_fixed
                        - candidate_uniform
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        candidate_fixed
                        - candidate_strat
                    )
                )
            ),
        )

        max_gradient_diff = max(
            float(
                np.max(
                    np.abs(
                        g_cbs_fixed
                        - g_cbs_uniform
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        g_cbs_fixed
                        - g_cbs_strat
                    )
                )
            ),
        )

        print(
            "candidate consistency max abs =",
            f"{max_candidate_diff:.3e}",
        )

        print(
            "CBS-gradient consistency max abs =",
            f"{max_gradient_diff:.3e}",
        )

        if max_candidate_diff > 1e-5:
            raise RuntimeError(
                "Candidate mismatch across "
                "source-sampling experiments."
            )

        if max_gradient_diff > 1e-8:
            raise RuntimeError(
                "CBS gradient mismatch across "
                "source-sampling experiments."
            )

        candidate = candidate_fixed
        g_cbs = g_cbs_fixed

        # ------------------------------------------------------------------
        # Exact physical baseline.
        # ------------------------------------------------------------------

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

        print()
        print(
            "J0               =",
            f"{J0:.10e}",
        )

        print(
            "measurement RRMSE=",
            f"{rr0:.8f}",
        )

        gradients = {
            "Exact-CBS":
                g_cbs,

            "Fixed8":
                g_fixed,

            "Uniform64":
                g_uniform,

            "Strat4+4":
                g_strat,
        }

        print()
        print(
            "Gradient cosine with Exact CBS:"
        )

        for name, g in gradients.items():
            print(
                f"  {name:12s} "
                f"{cosine(g, g_cbs):+.6f}"
            )

        method_summary = {}

        # ------------------------------------------------------------------
        # True CBS one-step objective test.
        #
        # normalize_direction() => RMS(direction)=1.
        #
        # Therefore:
        #   candidate - eps * direction
        #
        # changes the speed map by eps m/s RMS.
        # ------------------------------------------------------------------

        for name, grad in gradients.items():

            direction = normalize_direction(
                grad
            )

            print()
            print("-" * 120)

            print(
                f"DIRECTION: {name}"
            )

            print(
                "direction RMS =",
                f"{rms(direction):.8f}",
            )

            best_J = J0
            best_step = 0.0
            best_rr = rr0
            descent_count = 0

            for eps in steps:

                trial = (
                    candidate
                    - eps * direction
                ).astype(
                    np.float32
                )

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
                    / max(
                        abs(J0),
                        1e-30,
                    )
                )

                rel_drop = -rel_change

                is_descent = bool(
                    J1 < J0
                )

                if is_descent:
                    descent_count += 1

                if J1 < best_J:
                    best_J = J1
                    best_step = eps
                    best_rr = rr1

                row = {
                    "sample":
                        i,

                    "method":
                        name,

                    "step_rms_mps":
                        eps,

                    "cosine_to_cbs":
                        cosine(
                            grad,
                            g_cbs,
                        ),

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
                        is_descent,

                    "trial_min":
                        float(
                            trial.min()
                        ),

                    "trial_max":
                        float(
                            trial.max()
                        ),
                }

                all_rows.append(
                    row
                )

                print(
                    f"step={eps:6.3f} m/s "
                    f"J={J1:.10e} "
                    f"dJ/J={rel_change:+.6e} "
                    f"RR={rr1:.8f} "
                    f"descent={is_descent}"
                )

            best_drop = (
                (J0 - best_J)
                / max(
                    abs(J0),
                    1e-30,
                )
            )

            any_descent = bool(
                best_J < J0
            )

            method_summary[
                name
            ] = {
                "cosine":
                    cosine(
                        grad,
                        g_cbs,
                    ),

                "any_descent":
                    any_descent,

                "descent_steps":
                    descent_count,

                "best_step":
                    best_step,

                "best_J":
                    best_J,

                "best_relative_drop":
                    best_drop,

                "best_rr":
                    best_rr,
            }

            print(
                f"BEST {name:12s} | "
                f"step={best_step:.3f} "
                f"drop={best_drop:+.6e} "
                f"descent={any_descent}"
            )

        sample_summary.append({
            "sample":
                i,

            "J0":
                J0,

            "rr0":
                rr0,

            "methods":
                method_summary,
        })

    # ======================================================================
    # Aggregate
    # ======================================================================

    print()
    print("=" * 120)
    print(
        "C5.6 AGGREGATE TRUE-CBS DESCENT"
    )
    print("=" * 120)

    methods = [
        "Exact-CBS",
        "Fixed8",
        "Uniform64",
        "Strat4+4",
    ]

    aggregate = {}

    for name in methods:

        descents = [
            s["methods"][name][
                "any_descent"
            ]
            for s in sample_summary
        ]

        drops = np.asarray(
            [
                s["methods"][name][
                    "best_relative_drop"
                ]
                for s in sample_summary
            ],
            dtype=np.float64,
        )

        cosines = np.asarray(
            [
                s["methods"][name][
                    "cosine"
                ]
                for s in sample_summary
            ],
            dtype=np.float64,
        )

        n_descent = int(
            np.sum(
                descents
            )
        )

        aggregate[name] = {
            "descent_cases":
                n_descent,

            "total_cases":
                len(
                    sample_summary
                ),

            "cosine_mean":
                float(
                    cosines.mean()
                ),

            "cosine_min":
                float(
                    cosines.min()
                ),

            "best_drop_mean":
                float(
                    drops.mean()
                ),

            "best_drop_median":
                float(
                    np.median(
                        drops
                    )
                ),

            "best_drop_min":
                float(
                    drops.min()
                ),

            "best_drop_max":
                float(
                    drops.max()
                ),
        }

        print(
            f"{name:12s} | "
            f"descent={n_descent}/"
            f"{len(sample_summary)} | "
            f"cos={cosines.mean():+.6f} | "
            f"best_drop="
            f"{drops.mean():+.6e}"
            f" ± {drops.std():.6e}"
        )

    # ======================================================================
    # Decision gate
    # ======================================================================

    strat_pass = (
        aggregate[
            "Strat4+4"
        ][
            "descent_cases"
        ]
    )

    print()
    print("=" * 120)
    print(
        "C5.6 DECISION"
    )
    print("=" * 120)

    if strat_pass == 5:

        print(
            "[STRONG PASS] Strat4+4 yields "
            "a true-CBS descent step on 5/5 "
            "VAL samples."
        )

    elif strat_pass >= 4:

        print(
            "[PASS] Strat4+4 yields "
            "true-CBS descent on >=4/5 "
            "VAL samples."
        )

    else:

        print(
            "[FAIL] Gradient cosine improvement "
            "does not yet translate into stable "
            "true-CBS objective descent."
        )

    # ======================================================================
    # Save
    # ======================================================================

    csv_path = (
        out
        / "all_steps.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                list(
                    all_rows[0].keys()
                ),
        )

        writer.writeheader()
        writer.writerows(
            all_rows
        )

    json_path = (
        out
        / "summary.json"
    )

    json_path.write_text(
        json.dumps(
            {
                "steps":
                    steps,

                "samples":
                    sample_summary,

                "aggregate":
                    aggregate,
            },
            indent=2,
        )
    )

    print()
    print(
        "saved =",
        csv_path,
    )

    print(
        "saved =",
        json_path,
    )


if __name__ == "__main__":
    main()
