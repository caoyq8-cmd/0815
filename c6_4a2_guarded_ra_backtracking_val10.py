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


def mse(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.mean((a - b) ** 2))


def stat(x):
    x = np.asarray(x, np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--bridge_dir",
        required=True,
    )

    ap.add_argument(
        "--gradient_dir",
        required=True,
    )

    ap.add_argument(
        "--output_dir",
        required=True,
    )

    ap.add_argument(
        "--start",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--end",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--initial_step",
        type=float,
        default=1.5,
    )

    ap.add_argument(
        "--backtrack_factor",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--max_backtracks",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--device",
        default="cuda:0",
    )

    args = ap.parse_args()

    bridge = Path(args.bridge_dir)
    grad_root = Path(args.gradient_dir)
    out = Path(args.output_dir)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    attempts = []

    print("=" * 120)
    print("C6.4A2 GUARDED RA-ARSS — MONOTONE TRUE8 BACKTRACKING")
    print("=" * 120)

    print("initial step     =", args.initial_step)
    print("backtrack factor =", args.backtrack_factor)
    print("max backtracks   =", args.max_backtracks)

    for i in range(
        args.start,
        args.end + 1,
    ):

        z = np.load(
            bridge / f"test_{i}.npz",
            allow_pickle=True,
        )

        g = np.load(
            grad_root / f"test_{i}_gradients.npz",
            allow_pickle=True,
        )

        candidate = z[
            "candidate_480"
        ].astype(np.float32)

        cached = g[
            "candidate_speed"
        ].astype(np.float32)

        if not np.array_equal(
            candidate,
            cached,
        ):
            raise RuntimeError(
                f"test_{i}: candidate mismatch"
            )

        grad = g[
            "ra_arss_gradient"
        ].astype(np.float64)

        direction = normalize_direction(
            grad
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

        target = z[
            "target_480"
        ].astype(np.float32)

        # Current objective.
        # In the integrated method this comes from
        # the same true8 solve used to construct
        # the exact residual.
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

        mse0 = mse(
            candidate,
            target,
        )

        accepted = False
        accepted_step = 0.0
        accepted_trial = candidate.copy()
        accepted_J = J0
        accepted_rr = rr0
        num_trials = 0

        print()
        print("#" * 120)
        print(f"TEST {i}")
        print("#" * 120)

        print(
            f"J0={J0:.10e} "
            f"RR0={rr0:.8f}"
        )

        for k in range(
            args.max_backtracks + 1
        ):

            step = (
                args.initial_step
                *
                args.backtrack_factor ** k
            )

            trial = (
                candidate
                - step * direction
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

            num_trials += 1

            drop = (
                (J0 - J1)
                /
                max(abs(J0), 1e-30)
            )

            ok = bool(
                J1 < J0
            )

            attempts.append({
                "sample": i,
                "attempt": k,
                "step_mps": step,
                "J0": J0,
                "J1": J1,
                "relative_drop": drop,
                "rr0": rr0,
                "rr1": rr1,
                "accepted": ok,
            })

            print(
                f"  step={step:.6f} | "
                f"J={J1:.10e} | "
                f"drop={100*drop:+.4f}% | "
                f"accept={ok}"
            )

            if ok:

                accepted = True
                accepted_step = step
                accepted_trial = trial
                accepted_J = J1
                accepted_rr = rr1

                break

        final_drop = (
            (J0 - accepted_J)
            /
            max(abs(J0), 1e-30)
        )

        mse1 = mse(
            accepted_trial,
            target,
        )

        mse_gain = (
            (mse0 - mse1)
            /
            max(mse0, 1e-30)
        )

        # One initial true8 residual evaluation,
        # plus one true8 evaluation per line-search trial.
        estimated_true_sources = (
            8
            *
            (1 + num_trials)
        )

        row = {
            "sample": i,
            "accepted": accepted,
            "accepted_step_mps":
                accepted_step,
            "num_trial_evals":
                num_trials,
            "estimated_true_sources":
                estimated_true_sources,
            "J0": J0,
            "J_final": accepted_J,
            "relative_drop":
                final_drop,
            "rr0": rr0,
            "rr_final": accepted_rr,
            "mse0": mse0,
            "mse_final": mse1,
            "mse_gain": mse_gain,
        }

        rows.append(row)

        print(
            "FINAL | "
            f"accepted={accepted} | "
            f"step={accepted_step:.6f} | "
            f"drop={100*final_drop:+.4f}% | "
            f"true-source-equiv="
            f"{estimated_true_sources}"
        )

    # ---------------------------------------------------------
    # Save attempt history
    # ---------------------------------------------------------

    with (
        out / "backtracking_attempts.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                attempts[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(attempts)

    with (
        out / "summary.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(rows)

    drops = [
        r["relative_drop"]
        for r in rows
    ]

    strict_descent = sum(
        r["accepted"]
        for r in rows
    )

    source_counts = [
        r["estimated_true_sources"]
        for r in rows
    ]

    result = {
        "experiment":
            "C6.4A2 Guarded RA-ARSS",

        "development_set_note":
            (
                "This VAL10 was already inspected "
                "before designing backtracking and "
                "must be treated as development data."
            ),

        "initial_step":
            args.initial_step,

        "backtrack_factor":
            args.backtrack_factor,

        "max_backtracks":
            args.max_backtracks,

        "strict_descent_count":
            int(strict_descent),

        "nonincrease_count":
            len(rows),

        "relative_drop":
            stat(drops),

        "estimated_true_source_count":
            stat(source_counts),
    }

    (
        out / "summary.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("C6.4A2 SUMMARY")
    print("=" * 120)

    print(
        "strict descent =",
        f"{strict_descent}/{len(rows)}"
    )

    print(
        "non-increase   =",
        f"{len(rows)}/{len(rows)}"
    )

    print(
        "mean drop      =",
        f"{100*np.mean(drops):+.4f}%"
    )

    print(
        "worst drop     =",
        f"{100*np.min(drops):+.4f}%"
    )

    print(
        "mean true-source equivalent =",
        f"{np.mean(source_counts):.2f}"
    )


if __name__ == "__main__":
    main()
