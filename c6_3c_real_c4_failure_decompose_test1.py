import argparse
import json
from pathlib import Path

import numpy as np

from c5_paper_ano_val5 import (
    ano_gradient,
    cosine,
)

from c5_directional_fd_audit import (
    measurement_loss,
    normalize_direction,
    scalar,
)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--sample",
        required=True,
    )

    ap.add_argument(
        "--gradient",
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
        "--output",
        required=True,
    )

    args = ap.parse_args()

    z = np.load(
        args.sample,
        allow_pickle=True,
    )

    d = np.load(
        args.gradient,
        allow_pickle=True,
    )

    candidate = d[
        "candidate_speed"
    ].astype(np.float32)

    bridge_candidate = z[
        "candidate_480"
    ].astype(np.float32)

    if not np.array_equal(
        candidate,
        bridge_candidate,
    ):
        raise RuntimeError(
            "candidate mismatch"
        )

    true64 = d[
        "true64"
    ].astype(np.complex64)

    pred64 = d[
        "pred64"
    ].astype(np.complex64)

    exact_residual = d[
        "exact_residual"
    ].astype(np.complex64)

    neural_residual = d[
        "neural_residual"
    ].astype(np.complex64)

    source_positions = z[
        "source_positions"
    ].astype(np.int64)

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

    exact_tx = true64[
        source_positions
    ]

    neural_tx = pred64[
        source_positions
    ]

    # ------------------------------------------------------------
    # Four controlled gradients
    # ------------------------------------------------------------

    g_exact = ano_gradient(
        candidate=candidate,
        tx_waves=exact_tx,
        basis64=true64,
        residual=exact_residual,
        frequency=frequency,
    )

    # Only residual is neural.
    g_residual_only = ano_gradient(
        candidate=candidate,
        tx_waves=exact_tx,
        basis64=true64,
        residual=neural_residual,
        frequency=frequency,
    )

    # Only wavefield/basis is neural.
    g_basis_only = ano_gradient(
        candidate=candidate,
        tx_waves=neural_tx,
        basis64=pred64,
        residual=exact_residual,
        frequency=frequency,
    )

    # Fully neural ARSS.
    g_full = ano_gradient(
        candidate=candidate,
        tx_waves=neural_tx,
        basis64=pred64,
        residual=neural_residual,
        frequency=frequency,
    )

    methods = {
        "Exact": g_exact,
        "Neural-residual-only":
            g_residual_only,
        "Neural-basis-only":
            g_basis_only,
        "Full-ARSS":
            g_full,
    }

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

    print("=" * 115)
    print(
        "C6.3C REAL-C4 FAILURE DECOMPOSITION — TEST1"
    )
    print("=" * 115)

    print(
        f"J0  = {J0:.10e}"
    )

    print(
        f"RR0 = {rr0:.8f}"
    )

    results = {}

    for name, grad in methods.items():

        c = cosine(
            grad,
            g_exact,
        )

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

        drop = (
            (J0 - J1)
            /
            max(abs(J0), 1e-30)
        )

        descent = bool(
            J1 < J0
        )

        results[name] = {
            "cosine_to_exact":
                float(c),

            "J0":
                float(J0),

            "J1":
                float(J1),

            "relative_drop":
                float(drop),

            "rr0":
                float(rr0),

            "rr1":
                float(rr1),

            "descent":
                descent,
        }

        print()
        print("-" * 115)
        print(name)
        print("-" * 115)

        print(
            "cosine        =",
            f"{c:+.8f}",
        )

        print(
            "J1            =",
            f"{J1:.10e}",
        )

        print(
            "RR1           =",
            f"{rr1:.8f}",
        )

        print(
            "objective drop=",
            f"{100*drop:+.4f}%",
        )

        print(
            "descent       =",
            descent,
        )

    # ------------------------------------------------------------
    # Mechanism classification
    # ------------------------------------------------------------

    r_ok = results[
        "Neural-residual-only"
    ]["descent"]

    b_ok = results[
        "Neural-basis-only"
    ]["descent"]

    print()
    print("=" * 115)
    print("C6.3C DIAGNOSIS")
    print("=" * 115)

    if (not r_ok) and b_ok:

        diagnosis = (
            "RESIDUAL BOTTLENECK: "
            "neural residual destroys descent "
            "while neural wavefield/basis alone "
            "still preserves descent."
        )

    elif r_ok and (not b_ok):

        diagnosis = (
            "BASIS BOTTLENECK: "
            "neural wavefield/basis destroys "
            "descent while neural residual alone "
            "still preserves descent."
        )

    elif (not r_ok) and (not b_ok):

        diagnosis = (
            "COMPOUND FAILURE: "
            "both residual and wavefield/basis "
            "errors independently destroy descent."
        )

    else:

        diagnosis = (
            "COMPOUND INTERACTION: "
            "each isolated neural component "
            "preserves descent, but their combined "
            "error causes the full ARSS failure."
        )

    print(diagnosis)

    payload = {
        "sample": 1,
        "step": args.step,
        "J0": float(J0),
        "rr0": float(rr0),
        "results": results,
        "diagnosis": diagnosis,
    }

    out = Path(args.output)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("saved =", out)


if __name__ == "__main__":
    main()
