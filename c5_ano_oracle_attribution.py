import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from c5_paper_ano_val5 import (
    ano_gradient,
    metrics,
)


COMPONENTS = [
    "R",  # residual
    "T",  # transmitter wavefield
    "S",  # seen8 receiver-as-source basis
    "U",  # unseen56 receiver-as-source basis
]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--sample",
        required=True,
    )

    ap.add_argument(
        "--decompose_npz",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    # ================================================================
    # Load sample
    # ================================================================

    z = np.load(
        args.sample,
        allow_pickle=True,
    )

    dobs = z[
        "dobs_complex"
    ].astype(
        np.complex64
    )

    rec_indices = z[
        "rec_indices"
    ].astype(
        np.int64
    )

    source_positions = z[
        "source_positions"
    ].astype(
        np.int64
    )

    frequency = float(
        np.asarray(
            z["frequency"]
        ).reshape(-1)[0]
    )

    # ================================================================
    # Load C5.4d exact/neural 64-source data
    # ================================================================

    d = np.load(
        args.decompose_npz,
        allow_pickle=True,
    )

    candidate = d[
        "candidate_speed"
    ].astype(
        np.float32
    )

    true64 = d[
        "true64"
    ].astype(
        np.complex64
    )

    pred64 = d[
        "pred64"
    ].astype(
        np.complex64
    )

    g_cbs = d[
        "cbs_gradient"
    ].astype(
        np.float64
    )

    exact_tx = true64[
        source_positions
    ]

    neural_tx = pred64[
        source_positions
    ]

    rr = rec_indices[:, 0]
    cc = rec_indices[:, 1]

    exact_measurement = exact_tx[
        :,
        rr,
        cc,
    ]

    neural_measurement = neural_tx[
        :,
        rr,
        cc,
    ]

    exact_residual = (
        exact_measurement
        - dobs
    )

    neural_residual = (
        neural_measurement
        - dobs
    )

    seen_mask = np.zeros(
        64,
        dtype=bool,
    )

    seen_mask[
        source_positions
    ] = True

    unseen_mask = ~seen_mask

    # ================================================================
    # Evaluate arbitrary repair set
    # ================================================================

    def evaluate(repaired):
        """
        repaired is a set containing any of:
          R, T, S, U

        Repaired means that component is replaced
        by exact CBS quantity.
        """

        residual = (
            exact_residual
            if "R" in repaired
            else neural_residual
        )

        tx = (
            exact_tx
            if "T" in repaired
            else neural_tx
        )

        basis = pred64.copy()

        if "S" in repaired:
            basis[
                seen_mask
            ] = true64[
                seen_mask
            ]

        if "U" in repaired:
            basis[
                unseen_mask
            ] = true64[
                unseen_mask
            ]

        grad = ano_gradient(
            candidate=
                candidate,
            tx_waves=
                tx,
            basis64=
                basis,
            residual=
                residual,
            frequency=
                frequency,
        )

        return metrics(
            grad,
            g_cbs,
        )

    # ================================================================
    # All 16 combinations
    # ================================================================

    results = {}

    print("=" * 125)
    print("C5.4f ANO 4-FACTOR ORACLE REPAIR ATTRIBUTION")
    print("=" * 125)

    print(
        "Components:"
    )

    print(
        "  R = residual"
    )

    print(
        "  T = transmitter wavefield"
    )

    print(
        "  S = seen8 adjoint basis"
    )

    print(
        "  U = unseen56 adjoint basis"
    )

    print()

    print(
        f"{'exact/repaired components':30s}"
        f"{'raw':>12s}"
        f"{'crop32':>12s}"
        f"{'smooth5':>12s}"
        f"{'smooth9':>12s}"
    )

    for n in range(5):

        for combo in itertools.combinations(
            COMPONENTS,
            n,
        ):
            repaired = frozenset(
                combo
            )

            key = "".join(
                sorted(
                    repaired
                )
            )

            if key == "":
                key = "NONE"

            m = evaluate(
                repaired
            )

            results[key] = m

            print(
                f"{key:30s}"
                f"{m['raw']:12.6f}"
                f"{m['crop32']:12.6f}"
                f"{m['smooth5']:12.6f}"
                f"{m['smooth9']:12.6f}"
            )

    baseline = results[
        "NONE"
    ][
        "raw"
    ]

    exact = results[
        "RSTU"
    ][
        "raw"
    ]

    # ================================================================
    # Single-component repair gains
    # ================================================================

    print()
    print("=" * 125)
    print("SINGLE-COMPONENT ORACLE REPAIR GAIN")
    print("=" * 125)

    single_gain = {}

    for c in COMPONENTS:

        gain = (
            results[c]["raw"]
            - baseline
        )

        single_gain[c] = gain

        print(
            f"repair {c} only | "
            f"raw={results[c]['raw']:+.6f} | "
            f"gain={gain:+.6f}"
        )

    # ================================================================
    # Shapley attribution
    #
    # Utility = raw gradient cosine.
    # This includes nonlinear interaction effects.
    # ================================================================

    def utility(repaired):
        key = "".join(
            sorted(
                repaired
            )
        )

        if key == "":
            key = "NONE"

        return results[
            key
        ][
            "raw"
        ]

    N = len(
        COMPONENTS
    )

    shapley = {}

    for c in COMPONENTS:

        others = [
            x
            for x in COMPONENTS
            if x != c
        ]

        phi = 0.0

        for k in range(
            len(others) + 1
        ):

            for subset_tuple in itertools.combinations(
                others,
                k,
            ):
                subset = frozenset(
                    subset_tuple
                )

                weight = (
                    math.factorial(
                        len(subset)
                    )
                    *
                    math.factorial(
                        N
                        - len(subset)
                        - 1
                    )
                    /
                    math.factorial(N)
                )

                marginal = (
                    utility(
                        subset
                        | {c}
                    )
                    -
                    utility(
                        subset
                    )
                )

                phi += (
                    weight
                    * marginal
                )

        shapley[c] = phi

    print()
    print("=" * 125)
    print("SHAPLEY ATTRIBUTION OF RAW-COSINE RECOVERY")
    print("=" * 125)

    labels = {
        "R":
            "Residual",

        "T":
            "TX wavefield",

        "S":
            "Seen8 basis",

        "U":
            "Unseen56 basis",
    }

    total_gain = (
        exact
        - baseline
    )

    for c in COMPONENTS:

        frac = (
            shapley[c]
            / total_gain
            if abs(total_gain) > 1e-30
            else float("nan")
        )

        print(
            f"{labels[c]:18s} | "
            f"Shapley={shapley[c]:+.6f} | "
            f"share={100.0*frac:+.2f}%"
        )

    print()
    print(
        "baseline full neural raw =",
        f"{baseline:+.6f}",
    )

    print(
        "exact everything raw     =",
        f"{exact:+.6f}",
    )

    print(
        "recoverable raw gain      =",
        f"{total_gain:+.6f}",
    )

    # ================================================================
    # Recommended interpretation
    # ================================================================

    ranked = sorted(
        COMPONENTS,
        key=lambda c:
            shapley[c],
        reverse=True,
    )

    print()
    print("=" * 125)
    print("ATTRIBUTION RANKING")
    print("=" * 125)

    for i, c in enumerate(
        ranked,
        start=1,
    ):
        print(
            f"{i}. "
            f"{labels[c]} "
            f"({shapley[c]:+.6f})"
        )

    out = Path(
        args.output
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "all_combinations":
            results,

        "baseline_raw":
            baseline,

        "exact_raw":
            exact,

        "single_gain":
            single_gain,

        "shapley":
            shapley,

        "ranking":
            ranked,
    }

    out.write_text(
        json.dumps(
            payload,
            indent=2,
        )
    )

    print()
    print(
        "saved =",
        out,
    )


if __name__ == "__main__":
    main()
