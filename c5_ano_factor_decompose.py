import argparse
from pathlib import Path

import numpy as np

from c5_paper_ano_val5 import (
    ano_gradient,
    metrics,
    rrmse,
)


def show(name, g, ref):
    m = metrics(g, ref)

    print(
        f"{name:34s}"
        f"{m['raw']:12.6f}"
        f"{m['crop32']:12.6f}"
        f"{m['smooth5']:12.6f}"
        f"{m['smooth9']:12.6f}"
    )

    return m


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--sample",
        required=True,
    )

    ap.add_argument(
        "--gradient_npz",
        required=True,
    )

    ap.add_argument(
        "--decompose_npz",
        required=True,
    )

    args = ap.parse_args()

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

    gz = np.load(
        args.gradient_npz,
        allow_pickle=True,
    )

    candidate = gz[
        "candidate_speed"
    ].astype(
        np.float32
    )

    g_cbs = gz[
        "cbs_gradient"
    ].astype(
        np.float64
    )

    dz = np.load(
        args.decompose_npz,
        allow_pickle=True,
    )

    true64 = dz[
        "true64"
    ].astype(
        np.complex64
    )

    pred64 = dz[
        "pred64"
    ].astype(
        np.complex64
    )

    exact_tx = true64[
        source_positions
    ]

    neural_tx = pred64[
        source_positions
    ]

    rr = rec_indices[
        :,
        0
    ]

    cc = rec_indices[
        :,
        1
    ]

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

    # ----------------------------------------------------------
    # Hybrid receiver-as-source basis
    # ----------------------------------------------------------

    seen_mask = np.zeros(
        64,
        dtype=bool,
    )

    seen_mask[
        source_positions
    ] = True

    unseen_mask = ~seen_mask

    # Exact for 8 seen source locations;
    # neural for 56 unseen positions.
    basis_exact_seen = (
        pred64.copy()
    )

    basis_exact_seen[
        seen_mask
    ] = true64[
        seen_mask
    ]

    # Neural for seen8;
    # exact for unseen56.
    basis_exact_unseen = (
        true64.copy()
    )

    basis_exact_unseen[
        seen_mask
    ] = pred64[
        seen_mask
    ]

    def G(
        tx,
        basis,
        residual,
    ):
        return ano_gradient(
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

    grads = {}

    grads[
        "exact_all"
    ] = G(
        exact_tx,
        true64,
        exact_residual,
    )

    grads[
        "tx_only"
    ] = G(
        neural_tx,
        true64,
        exact_residual,
    )

    grads[
        "basis64_only"
    ] = G(
        exact_tx,
        pred64,
        exact_residual,
    )

    grads[
        "unseen56_only"
    ] = G(
        exact_tx,
        basis_exact_seen,
        exact_residual,
    )

    grads[
        "seen8_basis_only"
    ] = G(
        exact_tx,
        basis_exact_unseen,
        exact_residual,
    )

    grads[
        "all_field_factors"
    ] = G(
        neural_tx,
        pred64,
        exact_residual,
    )

    grads[
        "full_neural_ano"
    ] = G(
        neural_tx,
        pred64,
        neural_residual,
    )

    grads[
        "residual_only"
    ] = G(
        exact_tx,
        true64,
        neural_residual,
    )

    print("=" * 120)
    print("C5.4e TX / BASIS / SOURCE-COVERAGE DECOMPOSITION")
    print("=" * 120)

    print(
        "TX full RRMSE =",
        f"{rrmse(neural_tx, exact_tx):.6f}",
    )

    print(
        "measurement RRMSE =",
        f"{rrmse(neural_measurement, exact_measurement):.6f}",
    )

    print(
        "residual RRMSE =",
        f"{rrmse(neural_residual, exact_residual):.6f}",
    )

    print()
    print(
        f"{'method':34s}"
        f"{'raw':>12s}"
        f"{'crop32':>12s}"
        f"{'smooth5':>12s}"
        f"{'smooth9':>12s}"
    )

    reports = {}

    order = [
        (
            "Exact everything",
            "exact_all",
        ),
        (
            "Neural residual only",
            "residual_only",
        ),
        (
            "Neural TX only",
            "tx_only",
        ),
        (
            "Neural basis64 only",
            "basis64_only",
        ),
        (
            "Neural unseen56 basis only",
            "unseen56_only",
        ),
        (
            "Neural seen8 basis only",
            "seen8_basis_only",
        ),
        (
            "Neural TX + basis",
            "all_field_factors",
        ),
        (
            "Full neural ANO",
            "full_neural_ano",
        ),
    ]

    for label, key in order:
        reports[key] = show(
            label,
            grads[key],
            g_cbs,
        )

    print()
    print("=" * 120)
    print("DIAGNOSTIC GUIDE")
    print("=" * 120)

    print(
        "If TX-only is much worse than basis64-only:"
    )

    print(
        "  -> hard-medium transmitter forward accuracy "
        "is the main problem."
    )

    print()

    print(
        "If basis64-only is much worse than TX-only:"
    )

    print(
        "  -> adjoint basis quality is the main problem."
    )

    print()

    print(
        "If unseen56-only is much worse than seen8-basis-only:"
    )

    print(
        "  -> 8-source -> 64-source training is strongly justified."
    )

    print()

    print(
        "If unseen56-only remains good but basis64-only is poor:"
    )

    print(
        "  -> source coverage is NOT the main issue; "
        "medium generalization is."
    )


if __name__ == "__main__":
    main()
