import argparse
import json
from pathlib import Path

import numpy as np

from c5_audit_mgno_unseen64_sources import (
    solve_cbs_chunked,
    load_mgno,
    neural_predict,
)

from c5_paper_ano_val5 import (
    ano_gradient,
    metrics,
    rrmse,
)


def scalar(v):
    return np.asarray(v).reshape(-1)[0].item()


def stat(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "median": float(
            np.median(x)
        ),
    }


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
        "--background64_npz",
        required=True,
    )

    ap.add_argument(
        "--mgno_ckpt",
        required=True,
    )

    ap.add_argument(
        "--speed_mean",
        type=float,
        default=1488.39,
    )

    ap.add_argument(
        "--speed_std",
        type=float,
        default=27.53,
    )

    ap.add_argument(
        "--wave_scale",
        type=float,
        default=3.72290883e-02,
    )

    ap.add_argument(
        "--chunk_size",
        type=int,
        default=8,
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

    # ================================================================
    # Sample
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

    # ================================================================
    # Existing C5.3 gradient reference
    # ================================================================

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

    g_direct = gz[
        "mgno1_gradient"
    ].astype(
        np.float64
    )

    exact_tx_reference = gz[
        "cbs_wavefields"
    ].astype(
        np.complex64
    )

    # ================================================================
    # Reuse exact homogeneous background64 from test_1
    # ================================================================

    bz = np.load(
        args.background64_npz,
        allow_pickle=True,
    )

    background64 = bz[
        "background64"
    ].astype(
        np.complex64
    )

    background_rec = bz[
        "rec_indices"
    ].astype(
        np.int64
    )

    if not np.array_equal(
        background_rec,
        rec_indices,
    ):
        raise RuntimeError(
            "Receiver/source geometry mismatch "
            "between background64 and sample."
        )

    print("=" * 120)
    print("C5.4d ANO FAILURE DECOMPOSITION")
    print("=" * 120)

    print(
        "sample            =",
        args.sample,
    )

    print(
        "candidate range   =",
        float(candidate.min()),
        float(candidate.max()),
    )

    print(
        "frequency         =",
        frequency,
    )

    print(
        "CBS config        =",
        cbs_iters,
        boundary_width,
        boundary_strength,
        boundary_type,
    )

    # ================================================================
    # Exact 64-source wavefields for THIS candidate
    # ================================================================

    print()
    print("#" * 120)
    print("1. EXACT 64-SOURCE CANDIDATE WAVEFIELDS")
    print("#" * 120)

    true64 = solve_cbs_chunked(
        speed=candidate,
        source_indices=
            rec_indices,
        frequency=
            frequency,
        cbs_iters=
            cbs_iters,
        boundary_width=
            boundary_width,
        boundary_strength=
            boundary_strength,
        boundary_type=
            boundary_type,
        device=
            args.device,
        chunk_size=
            args.chunk_size,
    )

    exact_tx = true64[
        source_positions
    ]

    tx_alignment = rrmse(
        exact_tx,
        exact_tx_reference,
    )

    print()
    print(
        "exact64 transmitter alignment "
        "vs C5.3 =",
        f"{tx_alignment:.8e}",
    )

    if tx_alignment > 1e-6:
        raise RuntimeError(
            "Exact transmitter alignment failed."
        )

    # ================================================================
    # Neural 64-source wavefields
    # ================================================================

    print()
    print("#" * 120)
    print("2. MgNO-I 64-SOURCE WAVEFIELDS")
    print("#" * 120)

    (
        model,
        channels,
        vcycles,
    ) = load_mgno(
        args.mgno_ckpt,
        args.device,
    )

    print(
        "MgNO config =",
        f"C{channels}/V{vcycles}",
    )

    pred64 = neural_predict(
        model=model,
        speed=candidate,
        backgrounds=
            background64,
        speed_mean=
            args.speed_mean,
        speed_std=
            args.speed_std,
        wave_scale=
            args.wave_scale,
        device=
            args.device,
    )

    neural_tx = pred64[
        source_positions
    ]

    # ================================================================
    # Source accuracy
    # ================================================================

    rr = rec_indices[:, 0]
    cc = rec_indices[:, 1]

    seen_mask = np.zeros(
        64,
        dtype=bool,
    )

    seen_mask[
        source_positions
    ] = True

    unseen_mask = ~seen_mask

    full_err = np.asarray(
        [
            rrmse(
                pred64[s],
                true64[s],
            )
            for s in range(64)
        ]
    )

    rec_err = np.asarray(
        [
            rrmse(
                pred64[
                    s,
                    rr,
                    cc,
                ],
                true64[
                    s,
                    rr,
                    cc,
                ],
            )
            for s in range(64)
        ]
    )

    print()
    print("SOURCE-WAVEFIELD QUALITY")
    print("-" * 120)

    print(
        "seen8 full =",
        f"{full_err[seen_mask].mean():.6f}",
        "rec =",
        f"{rec_err[seen_mask].mean():.6f}",
    )

    print(
        "unseen56 full =",
        f"{full_err[unseen_mask].mean():.6f}",
        "rec =",
        f"{rec_err[unseen_mask].mean():.6f}",
    )

    # ================================================================
    # Residual
    # ================================================================

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

    measurement_error = rrmse(
        neural_measurement,
        exact_measurement,
    )

    residual_error = rrmse(
        neural_residual,
        exact_residual,
    )

    print()
    print("MEASUREMENT / RESIDUAL QUALITY")
    print("-" * 120)

    print(
        "measurement surrogate RRMSE =",
        f"{measurement_error:.6f}",
    )

    print(
        "residual RRMSE              =",
        f"{residual_error:.6f}",
    )

    # ================================================================
    # Fixed calibrated ANO convention from C5.4b test_1
    # ================================================================

    print()
    print("#" * 120)
    print("3. FIXED-CONVENTION ANO ERROR DECOMPOSITION")
    print("#" * 120)

    # Exact + Exact
    g_exact = ano_gradient(
        candidate=
            candidate,
        tx_waves=
            exact_tx,
        basis64=
            true64,
        residual=
            exact_residual,
        frequency=
            frequency,
    )

    # Neural residual ONLY
    g_residual = ano_gradient(
        candidate=
            candidate,
        tx_waves=
            exact_tx,
        basis64=
            true64,
        residual=
            neural_residual,
        frequency=
            frequency,
    )

    # Neural wavefield/basis ONLY
    g_basis = ano_gradient(
        candidate=
            candidate,
        tx_waves=
            neural_tx,
        basis64=
            pred64,
        residual=
            exact_residual,
        frequency=
            frequency,
    )

    # Full neural ANO
    g_ano = ano_gradient(
        candidate=
            candidate,
        tx_waves=
            neural_tx,
        basis64=
            pred64,
        residual=
            neural_residual,
        frequency=
            frequency,
    )

    reports = {
        "exact":
            metrics(
                g_exact,
                g_cbs,
            ),

        "neural_residual_only":
            metrics(
                g_residual,
                g_cbs,
            ),

        "neural_basis_only":
            metrics(
                g_basis,
                g_cbs,
            ),

        "full_neural_ano":
            metrics(
                g_ano,
                g_cbs,
            ),

        "direct_autograd":
            metrics(
                g_direct,
                g_cbs,
            ),
    }

    print()
    print("=" * 120)
    print("C5.4d FINAL DECOMPOSITION")
    print("=" * 120)

    print(
        f"{'method':26s}"
        f"{'raw':>12s}"
        f"{'crop32':>12s}"
        f"{'smooth5':>12s}"
        f"{'smooth9':>12s}"
    )

    order = [
        (
            "Exact formula",
            "exact",
        ),
        (
            "Neural residual only",
            "neural_residual_only",
        ),
        (
            "Neural basis only",
            "neural_basis_only",
        ),
        (
            "Full neural ANO",
            "full_neural_ano",
        ),
        (
            "Direct autograd",
            "direct_autograd",
        ),
    ]

    for label, key in order:

        m = reports[key]

        print(
            f"{label:26s}"
            f"{m['raw']:12.6f}"
            f"{m['crop32']:12.6f}"
            f"{m['smooth5']:12.6f}"
            f"{m['smooth9']:12.6f}"
        )

    residual_cos = reports[
        "neural_residual_only"
    ][
        "raw"
    ]

    basis_cos = reports[
        "neural_basis_only"
    ][
        "raw"
    ]

    full_cos = reports[
        "full_neural_ano"
    ][
        "raw"
    ]

    print()
    print("DIAGNOSIS")
    print("-" * 120)

    if (
        residual_cos < 0.5
        and basis_cos >= 0.5
    ):
        diagnosis = (
            "RESIDUAL BOTTLENECK: "
            "transmitter measurement residual "
            "is the dominant failure mode."
        )

    elif (
        basis_cos < 0.5
        and residual_cos >= 0.5
    ):
        diagnosis = (
            "BASIS BOTTLENECK: "
            "receiver-as-source wavefields "
            "are the dominant failure mode."
        )

    elif (
        residual_cos < 0.5
        and basis_cos < 0.5
    ):
        diagnosis = (
            "BOTH BOTTLENECKS: "
            "residual and receiver-basis "
            "predictions are both inadequate."
        )

    else:
        diagnosis = (
            "COMPOUND ERROR: individual "
            "components remain directionally "
            "reasonable, but their errors "
            "compound in the full ANO."
        )

    print(diagnosis)

    print()
    print(
        "full ANO raw cosine =",
        f"{full_cos:+.6f}",
    )

    output = {
        "measurement_rrmse":
            measurement_error,

        "residual_rrmse":
            residual_error,

        "seen8_full":
            stat(
                full_err[
                    seen_mask
                ]
            ),

        "unseen56_full":
            stat(
                full_err[
                    unseen_mask
                ]
            ),

        "seen8_rec":
            stat(
                rec_err[
                    seen_mask
                ]
            ),

        "unseen56_rec":
            stat(
                rec_err[
                    unseen_mask
                ]
            ),

        "reports":
            reports,

        "diagnosis":
            diagnosis,
    }

    out = Path(
        args.output
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    np.savez_compressed(
        out.with_suffix(
            ".npz"
        ),

        candidate_speed=
            candidate,

        true64=
            true64,

        pred64=
            pred64,

        exact_gradient=
            g_exact,

        residual_only_gradient=
            g_residual,

        basis_only_gradient=
            g_basis,

        full_ano_gradient=
            g_ano,

        cbs_gradient=
            g_cbs,
    )

    print()
    print(
        "saved JSON =",
        out,
    )

    print(
        "saved NPZ  =",
        out.with_suffix(
            ".npz"
        ),
    )


if __name__ == "__main__":
    main()
