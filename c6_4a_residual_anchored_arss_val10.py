import argparse
import csv
import hashlib
import json
import time
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
)

from c5_directional_fd_audit import (
    measurement_loss,
    normalize_direction,
    scalar,
)


def hash_array(x):
    return hashlib.sha256(
        np.ascontiguousarray(
            np.asarray(x, dtype=np.float32)
        ).tobytes()
    ).hexdigest()


def rrmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    return float(
        np.sqrt(
            np.mean(np.abs(a - b) ** 2)
        )
        /
        (
            np.sqrt(
                np.mean(np.abs(b) ** 2)
            )
            + 1e-12
        )
    )


def mse(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(
        np.mean((a - b) ** 2)
    )


def mae(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(
        np.mean(np.abs(a - b))
    )


def stat(values):
    x = np.asarray(values, np.float64)

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
        "--background64_npz",
        required=True,
    )

    ap.add_argument(
        "--mgno_ckpt",
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
        "--step",
        type=float,
        default=1.5,
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

    args = ap.parse_args()

    bridge_root = Path(
        args.bridge_dir
    )

    out = Path(
        args.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # Homogeneous source basis
    # ============================================================

    bz = np.load(
        args.background64_npz,
        allow_pickle=True,
    )

    background64 = bz[
        "background64"
    ].astype(np.complex64)

    background_rec = bz[
        "rec_indices"
    ].astype(np.int64)

    if background64.shape[0] != 64:
        raise RuntimeError(
            f"Expected background64[64,...], "
            f"got {background64.shape}"
        )

    # ============================================================
    # Frozen ARSS-trained MgNO-II
    # ============================================================

    (
        model,
        channels,
        vcycles,
    ) = load_mgno(
        args.mgno_ckpt,
        args.device,
    )

    print("=" * 125)
    print(
        "C6.4A RESIDUAL-ANCHORED ARSS — REAL C4 VAL"
    )
    print("=" * 125)

    print(
        "bridge       =",
        bridge_root,
    )

    print(
        "MgNO ckpt    =",
        args.mgno_ckpt,
    )

    print(
        "MgNO config  =",
        f"C{channels}/V{vcycles}",
    )

    print(
        "frozen step  =",
        args.step,
        "m/s",
    )

    print(
        "samples      =",
        f"{args.start}..{args.end}",
    )

    rows = []

    # ============================================================
    # Sample loop
    # ============================================================

    for i in range(
        args.start,
        args.end + 1,
    ):

        print()
        print("#" * 125)
        print(f"TEST {i}")
        print("#" * 125)

        p = (
            bridge_root
            / f"test_{i}.npz"
        )

        if not p.exists():
            raise FileNotFoundError(p)

        z = np.load(
            p,
            allow_pickle=True,
        )

        candidate = z[
            "candidate_480"
        ].astype(np.float32)

        target = z[
            "target_480"
        ].astype(np.float32)

        dobs = z[
            "dobs_complex"
        ].astype(np.complex64)

        source_positions = z[
            "source_positions"
        ].astype(np.int64)

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
            scalar(
                z["boundary_strength"]
            )
        )

        boundary_type = str(
            scalar(
                z["boundary_type"]
            )
        )

        if not np.array_equal(
            rec_indices,
            background_rec,
        ):
            raise RuntimeError(
                f"test_{i}: background geometry mismatch"
            )

        expected_src = rec_indices[
            source_positions
        ]

        if not np.array_equal(
            src_indices,
            expected_src,
        ):
            raise RuntimeError(
                f"test_{i}: source geometry mismatch"
            )

        candidate_hash = hash_array(
            candidate
        )

        rr = rec_indices[:, 0]
        cc = rec_indices[:, 1]

        # --------------------------------------------------------
        # Baseline true objective
        #
        # This is evaluation only and is NOT counted as
        # RA-ARSS method inference cost.
        # --------------------------------------------------------

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

        mae0 = mae(
            candidate,
            target,
        )

        print(
            f"J0={J0:.10e} "
            f"RR0={rr0:.8f} "
            f"MSE0={mse0:.4f}"
        )

        # ========================================================
        # A. RA-ARSS PHYSICAL PART:
        # ONLY 8 TRUE TRANSMITTER SOURCES
        # ========================================================

        t0 = time.perf_counter()

        true8 = solve_cbs_chunked(
            speed=candidate,
            source_indices=
                src_indices,
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

        true8 = np.asarray(
            true8,
            dtype=np.complex64,
        )

        true8_time = (
            time.perf_counter()
            - t0
        )

        true_measurement = true8[
            :,
            rr,
            cc,
        ]

        true_residual = (
            true_measurement
            - dobs
        )

        # ========================================================
        # B. FROZEN ARSS-MgNO 64-SOURCE BASIS
        # ========================================================

        t0 = time.perf_counter()

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

        pred64 = np.asarray(
            pred64,
            dtype=np.complex64,
        )

        neural64_time = (
            time.perf_counter()
            - t0
        )

        neural_tx = pred64[
            source_positions
        ]

        neural_measurement = (
            neural_tx[
                :,
                rr,
                cc,
            ]
        )

        neural_residual = (
            neural_measurement
            - dobs
        )

        # ========================================================
        # C. RA-ARSS GRADIENT
        # exact 8-source residual + neural 64-source basis
        # ========================================================

        g_ra = ano_gradient(
            candidate=candidate,
            tx_waves=
                neural_tx,
            basis64=
                pred64,
            residual=
                true_residual,
            frequency=
                frequency,
        )

        # ========================================================
        # D. FULL ARSS BASELINE
        # neural residual + neural basis
        # ========================================================

        g_full = ano_gradient(
            candidate=candidate,
            tx_waves=
                neural_tx,
            basis64=
                pred64,
            residual=
                neural_residual,
            frequency=
                frequency,
        )

        # ========================================================
        # E. EXACT 64-SOURCE ORACLE
        #
        # VALIDATION ONLY.
        # This 64-source solve must NOT be charged to RA-ARSS.
        # ========================================================

        t0 = time.perf_counter()

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

        true64 = np.asarray(
            true64,
            dtype=np.complex64,
        )

        true64_oracle_time = (
            time.perf_counter()
            - t0
        )

        true8_from64 = true64[
            source_positions
        ]

        tx_equiv_rr = rrmse(
            true8,
            true8_from64,
        )

        if tx_equiv_rr > 1e-6:
            raise RuntimeError(
                f"test_{i}: true8/true64 "
                f"TX mismatch {tx_equiv_rr}"
            )

        g_exact = ano_gradient(
            candidate=candidate,
            tx_waves=
                true8_from64,
            basis64=
                true64,
            residual=
                true_residual,
            frequency=
                frequency,
        )

        # ========================================================
        # Gradient diagnostics
        # ========================================================

        m_ra = metrics(
            g_ra,
            g_exact,
        )

        m_full = metrics(
            g_full,
            g_exact,
        )

        measurement_rr = rrmse(
            neural_measurement,
            true_measurement,
        )

        residual_rr = rrmse(
            neural_residual,
            true_residual,
        )

        print()
        print(
            "measurement RRMSE =",
            f"{measurement_rr:.6f}",
        )

        print(
            "residual RRMSE    =",
            f"{residual_rr:.6f}",
        )

        print(
            "RA cosine raw     =",
            f"{m_ra['raw']:+.6f}",
        )

        print(
            "Full cosine raw   =",
            f"{m_full['raw']:+.6f}",
        )

        # ========================================================
        # Frozen one-step descent
        # ========================================================

        methods = {
            "Exact-CBS":
                g_exact,

            "Full-ARSS":
                g_full,

            "RA-ARSS":
                g_ra,
        }

        gradients = {
            "Exact-CBS":
                g_exact,

            "Full-ARSS":
                g_full,

            "RA-ARSS":
                g_ra,
        }

        for name, grad in methods.items():

            direction = (
                normalize_direction(
                    grad
                )
            )

            direction_rms = float(
                np.sqrt(
                    np.mean(
                        direction ** 2
                    )
                )
            )

            if abs(
                direction_rms - 1.0
            ) > 1e-5:
                raise RuntimeError(
                    f"{name}: direction RMS "
                    f"{direction_rms}"
                )

            # Frozen C5 step.
            # NO clipping and NO retuning.
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
                max(
                    abs(J0),
                    1e-30,
                )
            )

            mse1 = mse(
                trial,
                target,
            )

            mae1 = mae(
                trial,
                target,
            )

            mse_gain = (
                (mse0 - mse1)
                /
                max(
                    mse0,
                    1e-30,
                )
            )

            if name == "Exact-CBS":
                cos_raw = 1.0
                cos_crop32 = 1.0
                cos_smooth5 = 1.0

            elif name == "RA-ARSS":
                cos_raw = m_ra["raw"]
                cos_crop32 = (
                    m_ra["crop32"]
                )
                cos_smooth5 = (
                    m_ra["smooth5"]
                )

            else:
                cos_raw = m_full["raw"]
                cos_crop32 = (
                    m_full["crop32"]
                )
                cos_smooth5 = (
                    m_full["smooth5"]
                )

            row = {
                "sample":
                    i,

                "method":
                    name,

                "step_rms_mps":
                    args.step,

                "cos_raw":
                    float(cos_raw),

                "cos_crop32":
                    float(
                        cos_crop32
                    ),

                "cos_smooth5":
                    float(
                        cos_smooth5
                    ),

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
                    bool(J1 < J0),

                "mse0":
                    float(mse0),

                "mse1":
                    float(mse1),

                "mse_gain":
                    float(mse_gain),

                "mae0":
                    float(mae0),

                "mae1":
                    float(mae1),

                "candidate_min":
                    float(
                        candidate.min()
                    ),

                "candidate_max":
                    float(
                        candidate.max()
                    ),

                "trial_min":
                    float(
                        trial.min()
                    ),

                "trial_max":
                    float(
                        trial.max()
                    ),

                "measurement_rrmse":
                    float(
                        measurement_rr
                    ),

                "residual_rrmse":
                    float(
                        residual_rr
                    ),

                "true_forward_sources_method":
                    (
                        64
                        if name == "Exact-CBS"
                        else (
                            8
                            if name == "RA-ARSS"
                            else 0
                        )
                    ),

                "candidate_sha256":
                    candidate_hash,
            }

            rows.append(row)

            print()
            print(
                f"{name:12s} | "
                f"cos={cos_raw:+.5f} | "
                f"drop={100*drop:+.3f}% | "
                f"RR={rr0:.5f}->{rr1:.5f} | "
                f"MSE={mse0:.2f}->{mse1:.2f} | "
                f"descent={J1 < J0}"
            )

        # ========================================================
        # Save light-weight gradient cache
        # ========================================================

        np.savez_compressed(
            out /
            f"test_{i}_gradients.npz",

            candidate_speed=
                candidate,

            candidate_sha256=
                np.asarray([
                    candidate_hash
                ]),

            exact_gradient=
                g_exact.astype(
                    np.float64
                ),

            full_arss_gradient=
                g_full.astype(
                    np.float64
                ),

            ra_arss_gradient=
                g_ra.astype(
                    np.float64
                ),

            true_residual=
                true_residual,

            neural_residual=
                neural_residual,

            source_positions=
                source_positions,

            src_indices=
                src_indices,

            rec_indices=
                rec_indices,
        )

        timing = {
            "sample": i,

            # Actual RA method components:
            "ra_true8_seconds":
                true8_time,

            "ra_neural64_seconds":
                neural64_time,

            "ra_core_seconds":
                (
                    true8_time
                    + neural64_time
                ),

            # Validation oracle only:
            "exact64_oracle_seconds":
                true64_oracle_time,

            "true8_true64_tx_rrmse":
                tx_equiv_rr,
        }

        (
            out /
            f"test_{i}_timing.json"
        ).write_text(
            json.dumps(
                timing,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ============================================================
    # CSV
    # ============================================================

    csv_path = (
        out /
        "all_steps.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    # ============================================================
    # Aggregate
    # ============================================================

    summary = {}

    for method in [
        "Exact-CBS",
        "Full-ARSS",
        "RA-ARSS",
    ]:

        subset = [
            r
            for r in rows
            if r["method"] == method
        ]

        summary[method] = {
            "num_samples":
                len(subset),

            "descent_count":
                int(
                    sum(
                        bool(
                            r["descent"]
                        )
                        for r in subset
                    )
                ),

            "relative_drop":
                stat([
                    r["relative_drop"]
                    for r in subset
                ]),

            "cos_raw":
                stat([
                    r["cos_raw"]
                    for r in subset
                ]),

            "cos_crop32":
                stat([
                    r["cos_crop32"]
                    for r in subset
                ]),

            "rr1":
                stat([
                    r["rr1"]
                    for r in subset
                ]),

            "mse1":
                stat([
                    r["mse1"]
                    for r in subset
                ]),

            "mse_gain":
                stat([
                    r["mse_gain"]
                    for r in subset
                ]),

            "true_forward_sources_method":
                subset[0][
                    "true_forward_sources_method"
                ],
        }

    ra = summary[
        "RA-ARSS"
    ]

    # Same robustness interpretation used
    # in the previous transfer discussion.
    if (
        ra["descent_count"] >= 8
        and
        ra["relative_drop"]["mean"] > 0
    ):
        ra_gate = "STRONG_PASS"

    elif (
        ra["descent_count"] >= 6
        and
        ra["relative_drop"]["mean"] > 0
    ):
        ra_gate = "PARTIAL_PASS"

    else:
        ra_gate = "FAIL"

    final = {
        "experiment":
            "C6.4A Residual-Anchored ARSS",

        "frozen_step_mps":
            args.step,

        "samples":
            [
                args.start,
                args.end,
            ],

        "ra_method_definition":
            (
                "8-source true-CBS residual + "
                "64-source frozen ARSS-MgNO "
                "wavefield/basis"
            ),

        "oracle_definition":
            (
                "64-source true-CBS wavefields + "
                "same calibrated adjoint formula"
            ),

        "evaluation_note":
            (
                "true-CBS J evaluations and "
                "64-source Exact oracle are "
                "validation-only costs; "
                "RA-ARSS method physics cost "
                "uses 8 true forward sources."
            ),

        "methods":
            summary,

        "ra_gate":
            ra_gate,
    }

    json_path = (
        out /
        "summary.json"
    )

    json_path.write_text(
        json.dumps(
            final,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 125)
    print("C6.4A VAL SUMMARY")
    print("=" * 125)

    for method in [
        "Exact-CBS",
        "Full-ARSS",
        "RA-ARSS",
    ]:

        s = summary[method]

        print()
        print(method)

        print(
            "  descent      =",
            f"{s['descent_count']}"
            f"/{s['num_samples']}"
        )

        print(
            "  mean drop    =",
            f"{100*s['relative_drop']['mean']:+.3f}%"
        )

        print(
            "  worst drop   =",
            f"{100*s['relative_drop']['min']:+.3f}%"
        )

        print(
            "  mean cosine  =",
            f"{s['cos_raw']['mean']:+.5f}"
        )

        print(
            "  mean MSEgain =",
            f"{100*s['mse_gain']['mean']:+.3f}%"
        )

        print(
            "  true sources =",
            s[
                "true_forward_sources_method"
            ]
        )

    print()
    print(
        "RA-ARSS gate =",
        ra_gate,
    )

    print()
    print(
        "saved CSV  =",
        csv_path,
    )

    print(
        "saved JSON =",
        json_path,
    )


if __name__ == "__main__":
    main()
