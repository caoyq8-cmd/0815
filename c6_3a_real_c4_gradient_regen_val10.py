import argparse
import csv
import hashlib
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


def sha256_array(x):
    x = np.ascontiguousarray(
        np.asarray(x, dtype=np.float32)
    )
    return hashlib.sha256(
        x.tobytes()
    ).hexdigest()


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--bridge_dir",
        required=True,
        help=(
            "C6.2 final bridge directory containing "
            "test_1.npz ... test_N.npz"
        ),
    )

    ap.add_argument(
        "--background64_npz",
        required=True,
        help=(
            "NPZ containing exact homogeneous "
            "background64 and rec_indices"
        ),
    )

    ap.add_argument(
        "--mgno_ckpt",
        required=True,
        help="Frozen ARSS/Strat4+4-trained MgNO checkpoint",
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

    bridge_dir = Path(args.bridge_dir)
    out = Path(args.output_dir)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ================================================================
    # Fixed homogeneous 64-source backgrounds
    # ================================================================

    bz = np.load(
        args.background64_npz,
        allow_pickle=True,
    )

    if "background64" not in bz.files:
        raise RuntimeError(
            f"{args.background64_npz} "
            "does not contain background64"
        )

    background64 = bz[
        "background64"
    ].astype(np.complex64)

    background_rec = bz[
        "rec_indices"
    ].astype(np.int64)

    if background64.shape[0] != 64:
        raise RuntimeError(
            "Expected 64 homogeneous source "
            f"wavefields, got {background64.shape}"
        )

    # ================================================================
    # Frozen ARSS / Strat4+4-trained MgNO
    # ================================================================

    (
        model,
        channels,
        vcycles,
    ) = load_mgno(
        args.mgno_ckpt,
        args.device,
    )

    print("=" * 120)
    print(
        "C6.3A REAL C4 STATE GRADIENT REGENERATION"
    )
    print("=" * 120)

    print(
        "bridge_dir       =",
        bridge_dir,
    )

    print(
        "background64     =",
        args.background64_npz,
    )

    print(
        "ARSS MgNO ckpt   =",
        args.mgno_ckpt,
    )

    print(
        "MgNO config      =",
        f"C{channels}/V{vcycles}",
    )

    print(
        "samples          =",
        f"{args.start}..{args.end}",
    )

    print(
        "speed norm       =",
        args.speed_mean,
        args.speed_std,
    )

    print(
        "wave_scale       =",
        args.wave_scale,
    )

    rows = []

    # ================================================================
    # Main loop
    # ================================================================

    for i in range(
        args.start,
        args.end + 1,
    ):

        print()
        print("#" * 120)
        print(f"TEST {i}")
        print("#" * 120)

        sample_path = (
            bridge_dir
            / f"test_{i}.npz"
        )

        if not sample_path.exists():
            raise FileNotFoundError(
                sample_path
            )

        z = np.load(
            sample_path,
            allow_pickle=True,
        )

        required = [
            "candidate_480",
            "dobs_complex",
            "source_positions",
            "src_indices",
            "rec_indices",
            "frequency",
            "cbs_iters",
            "boundary_width",
            "boundary_strength",
            "boundary_type",
        ]

        missing = [
            k
            for k in required
            if k not in z.files
        ]

        if missing:
            raise RuntimeError(
                f"test_{i}: missing keys {missing}"
            )

        candidate = z[
            "candidate_480"
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
            scalar(z["boundary_type"])
        )

        if candidate.shape != (480, 480):
            raise RuntimeError(
                f"test_{i}: candidate shape "
                f"{candidate.shape}"
            )

        if not np.array_equal(
            rec_indices,
            background_rec,
        ):
            raise RuntimeError(
                f"test_{i}: receiver geometry "
                "does not match background64"
            )

        if dobs.shape != (
            len(source_positions),
            len(rec_indices),
        ):
            raise RuntimeError(
                f"test_{i}: dobs shape "
                f"{dobs.shape}, expected "
                f"({len(source_positions)}, "
                f"{len(rec_indices)})"
            )

        candidate_hash = (
            sha256_array(candidate)
        )

        print(
            "candidate range =",
            float(candidate.min()),
            float(candidate.max()),
        )

        print(
            "candidate SHA256 =",
            candidate_hash,
        )

        print(
            "measurement sources =",
            source_positions.tolist(),
        )

        print(
            "CBS config =",
            cbs_iters,
            boundary_width,
            boundary_strength,
            boundary_type,
        )

        # ============================================================
        # 1. TRUE 64-SOURCE CBS WAVEFIELDS AT REAL C4 STATE
        # ============================================================

        print()
        print(
            "[1/4] exact 64-source CBS "
            "wavefields"
        )

        true64 = solve_cbs_chunked(
            speed=candidate,

            # 64 transducers are used as
            # source basis positions.
            source_indices=rec_indices,

            frequency=frequency,
            cbs_iters=cbs_iters,

            boundary_width=
                boundary_width,

            boundary_strength=
                boundary_strength,

            boundary_type=
                boundary_type,

            device=args.device,

            chunk_size=
                args.chunk_size,
        )

        true64 = np.asarray(
            true64,
            dtype=np.complex64,
        )

        if true64.shape[0] != 64:
            raise RuntimeError(
                f"test_{i}: true64 shape "
                f"{true64.shape}"
            )

        exact_tx = true64[
            source_positions
        ]

        rr = rec_indices[:, 0]
        cc = rec_indices[:, 1]

        exact_measurement = (
            exact_tx[
                :,
                rr,
                cc,
            ]
        )

        exact_residual = (
            exact_measurement
            - dobs
        )

        # ============================================================
        # 2. EXACT-WAVEFIELD ADJOINT GRADIENT
        # ============================================================

        print(
            "[2/4] exact-CBS adjoint "
            "gradient"
        )

        g_exact = ano_gradient(
            candidate=candidate,

            tx_waves=
                exact_tx,

            basis64=
                true64,

            residual=
                exact_residual,

            frequency=
                frequency,
        )

        # ============================================================
        # 3. FROZEN ARSS-TRAINED MgNO 64-SOURCE WAVEFIELDS
        # ============================================================

        print(
            "[3/4] frozen ARSS-MgNO "
            "64-source wavefields"
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

        pred64 = np.asarray(
            pred64,
            dtype=np.complex64,
        )

        if pred64.shape != true64.shape:
            raise RuntimeError(
                f"test_{i}: neural/CBS shape "
                f"mismatch {pred64.shape} "
                f"vs {true64.shape}"
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

        # ============================================================
        # 4. FULL ARSS / NEURAL ANO GRADIENT
        # ============================================================

        print(
            "[4/4] full ARSS-ANO gradient"
        )

        g_arss = ano_gradient(
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

        # ============================================================
        # Diagnostics
        # ============================================================

        measurement_rr = rrmse(
            neural_measurement,
            exact_measurement,
        )

        residual_rr = rrmse(
            neural_residual,
            exact_residual,
        )

        field_rr = rrmse(
            pred64,
            true64,
        )

        m = metrics(
            g_arss,
            g_exact,
        )

        g_exact_norm = float(
            np.linalg.norm(
                g_exact.ravel()
            )
        )

        g_arss_norm = float(
            np.linalg.norm(
                g_arss.ravel()
            )
        )

        print()
        print(
            "64-source field RRMSE       =",
            f"{field_rr:.8f}",
        )

        print(
            "measurement surrogate RRMSE =",
            f"{measurement_rr:.8f}",
        )

        print(
            "residual RRMSE              =",
            f"{residual_rr:.8f}",
        )

        print(
            "cos(ARSS,Exact) raw         =",
            f"{m['raw']:+.8f}",
        )

        print(
            "cos(ARSS,Exact) crop32      =",
            f"{m['crop32']:+.8f}",
        )

        print(
            "cos(ARSS,Exact) smooth5     =",
            f"{m['smooth5']:+.8f}",
        )

        print(
            "exact grad norm             =",
            f"{g_exact_norm:.8e}",
        )

        print(
            "ARSS grad norm              =",
            f"{g_arss_norm:.8e}",
        )

        # ============================================================
        # Save state-bound gradient cache
        # ============================================================

        cache_path = (
            out
            / f"decompose_test{i}.npz"
        )

        np.savez_compressed(
            cache_path,

            candidate_speed=
                candidate.astype(
                    np.float32
                ),

            candidate_sha256=
                np.asarray(
                    [candidate_hash]
                ),

            # Compatibility name for C6.3B.
            #
            # IMPORTANT:
            # This is the exact-CBS-wavefield
            # adjoint formula, not an old cached
            # gradient from another candidate.
            cbs_gradient=
                g_exact.astype(
                    np.float64
                ),

            exact_gradient=
                g_exact.astype(
                    np.float64
                ),

            full_ano_gradient=
                g_arss.astype(
                    np.float64
                ),

            true64=
                true64,

            pred64=
                pred64,

            exact_measurement=
                exact_measurement,

            neural_measurement=
                neural_measurement,

            exact_residual=
                exact_residual,

            neural_residual=
                neural_residual,

            source_positions=
                source_positions,

            src_indices=
                src_indices,

            rec_indices=
                rec_indices,

            frequency=
                np.asarray(
                    [frequency],
                    dtype=np.float64,
                ),

            cbs_iters=
                np.asarray(
                    [cbs_iters],
                    dtype=np.int32,
                ),

            boundary_width=
                np.asarray(
                    [boundary_width],
                    dtype=np.int32,
                ),

            boundary_strength=
                np.asarray(
                    [boundary_strength],
                    dtype=np.float32,
                ),

            boundary_type=
                np.asarray(
                    [boundary_type]
                ),

            exact_gradient_kind=
                np.asarray([
                    "exact_CBS_wavefields_"
                    "plus_C5p4b_calibrated_"
                    "adjoint_formula"
                ]),
        )

        row = {
            "sample": i,

            "candidate_sha256":
                candidate_hash,

            "candidate_min":
                float(
                    candidate.min()
                ),

            "candidate_max":
                float(
                    candidate.max()
                ),

            "field64_rrmse":
                field_rr,

            "measurement_rrmse":
                measurement_rr,

            "residual_rrmse":
                residual_rr,

            "cos_raw":
                m["raw"],

            "cos_crop32":
                m["crop32"],

            "cos_smooth5":
                m["smooth5"],

            "cos_smooth9":
                m["smooth9"],

            "exact_grad_norm":
                g_exact_norm,

            "arss_grad_norm":
                g_arss_norm,

            "cache_path":
                str(cache_path),
        }

        rows.append(row)

        print(
            "saved =",
            cache_path,
        )

    # ================================================================
    # Aggregate
    # ================================================================

    csv_path = (
        out
        / "gradient_summary.csv"
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

    def stat(values):
        x = np.asarray(
            values,
            dtype=np.float64,
        )

        return {
            "mean":
                float(x.mean()),

            "std":
                float(x.std()),

            "min":
                float(x.min()),

            "max":
                float(x.max()),
        }

    summary = {
        "num_samples":
            len(rows),

        "sample_start":
            args.start,

        "sample_end":
            args.end,

        "mgno_ckpt":
            str(args.mgno_ckpt),

        "background64_npz":
            str(args.background64_npz),

        "gradient_reference":
            (
                "exact CBS 64-source "
                "wavefields + C5.4b "
                "calibrated adjoint formula"
            ),

        "field64_rrmse":
            stat([
                r["field64_rrmse"]
                for r in rows
            ]),

        "measurement_rrmse":
            stat([
                r["measurement_rrmse"]
                for r in rows
            ]),

        "residual_rrmse":
            stat([
                r["residual_rrmse"]
                for r in rows
            ]),

        "cos_raw":
            stat([
                r["cos_raw"]
                for r in rows
            ]),

        "cos_crop32":
            stat([
                r["cos_crop32"]
                for r in rows
            ]),

        "cos_smooth5":
            stat([
                r["cos_smooth5"]
                for r in rows
            ]),
    }

    json_path = (
        out
        / "summary.json"
    )

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("C6.3A SUMMARY")
    print("=" * 120)

    print(
        "num samples =",
        len(rows),
    )

    for key in [
        "field64_rrmse",
        "measurement_rrmse",
        "residual_rrmse",
        "cos_raw",
        "cos_crop32",
        "cos_smooth5",
    ]:

        s = summary[key]

        print(
            f"{key:22s}: "
            f"{s['mean']:+.8f} "
            f"± {s['std']:.8f} "
            f"[{s['min']:+.8f}, "
            f"{s['max']:+.8f}]"
        )

    print()
    print(
        "saved CSV =",
        csv_path,
    )

    print(
        "saved JSON =",
        json_path,
    )


if __name__ == "__main__":
    main()
