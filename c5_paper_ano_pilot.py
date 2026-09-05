import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def scalar(v):
    return np.asarray(v).reshape(-1)[0].item()


def cosine(a, b, eps=1e-30):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    den = np.linalg.norm(a) * np.linalg.norm(b)

    if den < eps:
        return float("nan")

    return float(
        np.dot(a, b) / den
    )


def aligned_error(pred, target, eps=1e-30):
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(target, dtype=np.float64).reshape(-1)

    den = np.dot(p, p)

    if den < eps:
        return float("nan"), float("nan")

    alpha = float(
        np.dot(p, t) / den
    )

    err = float(
        np.linalg.norm(alpha * p - t)
        /
        (
            np.linalg.norm(t)
            + eps
        )
    )

    return alpha, err


def rrmse(a, b, eps=1e-12):
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
            + eps
        )
    )


def smooth_grad(g, k):
    if k <= 1:
        return np.asarray(g)

    x = torch.from_numpy(
        np.asarray(g, dtype=np.float32)
    )[None, None]

    y = F.avg_pool2d(
        x,
        kernel_size=k,
        stride=1,
        padding=k // 2,
    )

    return (
        y[0, 0]
        .numpy()
        .astype(np.float64)
    )


def crop(a, margin):
    if margin <= 0:
        return a

    return a[
        margin:-margin,
        margin:-margin,
    ]


def report_gradient(
    name,
    grad,
    reference,
):
    result = {}

    print()
    print("=" * 110)
    print(name)
    print("=" * 110)

    for margin in [0, 16, 32]:
        a = crop(grad, margin)
        b = crop(reference, margin)

        c = cosine(a, b)

        alpha, err = aligned_error(
            a,
            b,
        )

        key = (
            "full"
            if margin == 0
            else f"crop{margin}"
        )

        result[key] = {
            "cosine": c,
            "optimal_scale": alpha,
            "aligned_rel_error": err,
        }

        print(
            f"{key:8s} | "
            f"cos={c:+.8f} | "
            f"scale={alpha:+.8e} | "
            f"aligned_err={err:.8f}"
        )

    for k in [3, 5, 9]:
        c = cosine(
            smooth_grad(grad, k),
            smooth_grad(reference, k),
        )

        result[f"smooth{k}"] = {
            "cosine": c
        }

        print(
            f"smooth{k:<2d} | "
            f"cos={c:+.8f}"
        )

    return result


# ============================================================================
# ANO gradient
# ============================================================================

def transform_complex(x, mode):
    if mode == "plain":
        return x

    if mode == "conj":
        return np.conj(x)

    raise ValueError(mode)


def build_ano_gradient(
    candidate_speed,
    tx_waves,
    basis64,
    residual,
    frequency,
    residual_mode="plain",
    basis_mode="plain",
    tx_mode="plain",
    output_mode="real",
):
    """
    Generic complex-convention version of paper Eq. (19)-(20).

    residual: [N, M]
    tx_waves: [N, H, W]
    basis64 : [M, H, W]

    Eq.19:
        Lambda_n = -sum_m residual_nm * Y_m

    Eq.20:
        g = -2*w^2/X^3 sum_n Lambda_n Y_n
    """

    r = transform_complex(
        residual,
        residual_mode,
    )

    basis = transform_complex(
        basis64,
        basis_mode,
    )

    tx = transform_complex(
        tx_waves,
        tx_mode,
    )

    # [N,H,W]
    lam = -np.einsum(
        "nm,mhw->nhw",
        r,
        basis,
        optimize=True,
    )

    # [H,W]
    interaction = np.einsum(
        "nhw,nhw->hw",
        lam,
        tx,
        optimize=True,
    )

    omega = (
        2.0
        * np.pi
        * float(frequency)
    )

    factor = (
        -2.0
        * omega ** 2
        /
        (
            np.asarray(
                candidate_speed,
                dtype=np.float64,
            ) ** 3
        )
    )

    g_complex = (
        factor
        * interaction
    )

    if output_mode == "real":
        return np.real(
            g_complex
        ).astype(
            np.float64
        )

    if output_mode == "imag":
        return np.imag(
            g_complex
        ).astype(
            np.float64
        )

    raise ValueError(
        output_mode
    )


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
        "--source64_npz",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    # ---------------------------------------------------------------------
    # Load sample
    # ---------------------------------------------------------------------

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

    # ---------------------------------------------------------------------
    # Load validated CBS gradient
    # ---------------------------------------------------------------------

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

    direct_autograd_mgno1 = gz[
        "mgno1_gradient"
    ].astype(
        np.float64
    )

    cbs_tx_old = gz[
        "cbs_wavefields"
    ].astype(
        np.complex64
    )

    # ---------------------------------------------------------------------
    # Load 64-source exact + MgNO waves
    # ---------------------------------------------------------------------

    sz = np.load(
        args.source64_npz,
        allow_pickle=True,
    )

    candidate64 = sz[
        "candidate_speed"
    ].astype(
        np.float32
    )

    true64 = sz[
        "true64"
    ].astype(
        np.complex64
    )

    pred64 = sz[
        "pred64"
    ].astype(
        np.complex64
    )

    rec64 = sz[
        "rec_indices"
    ].astype(
        np.int64
    )

    # ---------------------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------------------

    print("=" * 110)
    print("C5.4b-0 EXACT FORMULA CALIBRATION + PAPER-STYLE MgNO-I ANO")
    print("=" * 110)

    print(
        "sample             =",
        args.sample,
    )

    print(
        "frequency          =",
        frequency,
    )

    print(
        "source_positions   =",
        source_positions.tolist(),
    )

    print(
        "true64 shape       =",
        true64.shape,
    )

    print(
        "pred64 shape       =",
        pred64.shape,
    )

    if not np.array_equal(
        rec_indices,
        rec64,
    ):
        raise RuntimeError(
            "receiver geometry mismatch"
        )

    candidate_alignment = float(
        np.max(
            np.abs(
                candidate
                - candidate64
            )
        )
    )

    print(
        "candidate max diff =",
        f"{candidate_alignment:.8e}",
    )

    if candidate_alignment > 1e-6:
        raise RuntimeError(
            "candidate mismatch"
        )

    # The 8 transmitters are receiver indices [0,8,...,56].
    exact_tx = true64[
        source_positions
    ]

    neural_tx = pred64[
        source_positions
    ]

    tx_alignment = rrmse(
        exact_tx,
        cbs_tx_old,
    )

    print(
        "exact tx vs C5.3a CBS RRMSE =",
        f"{tx_alignment:.8e}",
    )

    if tx_alignment > 1e-6:
        raise RuntimeError(
            "Exact 8-source wavefield "
            "alignment failed."
        )

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

    print()
    print(
        "exact measurement RRMSE vs dobs =",
        f"{rrmse(exact_measurement, dobs):.8f}",
    )

    print(
        "neural measurement RRMSE vs dobs=",
        f"{rrmse(neural_measurement, dobs):.8f}",
    )

    print(
        "neural residual vs exact residual "
        "RRMSE =",
        f"{rrmse(neural_residual, exact_residual):.8f}",
    )

    # =====================================================================
    # A. Complex-convention calibration with EXACT wavefields
    # =====================================================================

    print()
    print("#" * 110)
    print("A. EXACT-WAVEFIELD COMPLEX-CONVENTION CALIBRATION")
    print("#" * 110)

    variants = []

    for residual_mode in [
        "plain",
        "conj",
    ]:
        for basis_mode in [
            "plain",
            "conj",
        ]:
            for tx_mode in [
                "plain",
                "conj",
            ]:
                for output_mode in [
                    "real",
                    "imag",
                ]:

                    name = (
                        f"r={residual_mode},"
                        f"B={basis_mode},"
                        f"T={tx_mode},"
                        f"out={output_mode}"
                    )

                    g = build_ano_gradient(
                        candidate_speed=
                            candidate,
                        tx_waves=
                            exact_tx,
                        basis64=
                            true64,
                        residual=
                            exact_residual,
                        frequency=
                            frequency,
                        residual_mode=
                            residual_mode,
                        basis_mode=
                            basis_mode,
                        tx_mode=
                            tx_mode,
                        output_mode=
                            output_mode,
                    )

                    c = cosine(
                        g,
                        g_cbs,
                    )

                    alpha, err = (
                        aligned_error(
                            g,
                            g_cbs,
                        )
                    )

                    variants.append(
                        {
                            "name":
                                name,

                            "residual_mode":
                                residual_mode,

                            "basis_mode":
                                basis_mode,

                            "tx_mode":
                                tx_mode,

                            "output_mode":
                                output_mode,

                            "cosine":
                                c,

                            "abs_cosine":
                                abs(c),

                            "scale":
                                alpha,

                            "aligned_error":
                                err,

                            "gradient":
                                g,
                        }
                    )

                    print(
                        f"{name:54s} | "
                        f"cos={c:+.8f} | "
                        f"err={err:.8f}"
                    )

    # Paper literal Eq.(19)-(20)
    paper_name = (
        "r=plain,B=plain,"
        "T=plain,out=real"
    )

    paper_variant = next(
        x
        for x in variants
        if x["name"]
        == paper_name
    )

    best = max(
        variants,
        key=lambda x:
            x["abs_cosine"],
    )

    print()
    print("=" * 110)
    print("FORMULA CALIBRATION SUMMARY")
    print("=" * 110)

    print(
        "paper-direct cosine =",
        f"{paper_variant['cosine']:+.8f}",
    )

    print(
        "best convention     =",
        best["name"],
    )

    print(
        "best cosine         =",
        f"{best['cosine']:+.8f}",
    )

    print(
        "best aligned error  =",
        f"{best['aligned_error']:.8f}",
    )

    # If best cosine is negative, flip global sign.
    global_sign = (
        1.0
        if best["cosine"] >= 0
        else -1.0
    )

    print(
        "calibrated sign     =",
        global_sign,
    )

    g_exact_cal = (
        global_sign
        * best["gradient"]
    )

    # =====================================================================
    # B. Error decomposition
    # =====================================================================

    print()
    print("#" * 110)
    print("B. ANO ERROR DECOMPOSITION")
    print("#" * 110)

    cfg = {
        "residual_mode":
            best["residual_mode"],

        "basis_mode":
            best["basis_mode"],

        "tx_mode":
            best["tx_mode"],

        "output_mode":
            best["output_mode"],
    }

    def make_grad(
        tx,
        basis,
        residual,
    ):
        return (
            global_sign
            * build_ano_gradient(
                candidate_speed=
                    candidate,
                tx_waves=
                    tx,
                basis64=
                    basis,
                residual=
                    residual,
                frequency=
                    frequency,
                **cfg,
            )
        )

    # 1. exact residual + exact basis
    g_exact = make_grad(
        exact_tx,
        true64,
        exact_residual,
    )

    # 2. neural residual + exact basis
    g_residual_only = make_grad(
        exact_tx,
        true64,
        neural_residual,
    )

    # 3. exact residual + neural basis/tx
    g_basis_only = make_grad(
        neural_tx,
        pred64,
        exact_residual,
    )

    # 4. fully neural paper-style ANO
    g_neural_ano = make_grad(
        neural_tx,
        pred64,
        neural_residual,
    )

    reports = {}

    reports[
        "exact_formula"
    ] = report_gradient(
        "EXACT residual + EXACT wavefields",
        g_exact,
        g_cbs,
    )

    reports[
        "neural_residual_exact_basis"
    ] = report_gradient(
        "NEURAL residual + EXACT wavefields",
        g_residual_only,
        g_cbs,
    )

    reports[
        "exact_residual_neural_basis"
    ] = report_gradient(
        "EXACT residual + NEURAL wavefields",
        g_basis_only,
        g_cbs,
    )

    reports[
        "full_neural_ano"
    ] = report_gradient(
        "FULL MgNO-I PAPER-STYLE ANO",
        g_neural_ano,
        g_cbs,
    )

    reports[
        "direct_input_autograd"
    ] = report_gradient(
        "MgNO-I DIRECT INPUT-AUTOGRAD",
        direct_autograd_mgno1,
        g_cbs,
    )

    # =====================================================================
    # C. Summary
    # =====================================================================

    print()
    print("=" * 110)
    print("C5.4b PAPER-STYLE ANO FINAL SUMMARY")
    print("=" * 110)

    order = [
        (
            "Exact formula",
            "exact_formula",
        ),
        (
            "Neural residual only",
            "neural_residual_exact_basis",
        ),
        (
            "Neural basis only",
            "exact_residual_neural_basis",
        ),
        (
            "Full neural ANO",
            "full_neural_ano",
        ),
        (
            "Direct autograd",
            "direct_input_autograd",
        ),
    ]

    print(
        f"{'method':24s}"
        f"{'raw_cos':>12s}"
        f"{'crop32':>12s}"
        f"{'smooth5':>12s}"
        f"{'smooth9':>12s}"
    )

    for label, key in order:
        r = reports[key]

        print(
            f"{label:24s}"
            f"{r['full']['cosine']:12.6f}"
            f"{r['crop32']['cosine']:12.6f}"
            f"{r['smooth5']['cosine']:12.6f}"
            f"{r['smooth9']['cosine']:12.6f}"
        )

    exact_cos = reports[
        "exact_formula"
    ][
        "full"
    ][
        "cosine"
    ]

    ano_cos = reports[
        "full_neural_ano"
    ][
        "full"
    ][
        "cosine"
    ]

    auto_cos = reports[
        "direct_input_autograd"
    ][
        "full"
    ][
        "cosine"
    ]

    print()

    if exact_cos > 0.99:
        print(
            "[PASS] exact forward-only ANO "
            "formula reproduces the validated "
            "CBS adjoint direction."
        )
    else:
        print(
            "[WARNING] exact formula does not "
            "yet reproduce CBS gradient to "
            "cosine > 0.99."
        )

    if ano_cos > auto_cos:
        print(
            "[GAIN] paper-style neural ANO "
            "improves gradient direction over "
            "direct network input-autograd."
        )
    else:
        print(
            "[NO GAIN] paper-style neural ANO "
            "does not yet improve over direct "
            "input-autograd."
        )

    output = {
        "paper_direct": {
            k: v
            for k, v
            in paper_variant.items()
            if k != "gradient"
        },

        "calibrated_variant": {
            k: v
            for k, v
            in best.items()
            if k != "gradient"
        },

        "global_sign":
            global_sign,

        "tx_alignment_rrmse":
            tx_alignment,

        "exact_measurement_rrmse":
            rrmse(
                exact_measurement,
                dobs,
            ),

        "neural_measurement_rrmse":
            rrmse(
                neural_measurement,
                dobs,
            ),

        "reports":
            reports,
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
        out.with_suffix(".npz"),

        cbs_gradient=
            g_cbs,

        exact_ano_gradient=
            g_exact,

        neural_residual_exact_basis=
            g_residual_only,

        exact_residual_neural_basis=
            g_basis_only,

        neural_ano_gradient=
            g_neural_ano,

        direct_autograd_gradient=
            direct_autograd_mgno1,
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
