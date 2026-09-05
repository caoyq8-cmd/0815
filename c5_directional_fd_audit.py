import argparse
from pathlib import Path

import numpy as np
import torch

from cbs_model import ConvergentBornSeries_Batch


def scalar(v):
    return np.asarray(v).reshape(-1)[0].item()


def rrmse(a, b, eps=1e-12):
    return float(
        np.sqrt(
            np.mean(
                np.abs(a - b) ** 2
            )
        )
        /
        (
            np.sqrt(
                np.mean(
                    np.abs(b) ** 2
                )
            )
            + eps
        )
    )


def rms(a):
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    return float(
        np.sqrt(
            np.mean(
                a ** 2
            )
        )
    )


def normalize_direction(d):
    d = np.asarray(
        d,
        dtype=np.float64,
    )

    s = rms(d)

    if s < 1e-30:
        raise RuntimeError(
            "Direction has zero RMS."
        )

    return (
        d / s
    ).astype(
        np.float32
    )


def measurement_loss(
    speed,
    dobs,
    src_indices,
    rec_indices,
    frequency,
    cbs_iters,
    boundary_width,
    boundary_strength,
    boundary_type,
    device,
):
    """
    Exact physical objective:

        J(X) = 1/2 sum_{s,m}
               |P_m u_s(X) - d_{s,m}|^2
    """

    sos = (
        torch
        .from_numpy(
            speed.astype(
                np.float32
            )
        )
        .float()[
            None,
            None,
        ]
        .to(device)
    )

    solver = ConvergentBornSeries_Batch(
        f=frequency,
        sos=sos,
        boundary_width=[
            boundary_width,
            boundary_width,
        ],
        boundary_strength=
            boundary_strength,
        boundary_type=
            boundary_type,
        src_loc_set=
            src_indices,
        device=
            device,
    )

    with torch.no_grad():
        u = solver(
            max_iters=
                cbs_iters
        )

    u = (
        u[0]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.complex64
        )
    )

    rr = rec_indices[:, 0]
    cc = rec_indices[:, 1]

    pred = u[
        :,
        rr,
        cc,
    ]

    residual = (
        pred - dobs
    )

    loss = 0.5 * float(
        np.sum(
            np.abs(
                residual
            ) ** 2,
            dtype=np.float64,
        )
    )

    data_rr = rrmse(
        pred,
        dobs,
    )

    del solver
    del sos

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        loss,
        data_rr,
    )


def dot(a, b):
    return float(
        np.sum(
            np.asarray(
                a,
                dtype=np.float64,
            )
            *
            np.asarray(
                b,
                dtype=np.float64,
            ),
            dtype=np.float64,
        )
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
        "--eps",
        type=float,
        default=0.5,
        help=(
            "RMS speed perturbation "
            "in m/s."
        ),
    )

    ap.add_argument(
        "--device",
        default="cuda:0",
    )

    args = ap.parse_args()

    z = np.load(
        args.sample,
        allow_pickle=True,
    )

    gt = z[
        "target_480"
    ].astype(
        np.float32
    )

    dobs = z[
        "dobs_complex"
    ].astype(
        np.complex64
    )

    src_indices = z[
        "src_indices"
    ].astype(
        np.int64
    )

    rec_indices = z[
        "rec_indices"
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

    g = np.load(
        args.gradient_npz,
        allow_pickle=True,
    )

    candidate = g[
        "candidate_speed"
    ].astype(
        np.float32
    )

    g_cbs = g[
        "cbs_gradient"
    ].astype(
        np.float64
    )

    gradients = {
        "CBS":
            g_cbs,

        "FNO":
            g["fno_gradient"].astype(
                np.float64
            ),

        "MgNO-I":
            g["mgno1_gradient"].astype(
                np.float64
            ),

        "MgNO-II":
            g["mgno2_gradient"].astype(
                np.float64
            ),
    }

    print("=" * 110)
    print("C5.3b EXACT CBS DIRECTIONAL FINITE-DIFFERENCE AUDIT")
    print("=" * 110)

    print(
        "sample          =",
        args.sample,
    )

    print(
        "gradient_npz    =",
        args.gradient_npz,
    )

    print(
        "eps RMS (m/s)   =",
        args.eps,
    )

    print(
        "candidate range =",
        float(
            candidate.min()
        ),
        float(
            candidate.max()
        ),
    )

    print(
        "GT range        =",
        float(
            gt.min()
        ),
        float(
            gt.max()
        ),
    )

    print(
        "CBS config      =",
        frequency,
        cbs_iters,
        boundary_width,
        boundary_strength,
        boundary_type,
    )

    print()
    print(
        "[Baseline] using stored candidate "
        "from C5.3a"
    )

    # The baseline value is useful but not required
    # for the central finite difference.
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

    print(
        "J(candidate)        =",
        f"{J0:.10e}",
    )

    print(
        "measurement RRMSE   =",
        f"{rr0:.8f}",
    )

    rows = []

    for name, grad in gradients.items():

        d = normalize_direction(
            grad
        )

        x_plus = (
            candidate
            + args.eps * d
        ).astype(
            np.float32
        )

        x_minus = (
            candidate
            - args.eps * d
        ).astype(
            np.float32
        )

        print()
        print("-" * 110)
        print(
            f"DIRECTION: {name}"
        )
        print("-" * 110)

        print(
            "direction RMS =",
            f"{rms(d):.8f}",
        )

        print(
            "x+ range      =",
            float(
                x_plus.min()
            ),
            float(
                x_plus.max()
            ),
        )

        print(
            "x- range      =",
            float(
                x_minus.min()
            ),
            float(
                x_minus.max()
            ),
        )

        Jp, rrp = measurement_loss(
            x_plus,
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

        Jm, rrm = measurement_loss(
            x_minus,
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

        fd = (
            Jp - Jm
        ) / (
            2.0 * args.eps
        )

        adj = dot(
            g_cbs,
            d,
        )

        ratio = (
            fd / adj
            if abs(adj) > 1e-30
            else float("nan")
        )

        same_sign = (
            np.sign(fd)
            ==
            np.sign(adj)
        )

        print(
            "J(+)          =",
            f"{Jp:.10e}",
        )

        print(
            "J(-)          =",
            f"{Jm:.10e}",
        )

        print(
            "RR(+/-)       =",
            f"{rrp:.8f}",
            f"{rrm:.8f}",
        )

        print(
            "FD derivative =",
            f"{fd:+.10e}",
        )

        print(
            "<g_CBS,d>     =",
            f"{adj:+.10e}",
        )

        print(
            "FD/adj ratio  =",
            f"{ratio:+.10e}",
        )

        print(
            "same sign     =",
            bool(
                same_sign
            ),
        )

        rows.append(
            {
                "name":
                    name,

                "fd":
                    fd,

                "adj":
                    adj,

                "ratio":
                    ratio,

                "same_sign":
                    bool(
                        same_sign
                    ),
            }
        )

    print()
    print("=" * 110)
    print("C5.3b FINAL FD SUMMARY")
    print("=" * 110)

    print(
        f"{'direction':12s} "
        f"{'FD':>16s} "
        f"{'<g,d>':>16s} "
        f"{'FD/adj':>16s} "
        f"{'sign':>8s}"
    )

    valid_ratios = []

    for r in rows:

        print(
            f"{r['name']:12s} "
            f"{r['fd']:16.6e} "
            f"{r['adj']:16.6e} "
            f"{r['ratio']:16.6e} "
            f"{str(r['same_sign']):>8s}"
        )

        if (
            np.isfinite(
                r["ratio"]
            )
            and abs(
                r["adj"]
            ) > 1e-8
        ):
            valid_ratios.append(
                r["ratio"]
            )

    if valid_ratios:

        ratios = np.asarray(
            valid_ratios,
            dtype=np.float64,
        )

        abs_mean = float(
            np.mean(
                np.abs(
                    ratios
                )
            )
        )

        cv = float(
            np.std(
                ratios
            )
            /
            (
                abs(
                    np.mean(
                        ratios
                    )
                )
                + 1e-30
            )
        )

        print()
        print(
            "ratio mean =",
            f"{ratios.mean():+.8e}",
        )

        print(
            "ratio std  =",
            f"{ratios.std():.8e}",
        )

        print(
            "ratio CV   =",
            f"{cv:.6f}",
        )

        all_positive = bool(
            np.all(
                ratios > 0
            )
        )

        if (
            all_positive
            and cv < 0.20
        ):
            print()
            print(
                "[STRONG PASS] CBS adjoint is "
                "consistent with exact finite "
                "differences up to a global "
                "positive normalization factor."
            )

        elif all_positive:
            print()
            print(
                "[PASS/WARNING] signs agree, "
                "but normalization ratios vary. "
                "Inspect epsilon sensitivity."
            )

        else:
            print()
            print(
                "[FAIL] CBS adjoint direction "
                "does not consistently agree "
                "with exact finite differences."
            )


if __name__ == "__main__":
    main()
