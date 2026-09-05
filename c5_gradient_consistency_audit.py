import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cbs_model import (
    ConvergentBornSeries_Batch,
    ConvergentBornSeries_Batch_Adjoint,
)

from c5_mgno_paper import (
    MgNOBackgroundWavefield,
)

from c5_shared8_capacity_fno import (
    FNOBackgroundWavefield,
)


# =============================================================================
# Metrics
# =============================================================================

def rrmse(a, b, eps=1e-12):
    a = np.asarray(a)
    b = np.asarray(b)

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


def cosine(a, b, eps=1e-30):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    den = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if den < eps:
        return float("nan")

    return float(
        np.dot(a, b) / den
    )


def norm2(a):
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    return float(
        np.linalg.norm(
            a.reshape(-1)
        )
    )


def aligned_relative_error(
    pred,
    target,
    eps=1e-30,
):
    """
    Scale-invariant gradient error.

    Find scalar alpha minimizing:
        ||alpha * pred - target||_2
    """

    p = np.asarray(
        pred,
        dtype=np.float64,
    ).reshape(-1)

    t = np.asarray(
        target,
        dtype=np.float64,
    ).reshape(-1)

    den = np.dot(
        p,
        p,
    )

    if den < eps:
        return (
            float("nan"),
            float("nan"),
        )

    alpha = float(
        np.dot(p, t)
        / den
    )

    err = float(
        np.linalg.norm(
            alpha * p - t
        )
        /
        (
            np.linalg.norm(t)
            + eps
        )
    )

    return alpha, err


def sign_agreement(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    mask = (
        np.abs(a) > 0
    ) & (
        np.abs(b) > 0
    )

    if not np.any(mask):
        return float("nan")

    return float(
        np.mean(
            np.sign(a[mask])
            ==
            np.sign(b[mask])
        )
    )


def smooth_grad(
    grad,
    kernel_size,
):
    if kernel_size <= 1:
        return grad.copy()

    x = torch.from_numpy(
        np.asarray(
            grad,
            dtype=np.float32,
        )
    )[None, None]

    y = F.avg_pool2d(
        x,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    return (
        y[0, 0]
        .numpy()
        .astype(
            np.float64
        )
    )


def crop(a, margin):
    if margin <= 0:
        return a

    return a[
        margin:-margin,
        margin:-margin,
    ]


def gradient_report(
    name,
    neural,
    physical,
):
    report = {}

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    for margin in [
        0,
        16,
        32,
    ]:
        ng = crop(
            neural,
            margin,
        )

        pg = crop(
            physical,
            margin,
        )

        c = cosine(
            ng,
            pg,
        )

        alpha, aligned_err = (
            aligned_relative_error(
                ng,
                pg,
            )
        )

        sa = sign_agreement(
            ng,
            pg,
        )

        key = (
            "full"
            if margin == 0
            else f"crop{margin}"
        )

        report[key] = {
            "cosine":
                c,

            "neg_cosine":
                -c,

            "sign_agreement":
                sa,

            "optimal_scale":
                alpha,

            "aligned_rel_error":
                aligned_err,
        }

        print(
            f"{key:8s} | "
            f"cos={c:+.6f} | "
            f"cos(-g)={-c:+.6f} | "
            f"sign={sa:.4f} | "
            f"aligned_err={aligned_err:.6f}"
        )

    print()
    print("SMOOTHED COSINE")

    for k in [
        3,
        5,
        9,
    ]:
        ns = smooth_grad(
            neural,
            k,
        )

        ps = smooth_grad(
            physical,
            k,
        )

        c = cosine(
            ns,
            ps,
        )

        report[
            f"smooth{k}"
        ] = {
            "cosine": c
        }

        print(
            f"k={k:2d} | "
            f"cos={c:+.6f}"
        )

    print()
    print(
        "||g_neural|| =",
        f"{norm2(neural):.8e}",
    )

    print(
        "||g_CBS||    =",
        f"{norm2(physical):.8e}",
    )

    return report


# =============================================================================
# Background
# =============================================================================

def load_true_background(
    path,
    src_indices,
):
    z = np.load(
        path,
        allow_pickle=True,
    )

    if "background_field" not in z:
        raise KeyError(
            "background_field missing"
        )

    bg = z[
        "background_field"
    ].astype(
        np.complex64
    )

    if "src_indices" in z:
        bg_src = z[
            "src_indices"
        ].astype(
            np.int64
        )

        selected = []

        for s in src_indices:
            idx = np.where(
                np.all(
                    bg_src
                    == s[None],
                    axis=1,
                )
            )[0]

            if len(idx) != 1:
                raise RuntimeError(
                    f"Cannot match source "
                    f"{s.tolist()}"
                )

            selected.append(
                bg[
                    int(idx[0])
                ]
            )

        bg = np.stack(
            selected,
            axis=0,
        )

    if (
        bg.shape[0]
        != len(src_indices)
    ):
        raise RuntimeError(
            f"background source count "
            f"{bg.shape[0]} != "
            f"{len(src_indices)}"
        )

    return bg


# =============================================================================
# Exact CBS physical gradient
# =============================================================================

def compute_cbs_gradient(
    candidate_speed,
    src_indices,
    rec_indices,
    dobs,
    frequency,
    cbs_iters,
    boundary_width,
    boundary_strength,
    boundary_type,
    device,
):
    sos = torch.from_numpy(
        candidate_speed
    ).float()[
        None,
        None,
    ].to(
        device
    )

    forward_model = (
        ConvergentBornSeries_Batch(
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
    )

    print()
    print(
        "[CBS] FFT grid =",
        tuple(
            int(x)
            for x in
            forward_model.new_N
        ),
    )

    with torch.no_grad():
        u = forward_model(
            max_iters=
                cbs_iters
        )

    # [1,8,480,480]
    u_np = (
        u.detach()
        .cpu()
        .numpy()[0]
        .astype(
            np.complex64
        )
    )

    dobs_t = torch.from_numpy(
        dobs
    )[
        None
    ].to(
        device
    )

    # All stored 64 receiver samples are active.
    mask = np.ones(
        dobs.shape,
        dtype=np.float32,
    )

    adjoint_model = (
        ConvergentBornSeries_Batch_Adjoint(
            forward_model,
            rec_indices,
            dobs_t,
            mask,
        )
    )

    grad, rec_diff_value = (
        adjoint_model(
            u,
            max_iters=
                cbs_iters,
        )
    )

    grad_np = (
        grad[
            0,
            0,
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    rr = rec_indices[
        :,
        0
    ]

    cc = rec_indices[
        :,
        1
    ]

    pred_rec = u_np[
        :,
        rr,
        cc,
    ]

    data_rr = rrmse(
        pred_rec,
        dobs,
    )

    print()
    print(
        "[CBS] candidate measurement RRMSE =",
        f"{data_rr:.8f}",
    )

    print(
        "[CBS] mean abs receiver residual  =",
        float(
            rec_diff_value
            .detach()
            .cpu()
        ),
    )

    print(
        "[CBS] gradient norm              =",
        f"{norm2(grad_np):.8e}",
    )

    return (
        grad_np,
        u_np,
        data_rr,
    )


# =============================================================================
# Neural models
# =============================================================================

def load_fno(
    checkpoint,
    device,
):
    ckpt = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    args = ckpt.get(
        "args",
        {},
    )

    model = (
        FNOBackgroundWavefield(
            modes=int(
                args.get(
                    "modes",
                    25,
                )
            ),
            width=int(
                args.get(
                    "width",
                    32,
                )
            ),
            depth=int(
                args.get(
                    "depth",
                    4,
                )
            ),
        )
        .to(device)
    )

    model.load_state_dict(
        ckpt[
            "model_state"
        ]
    )

    return model


def load_mgno(
    checkpoint,
    device,
):
    ckpt = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    args = ckpt.get(
        "args",
        {},
    )

    channels = int(
        args.get(
            "channels",
            12,
        )
    )

    recurrent_iters = int(
        args.get(
            "recurrent_iters",
            4,
        )
    )

    model = (
        MgNOBackgroundWavefield(
            channels=
                channels,
            recurrent_iters=
                recurrent_iters,
            use_checkpoint=False,
        )
        .to(device)
    )

    model.load_state_dict(
        ckpt[
            "model_state"
        ]
    )

    return (
        model,
        channels,
        recurrent_iters,
    )


def neural_input_gradient(
    model,
    candidate_speed,
    background,
    dobs,
    rec_indices,
    speed_mean,
    speed_std,
    wave_scale,
    device,
):
    model.eval()

    # Freeze network weights.
    for p in model.parameters():
        p.requires_grad_(
            False
        )

    speed = torch.tensor(
        candidate_speed,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    bg = torch.from_numpy(
        background
    ).to(
        device
    )

    rr = torch.from_numpy(
        rec_indices[
            :,
            0
        ]
    ).long().to(
        device
    )

    cc = torch.from_numpy(
        rec_indices[
            :,
            1
        ]
    ).long().to(
        device
    )

    dobs_t = torch.from_numpy(
        dobs
    ).to(
        device
    )

    pred_waves = []

    total_loss = 0.0

    # Source-by-source backward substantially
    # lowers input-gradient memory.
    for s in range(
        background.shape[0]
    ):
        speed_norm = (
            speed
            - speed_mean
        ) / speed_std

        bg_s = bg[
            s
        ]

        bg_2ch = torch.stack(
            [
                bg_s.real,
                bg_s.imag,
            ],
            dim=0,
        ) / wave_scale

        inp = torch.cat(
            [
                speed_norm[
                    None,
                    None,
                ],
                bg_2ch[
                    None,
                ],
            ],
            dim=1,
        )

        out = model(
            inp
        )[0]

        pred_complex = (
            out[0]
            + 1j * out[1]
        ) * wave_scale

        pred_waves.append(
            pred_complex
            .detach()
            .cpu()
            .numpy()
        )

        pred_rec = pred_complex[
            rr,
            cc,
        ]

        residual = (
            pred_rec
            - dobs_t[s]
        )

        # Positive scalar normalization does not
        # affect gradient cosine.
        loss_s = torch.mean(
            torch.abs(
                residual
            ) ** 2
        ) / float(
            background.shape[0]
        )

        loss_s.backward()

        total_loss += float(
            loss_s
            .detach()
            .cpu()
        )

    grad = (
        speed.grad
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    pred_waves = np.stack(
        pred_waves,
        axis=0,
    ).astype(
        np.complex64
    )

    return (
        grad,
        pred_waves,
        total_loss,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--sample",
        required=True,
    )

    ap.add_argument(
        "--background_path",
        required=True,
    )

    ap.add_argument(
        "--fno_ckpt",
        required=True,
    )

    ap.add_argument(
        "--mgno1_ckpt",
        required=True,
    )

    ap.add_argument(
        "--mgno2_ckpt",
        required=True,
    )

    ap.add_argument(
        "--rho",
        type=float,
        default=0.8,
        help=(
            "candidate = rho*GT "
            "+ (1-rho)*background_speed"
        ),
    )

    ap.add_argument(
        "--background_speed",
        type=float,
        default=1500.0,
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
        "--device",
        default="cuda:0",
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    device = args.device

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
        np.asarray(
            z["frequency"]
        ).reshape(-1)[0]
    )

    cbs_iters = int(
        np.asarray(
            z["cbs_iters"]
        ).reshape(-1)[0]
    )

    boundary_width = int(
        np.asarray(
            z["boundary_width"]
        ).reshape(-1)[0]
    )

    boundary_strength = float(
        np.asarray(
            z["boundary_strength"]
        ).reshape(-1)[0]
    )

    boundary_type = str(
        np.asarray(
            z["boundary_type"]
        ).reshape(-1)[0]
    )

    candidate = (
        args.rho
        * gt
        +
        (
            1.0
            - args.rho
        )
        * args.background_speed
    ).astype(
        np.float32
    )

    print("=" * 100)
    print("C5.3a NEURAL-vs-CBS GRADIENT CONSISTENCY AUDIT")
    print("=" * 100)

    print(
        "sample             =",
        args.sample,
    )

    print(
        "rho                =",
        args.rho,
    )

    print(
        "GT min/max         =",
        float(
            gt.min()
        ),
        float(
            gt.max()
        ),
    )

    print(
        "candidate min/max  =",
        float(
            candidate.min()
        ),
        float(
            candidate.max()
        ),
    )

    print(
        "frequency          =",
        frequency,
    )

    print(
        "CBS iters          =",
        cbs_iters,
    )

    print(
        "boundary_width     =",
        boundary_width,
    )

    print(
        "boundary_strength  =",
        boundary_strength,
    )

    print(
        "boundary_type      =",
        boundary_type,
    )

    background = load_true_background(
        args.background_path,
        src_indices,
    )

    # -----------------------------------------------------------------
    # Physical ground truth
    # -----------------------------------------------------------------

    print()
    print("#" * 100)
    print("1. EXACT CBS FORWARD + ADJOINT")
    print("#" * 100)

    (
        g_cbs,
        u_cbs,
        cbs_data_rr,
    ) = compute_cbs_gradient(
        candidate_speed=
            candidate,
        src_indices=
            src_indices,
        rec_indices=
            rec_indices,
        dobs=
            dobs,
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
            device,
    )

    rr = rec_indices[
        :,
        0
    ]

    cc = rec_indices[
        :,
        1
    ]

    results = {
        "rho":
            args.rho,

        "cbs_measurement_rrmse":
            cbs_data_rr,
    }

    # -----------------------------------------------------------------
    # FNO
    # -----------------------------------------------------------------

    print()
    print("#" * 100)
    print("2. FNO INPUT-AUTOGRAD")
    print("#" * 100)

    fno = load_fno(
        args.fno_ckpt,
        device,
    )

    (
        g_fno,
        u_fno,
        loss_fno,
    ) = neural_input_gradient(
        model=fno,
        candidate_speed=
            candidate,
        background=
            background,
        dobs=
            dobs,
        rec_indices=
            rec_indices,
        speed_mean=
            args.speed_mean,
        speed_std=
            args.speed_std,
        wave_scale=
            args.wave_scale,
        device=
            device,
    )

    fno_forward = rrmse(
        u_fno,
        u_cbs,
    )

    fno_rec = rrmse(
        u_fno[
            :,
            rr,
            cc,
        ],
        u_cbs[
            :,
            rr,
            cc,
        ],
    )

    print(
        "FNO forward vs CBS full RRMSE =",
        f"{fno_forward:.6f}",
    )

    print(
        "FNO forward vs CBS rec  RRMSE =",
        f"{fno_rec:.6f}",
    )

    fno_report = gradient_report(
        "FNO GRADIENT vs CBS",
        g_fno,
        g_cbs,
    )

    results["FNO"] = {
        "surrogate_loss":
            loss_fno,

        "forward_full_rrmse":
            fno_forward,

        "forward_rec_rrmse":
            fno_rec,

        "gradient":
            fno_report,
    }

    del fno

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # MgNO-I
    # -----------------------------------------------------------------

    print()
    print("#" * 100)
    print("3. MgNO-I INPUT-AUTOGRAD")
    print("#" * 100)

    (
        mg1,
        mg1_c,
        mg1_v,
    ) = load_mgno(
        args.mgno1_ckpt,
        device,
    )

    print(
        "MgNO-I config =",
        f"C{mg1_c}/V{mg1_v}",
    )

    (
        g_mg1,
        u_mg1,
        loss_mg1,
    ) = neural_input_gradient(
        model=mg1,
        candidate_speed=
            candidate,
        background=
            background,
        dobs=
            dobs,
        rec_indices=
            rec_indices,
        speed_mean=
            args.speed_mean,
        speed_std=
            args.speed_std,
        wave_scale=
            args.wave_scale,
        device=
            device,
    )

    mg1_forward = rrmse(
        u_mg1,
        u_cbs,
    )

    mg1_rec = rrmse(
        u_mg1[
            :,
            rr,
            cc,
        ],
        u_cbs[
            :,
            rr,
            cc,
        ],
    )

    print(
        "MgNO-I forward vs CBS full RRMSE =",
        f"{mg1_forward:.6f}",
    )

    print(
        "MgNO-I forward vs CBS rec  RRMSE =",
        f"{mg1_rec:.6f}",
    )

    mg1_report = gradient_report(
        "MgNO-I GRADIENT vs CBS",
        g_mg1,
        g_cbs,
    )

    results["MgNO-I"] = {
        "channels":
            mg1_c,

        "vcycles":
            mg1_v,

        "surrogate_loss":
            loss_mg1,

        "forward_full_rrmse":
            mg1_forward,

        "forward_rec_rrmse":
            mg1_rec,

        "gradient":
            mg1_report,
    }

    del mg1

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # MgNO-II
    # -----------------------------------------------------------------

    print()
    print("#" * 100)
    print("4. MgNO-II INPUT-AUTOGRAD")
    print("#" * 100)

    (
        mg2,
        mg2_c,
        mg2_v,
    ) = load_mgno(
        args.mgno2_ckpt,
        device,
    )

    print(
        "MgNO-II config =",
        f"C{mg2_c}/V{mg2_v}",
    )

    (
        g_mg2,
        u_mg2,
        loss_mg2,
    ) = neural_input_gradient(
        model=mg2,
        candidate_speed=
            candidate,
        background=
            background,
        dobs=
            dobs,
        rec_indices=
            rec_indices,
        speed_mean=
            args.speed_mean,
        speed_std=
            args.speed_std,
        wave_scale=
            args.wave_scale,
        device=
            device,
    )

    mg2_forward = rrmse(
        u_mg2,
        u_cbs,
    )

    mg2_rec = rrmse(
        u_mg2[
            :,
            rr,
            cc,
        ],
        u_cbs[
            :,
            rr,
            cc,
        ],
    )

    print(
        "MgNO-II forward vs CBS full RRMSE =",
        f"{mg2_forward:.6f}",
    )

    print(
        "MgNO-II forward vs CBS rec  RRMSE =",
        f"{mg2_rec:.6f}",
    )

    mg2_report = gradient_report(
        "MgNO-II GRADIENT vs CBS",
        g_mg2,
        g_cbs,
    )

    results["MgNO-II"] = {
        "channels":
            mg2_c,

        "vcycles":
            mg2_v,

        "surrogate_loss":
            loss_mg2,

        "forward_full_rrmse":
            mg2_forward,

        "forward_rec_rrmse":
            mg2_rec,

        "gradient":
            mg2_report,
    }

    # -----------------------------------------------------------------
    # Cross-model summary
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("C5.3a FINAL GRADIENT SUMMARY")
    print("=" * 100)

    print(
        f"{'model':12s} "
        f"{'fwd_full':>10s} "
        f"{'fwd_rec':>10s} "
        f"{'grad_cos':>10s} "
        f"{'crop16':>10s} "
        f"{'smooth5':>10s}"
    )

    for name in [
        "FNO",
        "MgNO-I",
        "MgNO-II",
    ]:
        r = results[
            name
        ]

        print(
            f"{name:12s} "
            f"{r['forward_full_rrmse']:10.6f} "
            f"{r['forward_rec_rrmse']:10.6f} "
            f"{r['gradient']['full']['cosine']:10.6f} "
            f"{r['gradient']['crop16']['cosine']:10.6f} "
            f"{r['gradient']['smooth5']['cosine']:10.6f}"
        )

    print()
    print("Interpretation guide:")

    print(
        "cos > 0.8  : strong directional agreement"
    )

    print(
        "0.5-0.8    : usable but imperfect"
    )

    print(
        "0.2-0.5    : weak gradient fidelity"
    )

    print(
        "|cos| < 0.2: essentially poor/near-orthogonal"
    )

    print(
        "cos < 0    : wrong descent direction / sign issue"
    )

    out = Path(
        args.output
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            results,
            indent=2,
        )
    )

    np.savez_compressed(
        out.with_suffix(
            ".npz"
        ),
        candidate_speed=
            candidate,

        cbs_gradient=
            g_cbs,

        fno_gradient=
            g_fno,

        mgno1_gradient=
            g_mg1,

        mgno2_gradient=
            g_mg2,

        cbs_wavefields=
            u_cbs,

        fno_wavefields=
            u_fno,

        mgno1_wavefields=
            u_mg1,

        mgno2_wavefields=
            u_mg2,
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
