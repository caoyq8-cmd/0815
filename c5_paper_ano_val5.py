import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from c5_mgno_paper import MgNOBackgroundWavefield


def cosine(a, b, eps=1e-30):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    den = np.linalg.norm(a) * np.linalg.norm(b)

    if den < eps:
        return float("nan")

    return float(
        np.dot(a, b) / den
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


def stat(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def load_mgno(
    checkpoint,
    device,
):
    ckpt = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    cfg = ckpt.get(
        "args",
        {},
    )

    channels = int(
        cfg.get(
            "channels",
            12,
        )
    )

    recurrent_iters = int(
        cfg.get(
            "recurrent_iters",
            4,
        )
    )

    model = MgNOBackgroundWavefield(
        channels=channels,
        recurrent_iters=
            recurrent_iters,
        use_checkpoint=False,
    ).to(device)

    model.load_state_dict(
        ckpt[
            "model_state"
        ]
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


def predict64(
    model,
    speed,
    background64,
    speed_mean,
    speed_std,
    wave_scale,
    device,
):
    speed_t = torch.from_numpy(
        speed.astype(np.float32)
    ).to(device)

    speed_norm = (
        speed_t
        - speed_mean
    ) / speed_std

    pred = []

    with torch.no_grad():

        for s in range(64):

            bg = torch.from_numpy(
                background64[s]
            ).to(device)

            bg2 = torch.stack(
                [
                    bg.real,
                    bg.imag,
                ],
                dim=0,
            ) / wave_scale

            inp = torch.cat(
                [
                    speed_norm[
                        None,
                        None,
                    ],
                    bg2[
                        None,
                    ],
                ],
                dim=1,
            )

            out = model(
                inp
            )[0]

            u = (
                out[0]
                + 1j * out[1]
            ) * wave_scale

            pred.append(
                u.cpu().numpy()
            )

    return np.stack(
        pred,
        axis=0,
    ).astype(np.complex64)


def ano_gradient(
    candidate,
    tx_waves,
    basis64,
    residual,
    frequency,
):
    """
    Calibrated C5.4b convention:

      residual = plain
      basis    = conjugated
      tx       = conjugated
      output   = real
      global sign = -1
    """

    lam = -np.einsum(
        "nm,mhw->nhw",
        residual,
        np.conj(basis64),
        optimize=True,
    )

    interaction = np.einsum(
        "nhw,nhw->hw",
        lam,
        np.conj(tx_waves),
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
                candidate,
                dtype=np.float64,
            ) ** 3
        )
    )

    # C5.4b calibration found global_sign = -1.
    grad = (
        -1.0
        * np.real(
            factor
            * interaction
        )
    )

    return grad.astype(
        np.float64
    )


def metrics(
    grad,
    reference,
):
    return {
        "raw":
            cosine(
                grad,
                reference,
            ),

        "crop16":
            cosine(
                crop(grad, 16),
                crop(reference, 16),
            ),

        "crop32":
            cosine(
                crop(grad, 32),
                crop(reference, 32),
            ),

        "smooth3":
            cosine(
                smooth_grad(grad, 3),
                smooth_grad(reference, 3),
            ),

        "smooth5":
            cosine(
                smooth_grad(grad, 5),
                smooth_grad(reference, 5),
            ),

        "smooth9":
            cosine(
                smooth_grad(grad, 9),
                smooth_grad(reference, 9),
            ),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--sample_dir",
        required=True,
    )

    ap.add_argument(
        "--gradient_root",
        required=True,
    )

    ap.add_argument(
        "--source64_npz",
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
        "--device",
        default="cuda:0",
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    source64 = np.load(
        args.source64_npz,
        allow_pickle=True,
    )

    background64 = source64[
        "background64"
    ].astype(
        np.complex64
    )

    reference_rec = source64[
        "rec_indices"
    ].astype(
        np.int64
    )

    model = load_mgno(
        args.mgno_ckpt,
        args.device,
    )

    all_rows = []

    print("=" * 120)
    print("C5.4c VAL5 PAPER-STYLE MgNO-I ANO")
    print("=" * 120)

    for i in range(1, 6):

        print()
        print("#" * 120)
        print(
            f"TEST {i}"
        )
        print("#" * 120)

        sample_path = (
            Path(args.sample_dir)
            / f"test_{i}.npz"
        )

        grad_path = (
            Path(args.gradient_root)
            / f"test_{i}.npz"
        )

        z = np.load(
            sample_path,
            allow_pickle=True,
        )

        gz = np.load(
            grad_path,
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

        if not np.array_equal(
            rec_indices,
            reference_rec,
        ):
            raise RuntimeError(
                f"test_{i}: receiver geometry mismatch"
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

        exact_tx = gz[
            "cbs_wavefields"
        ].astype(
            np.complex64
        )

        pred64 = predict64(
            model=model,
            speed=candidate,
            background64=
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

        m_ano = metrics(
            g_ano,
            g_cbs,
        )

        m_direct = metrics(
            g_direct,
            g_cbs,
        )

        row = {
            "sample":
                i,

            "measurement_rrmse":
                rrmse(
                    neural_measurement,
                    exact_measurement,
                ),

            "residual_rrmse":
                rrmse(
                    neural_residual,
                    exact_residual,
                ),

            "ano":
                m_ano,

            "direct":
                m_direct,
        }

        all_rows.append(
            row
        )

        print(
            "measurement surrogate "
            "RRMSE =",
            f"{row['measurement_rrmse']:.6f}",
        )

        print(
            "residual RRMSE =",
            f"{row['residual_rrmse']:.6f}",
        )

        print(
            "ANO    | "
            f"raw={m_ano['raw']:+.6f} "
            f"crop32={m_ano['crop32']:+.6f} "
            f"s5={m_ano['smooth5']:+.6f} "
            f"s9={m_ano['smooth9']:+.6f}"
        )

        print(
            "DIRECT | "
            f"raw={m_direct['raw']:+.6f} "
            f"crop32={m_direct['crop32']:+.6f} "
            f"s5={m_direct['smooth5']:+.6f} "
            f"s9={m_direct['smooth9']:+.6f}"
        )

    print()
    print("=" * 120)
    print("C5.4c VAL5 AGGREGATE")
    print("=" * 120)

    keys = [
        "raw",
        "crop32",
        "smooth5",
        "smooth9",
    ]

    summary = {}

    for method in [
        "ano",
        "direct",
    ]:

        summary[method] = {}

        print()
        print(
            method.upper()
        )

        for k in keys:

            values = [
                row[
                    method
                ][k]
                for row in all_rows
            ]

            s = stat(
                values
            )

            summary[
                method
            ][k] = s

            print(
                f"{k:8s}: "
                f"{s['mean']:+.6f}"
                f" ± {s['std']:.6f} "
                f"[{s['min']:+.6f},"
                f" {s['max']:+.6f}]"
            )

    ano_raw = np.asarray(
        [
            x["ano"]["raw"]
            for x in all_rows
        ]
    )

    direct_raw = np.asarray(
        [
            x["direct"]["raw"]
            for x in all_rows
        ]
    )

    wins = int(
        np.sum(
            ano_raw
            > direct_raw
        )
    )

    pass05 = int(
        np.sum(
            ano_raw
            > 0.5
        )
    )

    pass08_s5 = int(
        np.sum(
            np.asarray(
                [
                    x[
                        "ano"
                    ][
                        "smooth5"
                    ]
                    for x in all_rows
                ]
            )
            > 0.8
        )
    )

    print()
    print(
        "ANO raw > direct raw =",
        wins,
        "/5",
    )

    print(
        "ANO raw > 0.5        =",
        pass05,
        "/5",
    )

    print(
        "ANO smooth5 > 0.8    =",
        pass08_s5,
        "/5",
    )

    output = {
        "rows":
            all_rows,

        "summary":
            summary,

        "ano_raw_wins":
            wins,

        "ano_raw_gt_0p5":
            pass05,

        "ano_smooth5_gt_0p8":
            pass08_s5,
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

    print()
    print(
        "saved =",
        out,
    )


if __name__ == "__main__":
    main()
