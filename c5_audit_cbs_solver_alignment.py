import argparse
import re
from pathlib import Path

import numpy as np
import torch

from cbs_model import ConvergentBornSeries_Batch


def numeric_key(path):
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


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


def best_complex_scale(pred, target):
    p = pred.reshape(-1)
    t = target.reshape(-1)

    den = np.vdot(
        p,
        p,
    )

    if abs(den) < 1e-30:
        return 1.0 + 0j

    return np.vdot(
        p,
        t,
    ) / den


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data_root",
        required=True,
    )

    ap.add_argument(
        "--split",
        default="val",
    )

    ap.add_argument(
        "--image_idx",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--frequency",
        type=float,
        default=500000.0,
    )

    ap.add_argument(
        "--iters",
        type=int,
        default=80,
    )

    ap.add_argument(
        "--boundary_width",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--boundary_strength",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--boundary_type",
        default="PML3",
    )

    ap.add_argument(
        "--device",
        default="cuda:0",
    )

    args = ap.parse_args()

    split_dir = (
        Path(args.data_root)
        / args.split
    )

    files = sorted(
        split_dir.glob("*.npz"),
        key=numeric_key,
    )

    if not files:
        raise RuntimeError(
            f"No npz files under {split_dir}"
        )

    path = files[
        args.image_idx
    ]

    print("=" * 100)
    print("C5.3-0 CBS SOLVER ALIGNMENT AUDIT")
    print("=" * 100)

    print(
        "file              =",
        path,
    )

    print(
        "frequency         =",
        args.frequency,
    )

    print(
        "CBS iterations    =",
        args.iters,
    )

    print(
        "boundary_width    =",
        args.boundary_width,
    )

    print(
        "boundary_strength =",
        args.boundary_strength,
    )

    print(
        "boundary_type     =",
        args.boundary_type,
    )

    with np.load(
        path,
        allow_pickle=True,
    ) as z:

        print(
            "npz keys          =",
            list(z.keys()),
        )

        speed = z[
            "target_480"
        ].astype(
            np.float32
        )

        target_wave = z[
            "wavefields"
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

        dobs = (
            z[
                "dobs_complex"
            ].astype(
                np.complex64
            )
            if "dobs_complex" in z
            else None
        )

    print()
    print(
        "speed shape       =",
        speed.shape,
    )

    print(
        "target wave shape =",
        target_wave.shape,
    )

    print(
        "src shape         =",
        src_indices.shape,
    )

    print(
        "rec shape         =",
        rec_indices.shape,
    )

    if dobs is not None:
        print(
            "dobs shape        =",
            dobs.shape,
        )

    device = torch.device(
        args.device
    )

    sos = torch.from_numpy(
        speed
    ).float()[
        None,
        None,
    ].to(
        device
    )

    model = ConvergentBornSeries_Batch(
        f=args.frequency,
        sos=sos,
        boundary_width=[
            args.boundary_width,
            args.boundary_width,
        ],
        boundary_strength=
            args.boundary_strength,
        boundary_type=
            args.boundary_type,
        src_loc_set=
            src_indices,
        device=args.device,
    )

    print()
    print(
        "[CBS] solving all",
        len(src_indices),
        "sources ..."
    )

    with torch.no_grad():
        pred = model(
            max_iters=args.iters
        )

    pred = (
        pred[0]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.complex64
        )
    )

    print(
        "[CBS] prediction shape =",
        pred.shape,
    )

    if (
        pred.shape
        != target_wave.shape
    ):
        raise RuntimeError(
            f"Shape mismatch: "
            f"CBS={pred.shape}, "
            f"stored={target_wave.shape}"
        )

    rr = rec_indices[
        :,
        0
    ]

    cc = rec_indices[
        :,
        1
    ]

    target_rec = target_wave[
        :,
        rr,
        cc,
    ]

    pred_rec = pred[
        :,
        rr,
        cc,
    ]

    raw_full = rrmse(
        pred,
        target_wave,
    )

    raw_rec = rrmse(
        pred_rec,
        target_rec,
    )

    # Diagnose pure source-amplitude / phase mismatch.
    alpha = best_complex_scale(
        pred,
        target_wave,
    )

    scaled_pred = (
        alpha
        * pred
    )

    scaled_full = rrmse(
        scaled_pred,
        target_wave,
    )

    scaled_rec = rrmse(
        scaled_pred[
            :,
            rr,
            cc,
        ],
        target_rec,
    )

    print()
    print("=" * 100)
    print("GLOBAL ALIGNMENT")
    print("=" * 100)

    print(
        "raw full RRMSE    =",
        f"{raw_full:.8e}",
    )

    print(
        "raw rec  RRMSE    =",
        f"{raw_rec:.8e}",
    )

    print(
        "best complex alpha=",
        alpha,
    )

    print(
        "scaled full RRMSE =",
        f"{scaled_full:.8e}",
    )

    print(
        "scaled rec  RRMSE =",
        f"{scaled_rec:.8e}",
    )

    print()
    print("=" * 100)
    print("PER-SOURCE ALIGNMENT")
    print("=" * 100)

    source_full = []
    source_rec = []

    for s in range(
        target_wave.shape[0]
    ):
        sf = rrmse(
            pred[s],
            target_wave[s],
        )

        sr = rrmse(
            pred_rec[s],
            target_rec[s],
        )

        source_full.append(
            sf
        )

        source_rec.append(
            sr
        )

        print(
            f"source {s:02d} | "
            f"full={sf:.8e} | "
            f"rec={sr:.8e}"
        )

    if dobs is not None:
        print()
        print("=" * 100)
        print("STORED DOBS AUDIT")
        print("=" * 100)

        print(
            "receiver target shape =",
            target_rec.shape,
        )

        print(
            "stored dobs shape      =",
            dobs.shape,
        )

        if (
            dobs.shape
            == target_rec.shape
        ):
            dobs_rr = rrmse(
                target_rec,
                dobs,
            )

            print(
                "stored wave receiver "
                "vs dobs RRMSE =",
                f"{dobs_rr:.8e}",
            )

        else:
            print(
                "[INFO] dobs shape differs "
                "from 8-source receiver slice; "
                "no direct comparison made."
            )

    print()
    print("=" * 100)
    print("C5.3-0 ALIGNMENT DECISION")
    print("=" * 100)

    if (
        raw_full < 1e-3
        and raw_rec < 1e-3
    ):
        print(
            "[STRONG PASS] CBS solver "
            "reproduces stored wavefields."
        )

    elif (
        raw_full < 1e-2
        and raw_rec < 1e-2
    ):
        print(
            "[PASS] CBS solver alignment "
            "is adequate for gradient audit."
        )

    elif (
        scaled_full < 1e-2
        and raw_full >= 1e-2
    ):
        print(
            "[SCALING MISMATCH] spatial solution "
            "matches after a global complex scale. "
            "Do NOT run gradient audit yet."
        )

    else:
        print(
            "[FAIL] current CBS configuration does "
            "not reproduce the stored C5 wavefields. "
            "Do NOT treat its adjoint as ground truth."
        )


if __name__ == "__main__":
    main()
