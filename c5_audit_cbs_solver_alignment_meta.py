import argparse
import re
from pathlib import Path

import numpy as np
import torch

from cbs_model import ConvergentBornSeries_Batch


def numeric_key(path):
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


def scalar(v):
    a = np.asarray(v)

    if a.size != 1:
        raise ValueError(
            f"Expected scalar metadata, got "
            f"shape={a.shape}, value={a}"
        )

    return a.reshape(-1)[0].item()


def string_value(v):
    x = scalar(v)

    if isinstance(x, bytes):
        return x.decode("utf-8")

    return str(x)


def boundary_width_value(v):
    a = np.asarray(v).reshape(-1)

    if len(a) == 1:
        x = int(a[0])
        return [x, x]

    if len(a) == 2:
        return [
            int(a[0]),
            int(a[1]),
        ]

    raise ValueError(
        f"Unexpected boundary_width: "
        f"shape={np.asarray(v).shape}, "
        f"value={v}"
    )


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
            f"No NPZ files in {split_dir}"
        )

    path = files[
        args.image_idx
    ]

    with np.load(
        path,
        allow_pickle=True,
    ) as z:

        print("=" * 100)
        print("C5.3-0b CBS METADATA-DRIVEN ALIGNMENT")
        print("=" * 100)

        print(
            "file =",
            path,
        )

        # ------------------------------------------------------------
        # Exact metadata used when dataset was generated
        # ------------------------------------------------------------

        required_meta = [
            "frequency",
            "cbs_iters",
            "boundary_width",
            "boundary_strength",
            "boundary_type",
        ]

        for k in required_meta:
            if k not in z:
                raise KeyError(
                    f"Dataset missing required "
                    f"generation metadata: {k}"
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

        boundary_width = (
            boundary_width_value(
                z["boundary_width"]
            )
        )

        boundary_strength = float(
            scalar(
                z["boundary_strength"]
            )
        )

        boundary_type = string_value(
            z["boundary_type"]
        )

        measurement_mode = (
            string_value(
                z["measurement_mode"]
            )
            if "measurement_mode" in z
            else "UNKNOWN"
        )

        stored_alignment = (
            float(
                scalar(
                    z["alignment_rrmse"]
                )
            )
            if "alignment_rrmse" in z
            else float("nan")
        )

        source_file = (
            string_value(
                z["source_file"]
            )
            if "source_file" in z
            else "UNKNOWN"
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

        source_positions = z[
            "source_positions"
        ].copy()

        rec_indices = z[
            "rec_indices"
        ].astype(
            np.int64
        )

        dobs = z[
            "dobs_complex"
        ].astype(
            np.complex64
        )

    print()
    print("STORED GENERATION CONFIG")
    print("-" * 100)

    print(
        "frequency         =",
        frequency,
    )

    print(
        "cbs_iters         =",
        cbs_iters,
    )

    print(
        "boundary_width    =",
        boundary_width,
    )

    print(
        "boundary_strength =",
        boundary_strength,
    )

    print(
        "boundary_type     =",
        boundary_type,
    )

    print(
        "measurement_mode  =",
        measurement_mode,
    )

    print(
        "alignment_rrmse   =",
        stored_alignment,
    )

    print(
        "source_file       =",
        source_file,
    )

    print()
    print(
        "source_positions shape =",
        source_positions.shape,
    )

    print(
        "src_indices shape      =",
        src_indices.shape,
    )

    print(
        "src_indices:"
    )

    print(
        src_indices
    )

    # ------------------------------------------------------------
    # Internal dataset consistency
    # ------------------------------------------------------------

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

    internal_rr = rrmse(
        target_rec,
        dobs,
    )

    print()
    print(
        "stored wave -> dobs RRMSE =",
        f"{internal_rr:.8e}",
    )

    # ------------------------------------------------------------
    # Re-run exact stored configuration
    # ------------------------------------------------------------

    device = torch.device(
        args.device
    )

    sos = (
        torch
        .from_numpy(
            speed
        )
        .float()[
            None,
            None,
        ]
        .to(device)
    )

    print()
    print("[CBS] constructing exact-metadata solver...")

    solver = ConvergentBornSeries_Batch(
        f=frequency,
        sos=sos,
        boundary_width=
            boundary_width,
        boundary_strength=
            boundary_strength,
        boundary_type=
            boundary_type,
        src_loc_set=
            src_indices,
        device=
            args.device,
    )

    print(
        "[CBS] internal FFT grid =",
        tuple(
            int(x)
            for x in solver.new_N
        ),
    )

    print(
        "[CBS] ROI =",
        solver.roi,
    )

    print(
        "[CBS] source amplitude max =",
        float(
            torch.abs(
                solver.src_set
            ).max()
            .detach()
            .cpu()
        ),
    )

    print(
        "[CBS] solving with",
        cbs_iters,
        "iterations ..."
    )

    with torch.no_grad():
        pred = solver(
            max_iters=
                cbs_iters
        )

    pred = (
        pred[
            0
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.complex64
        )
    )

    if (
        pred.shape
        != target_wave.shape
    ):
        raise RuntimeError(
            f"Shape mismatch: "
            f"pred={pred.shape}, "
            f"target={target_wave.shape}"
        )

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

    alpha = best_complex_scale(
        pred,
        target_wave,
    )

    scaled = (
        alpha
        * pred
    )

    scaled_full = rrmse(
        scaled,
        target_wave,
    )

    scaled_rec = rrmse(
        scaled[
            :,
            rr,
            cc,
        ],
        target_rec,
    )

    print()
    print("=" * 100)
    print("ALIGNMENT RESULT")
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
    print("PER SOURCE")
    print("-" * 100)

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

        print(
            f"source {s:02d} | "
            f"full={sf:.8e} | "
            f"rec={sr:.8e}"
        )

    print()
    print("=" * 100)
    print("DECISION")
    print("=" * 100)

    if (
        raw_full < 1e-4
        and raw_rec < 1e-4
    ):
        print(
            "[STRONG PASS] exact stored "
            "generation configuration reproduced."
        )

    elif (
        raw_full < 1e-2
        and raw_rec < 1e-2
    ):
        print(
            "[PASS] alignment adequate "
            "for gradient ground truth."
        )

    else:
        print(
            "[FAIL] metadata matches, but "
            "current cbs_model.py still does "
            "not reproduce the stored wavefield."
        )

        print()
        print(
            "If stored alignment_rrmse is tiny "
            "but current alignment is large, "
            "then the CBS implementation/version "
            "used for dataset generation differs "
            "from the current cbs_model.py."
        )


if __name__ == "__main__":
    main()
