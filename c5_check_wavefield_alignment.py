import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cbs_model import ConvergentBornSeries_Batch


def complex_rrmse(pred, target, eps=1e-12):
    num = np.linalg.norm(
        (pred - target).reshape(-1)
    )
    den = np.linalg.norm(
        target.reshape(-1)
    )
    return float(
        num / max(float(den), eps)
    )


def relative_l1(pred, target, eps=1e-12):
    num = np.mean(
        np.abs(pred - target)
    )
    den = np.mean(
        np.abs(target)
    )
    return float(
        num / max(float(den), eps)
    )


def complex_cosine(pred, target, eps=1e-12):
    p = pred.reshape(-1)
    t = target.reshape(-1)

    inner = np.real(
        np.vdot(t, p)
    )

    denom = (
        np.linalg.norm(p)
        *
        np.linalg.norm(t)
    )

    return float(
        inner / max(float(denom), eps)
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--sample",
        required=True,
    )

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    ap.add_argument(
        "--source_positions",
        default="0,8,16,24,32,40,48,56",
    )

    ap.add_argument(
        "--device",
        default="cuda:0",
    )

    args = ap.parse_args()

    sample_path = Path(
        args.sample
    )

    config_path = Path(
        args.config
    )

    out = Path(
        args.out
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Load original current-data CBS sample
    # --------------------------------------------------

    z = np.load(
        sample_path,
        allow_pickle=True,
    )

    cfg = json.load(
        open(
            config_path,
            "r",
            encoding="utf-8",
        )
    )

    target_480 = (
        z["target_480"]
        .astype(np.float32)
    )

    dobs_saved = (
        z["dobs_complex"]
        .astype(np.complex64)
    )

    src_all = (
        z["src_indices"]
        .astype(np.int64)
    )

    rec_indices = (
        z["rec_indices"]
        .astype(np.int64)
    )

    source_positions = np.asarray(
        [
            int(x)
            for x
            in args.source_positions.split(",")
            if x.strip()
        ],
        dtype=np.int64,
    )

    if np.any(
        source_positions < 0
    ) or np.any(
        source_positions
        >= len(src_all)
    ):
        raise ValueError(
            "source_positions out of range"
        )

    src_selected = (
        src_all[source_positions]
    )

    dobs_target = (
        dobs_saved[source_positions]
    )

    print("=" * 100)
    print(
        "C5.1a CURRENT-DATA WAVEFIELD ALIGNMENT"
    )
    print("=" * 100)

    print(
        "sample              =",
        sample_path,
    )

    print(
        "target_480 shape    =",
        target_480.shape,
    )

    print(
        "saved dobs shape    =",
        dobs_saved.shape,
    )

    print(
        "source positions    =",
        source_positions.tolist(),
    )

    print(
        "selected src coords ="
    )

    print(
        src_selected
    )

    print(
        "receivers           =",
        rec_indices.shape[0],
    )

    print()
    print(
        "frequency           =",
        cfg["frequency"],
    )

    print(
        "cbs_iters           =",
        cfg["cbs_iters"],
    )

    print(
        "boundary_width      =",
        cfg["boundary_width"],
    )

    print(
        "boundary_strength   =",
        cfg["boundary_strength"],
    )

    print(
        "boundary_type       =",
        cfg["boundary_type"],
    )

    # --------------------------------------------------
    # Run CBS with EXACT same physical configuration
    # --------------------------------------------------

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    sos = torch.from_numpy(
        target_480
    ).view(
        1, 1, 480, 480
    ).to(
        device=device,
        dtype=torch.float32,
    )

    print()
    print(
        "device              =",
        device,
    )

    print()
    print(
        "[RUN] regenerating "
        f"{len(src_selected)} full wavefields ..."
    )

    model = ConvergentBornSeries_Batch(
        f=float(
            cfg["frequency"]
        ),
        sos=sos,
        boundary_width=[
            int(cfg["boundary_width"]),
            int(cfg["boundary_width"]),
        ],
        boundary_strength=float(
            cfg["boundary_strength"]
        ),
        boundary_type=str(
            cfg["boundary_type"]
        ),
        src_loc_set=src_selected,
        device=str(device),
    )

    with torch.no_grad():
        u = model(
            max_iters=int(
                cfg["cbs_iters"]
            )
        )

    # [1, M, 480, 480]
    wavefields = (
        u[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.complex64)
    )

    rec_t = torch.from_numpy(
        rec_indices
    ).to(
        device=device,
        dtype=torch.long,
    )

    with torch.no_grad():

        dobs_pred_t = u[
            0,
            :,
            rec_t[:, 0],
            rec_t[:, 1],
        ]

    dobs_pred = (
        dobs_pred_t
        .detach()
        .cpu()
        .numpy()
        .astype(np.complex64)
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    diff = (
        dobs_pred
        -
        dobs_target
    )

    rr = complex_rrmse(
        dobs_pred,
        dobs_target,
    )

    rl1 = relative_l1(
        dobs_pred,
        dobs_target,
    )

    cos = complex_cosine(
        dobs_pred,
        dobs_target,
    )

    mean_abs_diff = float(
        np.mean(
            np.abs(diff)
        )
    )

    max_abs_diff = float(
        np.max(
            np.abs(diff)
        )
    )

    mean_abs_target = float(
        np.mean(
            np.abs(dobs_target)
        )
    )

    mean_abs_pred = float(
        np.mean(
            np.abs(dobs_pred)
        )
    )

    real_rmse = float(
        np.sqrt(
            np.mean(
                (
                    dobs_pred.real
                    -
                    dobs_target.real
                ) ** 2
            )
        )
    )

    imag_rmse = float(
        np.sqrt(
            np.mean(
                (
                    dobs_pred.imag
                    -
                    dobs_target.imag
                ) ** 2
            )
        )
    )

    print()
    print("=" * 100)
    print(
        "RECEIVER ALIGNMENT RESULTS"
    )
    print("=" * 100)

    print(
        f"complex RRMSE       = "
        f"{rr:.10e}"
    )

    print(
        f"relative L1         = "
        f"{rl1:.10e}"
    )

    print(
        f"complex cosine      = "
        f"{cos:.10f}"
    )

    print(
        f"mean |saved dobs|   = "
        f"{mean_abs_target:.10e}"
    )

    print(
        f"mean |regen dobs|   = "
        f"{mean_abs_pred:.10e}"
    )

    print(
        f"mean |difference|   = "
        f"{mean_abs_diff:.10e}"
    )

    print(
        f"max  |difference|   = "
        f"{max_abs_diff:.10e}"
    )

    print(
        f"real RMSE           = "
        f"{real_rmse:.10e}"
    )

    print(
        f"imag RMSE           = "
        f"{imag_rmse:.10e}"
    )

    # --------------------------------------------------
    # Per-source audit
    # --------------------------------------------------

    print()
    print("=" * 100)
    print(
        "PER-SOURCE ALIGNMENT"
    )
    print("=" * 100)

    per_source = []

    for i, pos in enumerate(
        source_positions
    ):

        src_rr = complex_rrmse(
            dobs_pred[i],
            dobs_target[i],
        )

        src_abs = float(
            np.mean(
                np.abs(
                    dobs_pred[i]
                    -
                    dobs_target[i]
                )
            )
        )

        row = {
            "source_position":
                int(pos),

            "source_coord":
                src_selected[i]
                .tolist(),

            "rrmse":
                src_rr,

            "mean_abs_diff":
                src_abs,
        }

        per_source.append(
            row
        )

        print(
            f"source={int(pos):2d} "
            f"coord="
            f"({src_selected[i,0]:3d},"
            f"{src_selected[i,1]:3d}) "
            f"RRMSE="
            f"{src_rr:.10e} "
            f"mean_abs_diff="
            f"{src_abs:.10e}"
        )

    # --------------------------------------------------
    # Save regenerated current-data wavefields
    # --------------------------------------------------

    np.savez_compressed(
        out /
        "train1_wavefield_alignment.npz",

        target_480=target_480,

        source_positions=
            source_positions,

        src_indices=
            src_selected,

        rec_indices=
            rec_indices,

        wavefields=
            wavefields,

        dobs_saved=
            dobs_target,

        dobs_regenerated=
            dobs_pred,
    )

    summary = {
        "sample":
            str(sample_path),

        "source_positions":
            source_positions.tolist(),

        "frequency":
            float(cfg["frequency"]),

        "cbs_iters":
            int(cfg["cbs_iters"]),

        "boundary_width":
            int(cfg["boundary_width"]),

        "boundary_strength":
            float(
                cfg["boundary_strength"]
            ),

        "boundary_type":
            str(cfg["boundary_type"]),

        "complex_rrmse":
            rr,

        "relative_l1":
            rl1,

        "complex_cosine":
            cos,

        "mean_abs_saved":
            mean_abs_target,

        "mean_abs_regenerated":
            mean_abs_pred,

        "mean_abs_diff":
            mean_abs_diff,

        "max_abs_diff":
            max_abs_diff,

        "real_rmse":
            real_rmse,

        "imag_rmse":
            imag_rmse,

        "per_source":
            per_source,
    }

    with open(
        out /
        "alignment_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 100)

    if rr < 1e-3:

        print(
            "[PASS] CURRENT-DATA "
            "WAVEFIELD ALIGNMENT"
        )

        print(
            "Receiver RRMSE < 1e-3."
        )

        print(
            "Safe to generate C5 "
            "wavefield dataset."
        )

    else:

        print(
            "[FAIL] ALIGNMENT NOT "
            "SUFFICIENT"
        )

        print(
            "Do NOT generate the "
            "wavefield dataset yet."
        )

    print("=" * 100)

    print(
        "output =",
        out,
    )


if __name__ == "__main__":
    main()
