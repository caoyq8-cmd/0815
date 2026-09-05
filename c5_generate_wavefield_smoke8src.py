import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from cbs_model import ConvergentBornSeries_Batch


def numeric_key(p):
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else -1


def complex_rrmse(pred, target, eps=1e-12):
    pred = np.asarray(pred)
    target = np.asarray(target)

    num = np.linalg.norm(
        (pred - target).reshape(-1)
    )

    den = np.linalg.norm(
        target.reshape(-1)
    )

    return float(
        num / max(float(den), eps)
    )


@torch.no_grad()
def regenerate(
    target_480,
    src_selected,
    rec_indices,
    cfg,
    device,
):

    sos = torch.from_numpy(
        target_480.astype(np.float32)
    ).view(
        1, 1, 480, 480
    ).to(
        device=device,
        dtype=torch.float32,
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

    u = model(
        max_iters=int(
            cfg["cbs_iters"]
        )
    )

    wavefields = (
        u[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.complex64)
    )

    rec_t = torch.from_numpy(
        rec_indices.astype(np.int64)
    ).to(
        device=device,
        dtype=torch.long,
    )

    dobs_regen = (
        u[
            0,
            :,
            rec_t[:, 0],
            rec_t[:, 1],
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.complex64)
    )

    return wavefields, dobs_regen


def process_split(
    input_dir,
    output_dir,
    max_samples,
    source_positions,
    cfg,
    device,
    alignment_tol,
):

    files = sorted(
        input_dir.glob("*.npz"),
        key=numeric_key,
    )

    if max_samples > 0:
        files = files[:max_samples]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for idx, p in enumerate(files, 1):

        with np.load(
            p,
            allow_pickle=True,
        ) as z:

            target_256 = (
                z["target_256"]
                .astype(np.float32)
            )

            target_480 = (
                z["target_480"]
                .astype(np.float32)
            )

            dobs_full = (
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

            source_file = (
                z["source_file"]
                .copy()
            )

            measurement_mode = (
                z["measurement_mode"]
                .copy()
            )

            frequency = (
                z["frequency"]
                .copy()
            )

            cbs_iters = (
                z["cbs_iters"]
                .copy()
            )

        src_selected = (
            src_all[source_positions]
        )

        dobs_saved = (
            dobs_full[source_positions]
        )

        print()
        print("=" * 100)

        print(
            f"[{idx:03d}/{len(files):03d}] "
            f"{p.name}"
        )

        print(
            "sources =",
            source_positions.tolist(),
        )

        wavefields, dobs_regen = regenerate(
            target_480=target_480,
            src_selected=src_selected,
            rec_indices=rec_indices,
            cfg=cfg,
            device=device,
        )

        rr = complex_rrmse(
            dobs_regen,
            dobs_saved,
        )

        print(
            f"receiver alignment RRMSE = "
            f"{rr:.10e}"
        )

        if not np.isfinite(rr):
            raise RuntimeError(
                f"Non-finite RRMSE: {p}"
            )

        if rr > alignment_tol:
            raise RuntimeError(
                f"Alignment failed for {p}: "
                f"RRMSE={rr:.6e} > "
                f"{alignment_tol:.6e}"
            )

        out_path = (
            output_dir /
            p.name
        )

        np.savez_compressed(
            out_path,

            target_256=
                target_256,

            target_480=
                target_480,

            source_positions=
                source_positions.astype(
                    np.int64
                ),

            src_indices=
                src_selected.astype(
                    np.int64
                ),

            rec_indices=
                rec_indices.astype(
                    np.int64
                ),

            wavefields=
                wavefields,

            dobs_complex=
                dobs_regen,

            dobs_saved=
                dobs_saved,

            source_file=
                source_file,

            measurement_mode=
                measurement_mode,

            frequency=
                frequency,

            cbs_iters=
                cbs_iters,

            boundary_width=
                np.array(
                    [cfg["boundary_width"]],
                    dtype=np.int32,
                ),

            boundary_strength=
                np.array(
                    [cfg["boundary_strength"]],
                    dtype=np.float32,
                ),

            boundary_type=
                np.array(
                    [cfg["boundary_type"]]
                ),

            alignment_rrmse=
                np.array(
                    [rr],
                    dtype=np.float64,
                ),
        )

        rows.append({
            "input": str(p),
            "output": str(out_path),
            "alignment_rrmse": rr,
        })

        print(
            "saved =",
            out_path,
        )

        print(
            "wavefields shape =",
            wavefields.shape,
        )

    return rows


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input_root",
        required=True,
    )

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--output_root",
        required=True,
    )

    ap.add_argument(
        "--max_train",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--max_val",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--source_positions",
        default="0,8,16,24,32,40,48,56",
    )

    ap.add_argument(
        "--alignment_tol",
        type=float,
        default=1e-3,
    )

    ap.add_argument(
        "--device",
        default="cuda:0",
    )

    args = ap.parse_args()

    input_root = Path(
        args.input_root
    )

    output_root = Path(
        args.output_root
    )

    cfg = json.load(
        open(
            args.config,
            "r",
            encoding="utf-8",
        )
    )

    source_positions = np.asarray(
        [
            int(x)
            for x in
            args.source_positions.split(",")
            if x.strip()
        ],
        dtype=np.int64,
    )

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "C5.1b CURRENT-DATA "
        "WAVEFIELD SMOKE GENERATOR"
    )
    print("=" * 100)

    print(
        "input root       =",
        input_root,
    )

    print(
        "output root      =",
        output_root,
    )

    print(
        "source positions =",
        source_positions.tolist(),
    )

    print(
        "frequency        =",
        cfg["frequency"],
    )

    print(
        "CBS iters        =",
        cfg["cbs_iters"],
    )

    print(
        "PML              =",
        cfg["boundary_type"],
        cfg["boundary_width"],
        cfg["boundary_strength"],
    )

    train_rows = process_split(
        input_dir=
            input_root / "train",

        output_dir=
            output_root / "train",

        max_samples=
            args.max_train,

        source_positions=
            source_positions,

        cfg=cfg,

        device=device,

        alignment_tol=
            args.alignment_tol,
    )

    val_rows = process_split(
        input_dir=
            input_root / "test",

        output_dir=
            output_root / "val",

        max_samples=
            args.max_val,

        source_positions=
            source_positions,

        cfg=cfg,

        device=device,

        alignment_tol=
            args.alignment_tol,
    )

    all_rr = np.array(
        [
            r["alignment_rrmse"]
            for r in
            train_rows + val_rows
        ],
        dtype=np.float64,
    )

    manifest = {
        "input_root":
            str(input_root),

        "train_count":
            len(train_rows),

        "val_count":
            len(val_rows),

        "num_sources":
            int(
                len(source_positions)
            ),

        "source_positions":
            source_positions.tolist(),

        "frequency":
            float(cfg["frequency"]),

        "cbs_iters":
            int(cfg["cbs_iters"]),

        "boundary_width":
            int(
                cfg["boundary_width"]
            ),

        "boundary_strength":
            float(
                cfg["boundary_strength"]
            ),

        "boundary_type":
            str(
                cfg["boundary_type"]
            ),

        "alignment_rrmse_mean":
            float(
                all_rr.mean()
            ),

        "alignment_rrmse_max":
            float(
                all_rr.max()
            ),

        "train":
            train_rows,

        "val":
            val_rows,
    }

    with open(
        output_root /
        "manifest.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print("=" * 100)
    print(
        "C5 WAVEFIELD DATASET SUMMARY"
    )
    print("=" * 100)

    print(
        "train images       =",
        len(train_rows),
    )

    print(
        "val images         =",
        len(val_rows),
    )

    print(
        "sources/image      =",
        len(source_positions),
    )

    print(
        "train wavefields   =",
        len(train_rows)
        *
        len(source_positions),
    )

    print(
        "val wavefields     =",
        len(val_rows)
        *
        len(source_positions),
    )

    print(
        "alignment mean     =",
        f"{all_rr.mean():.10e}",
    )

    print(
        "alignment max      =",
        f"{all_rr.max():.10e}",
    )

    print()

    print(
        "[PASS] C5.1b wavefield "
        "smoke dataset generated."
    )


if __name__ == "__main__":
    main()
