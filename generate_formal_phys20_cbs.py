import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F

from cbs_model import ConvergentBornSeries_Batch


def load_mat_auto(path):
    path = Path(path)

    try:
        z = sio.loadmat(path)
        return {
            k: v
            for k, v in z.items()
            if not k.startswith("__")
        }

    except (NotImplementedError, ValueError):
        out = {}

        with h5py.File(path, "r") as f:
            for k in f.keys():
                obj = f[k]

                if isinstance(obj, h5py.Dataset):
                    out[k] = obj[()]

        return out


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def resize_480_to_256(x):
    t = torch.from_numpy(
        np.ascontiguousarray(x)
    ).float()[None, None]

    try:
        y = F.interpolate(
            t,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    except TypeError:
        y = F.interpolate(
            t,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )

    return y[0, 0].numpy()


@torch.no_grad()
def run_cbs(
    target_480,
    src_indices,
    rec_indices,
    frequency,
    cbs_iters,
    boundary_width,
    boundary_strength,
    boundary_type,
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
        f=float(frequency),
        sos=sos,
        boundary_width=[
            int(boundary_width),
            int(boundary_width),
        ],
        boundary_strength=float(
            boundary_strength
        ),
        boundary_type=str(
            boundary_type
        ),
        src_loc_set=src_indices,
        device=str(device),
    )

    u = model(
        max_iters=int(cbs_iters)
    )

    rec_t = torch.from_numpy(
        rec_indices.astype(np.int64)
    ).to(
        device=device,
        dtype=torch.long,
    )

    dobs = (
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

    wavefields = (
        u[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.complex64)
    )

    return dobs, wavefields


def scalar_from_npz(z, key):
    x = np.asarray(z[key])

    if x.size == 1:
        return x.reshape(-1)[0].item()

    raise ValueError(
        f"{key} is not scalar: {x.shape}"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--raw_root",
        required=True,
    )

    ap.add_argument(
        "--processed_root",
        required=True,
    )

    ap.add_argument(
        "--manifest_csv",
        required=True,
    )

    ap.add_argument(
        "--reference_npz",
        required=True,
    )

    ap.add_argument(
        "--output_root",
        required=True,
    )

    ap.add_argument(
        "--start_id",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--end_id",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--device",
        default="cuda:0",
    )

    ap.add_argument(
        "--save_wavefields",
        action="store_true",
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
    )

    ap.add_argument(
        "--identity_tol",
        type=float,
        default=1e-5,
    )

    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    processed_root = Path(
        args.processed_root
    )
    output_root = Path(
        args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device =",
        device,
    )

    # ---------------------------------------------------------
    # Load formal validation provenance
    # ---------------------------------------------------------

    mapping = {}

    with open(
        args.manifest_csv,
        newline="",
        encoding="utf-8-sig",
    ) as f:

        for row in csv.DictReader(f):
            mapping[
                int(row["val_id"])
            ] = row

    # ---------------------------------------------------------
    # Freeze physics protocol from old C5 reference
    # ---------------------------------------------------------

    ref = np.load(
        args.reference_npz,
        allow_pickle=True,
    )

    src_indices = (
        ref["src_indices"]
        .astype(np.int64)
    )

    rec_indices = (
        ref["rec_indices"]
        .astype(np.int64)
    )

    if src_indices.shape != (64, 2):
        raise RuntimeError(
            f"Unexpected src shape: "
            f"{src_indices.shape}"
        )

    if rec_indices.shape != (64, 2):
        raise RuntimeError(
            f"Unexpected rec shape: "
            f"{rec_indices.shape}"
        )

    source_positions = (
        ref["source_positions"]
        .astype(np.int64)
    )

    frequency = float(
        scalar_from_npz(
            ref,
            "frequency",
        )
    )

    cbs_iters = int(
        scalar_from_npz(
            ref,
            "cbs_iters",
        )
    )

    boundary_width = int(
        scalar_from_npz(
            ref,
            "boundary_width",
        )
    )

    boundary_strength = float(
        scalar_from_npz(
            ref,
            "boundary_strength",
        )
    )

    boundary_type = str(
        np.asarray(
            ref["boundary_type"]
        ).reshape(-1)[0]
    )

    measurement_mode = str(
        np.asarray(
            ref["measurement_mode"]
        ).reshape(-1)[0]
    )

    print("=" * 100)
    print("FORMAL PHYS20 CBS GENERATOR")
    print("=" * 100)

    print(
        "frequency         =",
        frequency,
    )
    print(
        "sources           =",
        src_indices.shape,
    )
    print(
        "receivers         =",
        rec_indices.shape,
    )
    print(
        "CBS iterations    =",
        cbs_iters,
    )
    print(
        "boundary          =",
        boundary_type,
        boundary_width,
        boundary_strength,
    )
    print(
        "measurement_mode  =",
        measurement_mode,
    )
    print(
        "save_wavefields   =",
        args.save_wavefields,
    )

    rows = []

    for sid in range(
        args.start_id,
        args.end_id + 1,
    ):

        raw_path = (
            raw_root
            / f"test_{sid}.mat"
        )

        proc_path = (
            processed_root
            / "val"
            / f"val_{sid}.mat"
        )

        out_path = (
            output_root
            / f"test_{sid}.npz"
        )

        if (
            out_path.exists()
            and not args.overwrite
        ):
            print(
                f"[SKIP] {out_path}"
            )
            continue

        if not raw_path.exists():
            raise FileNotFoundError(
                raw_path
            )

        if not proc_path.exists():
            raise FileNotFoundError(
                proc_path
            )

        raw = load_mat_auto(
            raw_path
        )

        proc = load_mat_auto(
            proc_path
        )

        if "slice" not in raw:
            raise KeyError(
                f"slice missing: "
                f"{raw_path}"
            )

        if "target_256" not in proc:
            raise KeyError(
                f"target_256 missing: "
                f"{proc_path}"
            )

        # IMPORTANT:
        # raw MAT coordinate -> physical CBS coordinate
        target_480 = (
            np.asarray(
                raw["slice"]
            )
            .squeeze()
            .astype(np.float32)
            .T
        )

        target_256 = (
            np.asarray(
                proc["target_256"]
            )
            .squeeze()
            .astype(np.float32)
        )

        if target_480.shape != (
            480,
            480,
        ):
            raise RuntimeError(
                f"Unexpected target480 "
                f"{target_480.shape}"
            )

        if target_256.shape != (
            256,
            256,
        ):
            raise RuntimeError(
                f"Unexpected target256 "
                f"{target_256.shape}"
            )

        # -----------------------------------------------------
        # Mandatory identity check
        # -----------------------------------------------------

        target_256_regen = (
            resize_480_to_256(
                target_480
            )
        )

        identity_mse = float(
            np.mean(
                (
                    target_256_regen
                    - target_256
                ) ** 2
            )
        )

        identity_mae = float(
            np.mean(
                np.abs(
                    target_256_regen
                    - target_256
                )
            )
        )

        if identity_mse > args.identity_tol:
            raise RuntimeError(
                f"Identity audit failed "
                f"for val_{sid}: "
                f"MSE={identity_mse:.8e}"
            )

        print()
        print("=" * 100)
        print(
            f"VAL {sid:02d}"
        )
        print("=" * 100)

        print(
            "original index =",
            mapping[sid][
                "original_index"
            ],
        )

        print(
            "speed range    =",
            float(
                target_480.min()
            ),
            float(
                target_480.max()
            ),
        )

        print(
            "480->256 MSE   =",
            f"{identity_mse:.10e}",
        )

        print(
            "480->256 MAE   =",
            f"{identity_mae:.10e}",
        )

        # -----------------------------------------------------
        # True CBS forward
        # -----------------------------------------------------

        dobs, wavefields = run_cbs(
            target_480=target_480,
            src_indices=src_indices,
            rec_indices=rec_indices,
            frequency=frequency,
            cbs_iters=cbs_iters,
            boundary_width=boundary_width,
            boundary_strength=boundary_strength,
            boundary_type=boundary_type,
            device=device,
        )

        if dobs.shape != (
            64,
            64,
        ):
            raise RuntimeError(
                f"Unexpected dobs shape: "
                f"{dobs.shape}"
            )

        if not (
            np.isfinite(
                dobs.real
            ).all()
            and np.isfinite(
                dobs.imag
            ).all()
        ):
            raise RuntimeError(
                f"Non-finite dobs: "
                f"val_{sid}"
            )

        raw_hash = sha256_file(
            raw_path
        )

        save_dict = dict(
            formal_val_id=np.array(
                [sid],
                dtype=np.int32,
            ),

            original_split=np.array(
                [
                    mapping[sid][
                        "original_split"
                    ]
                ]
            ),

            original_index=np.array(
                [
                    int(
                        mapping[sid][
                            "original_index"
                        ]
                    )
                ],
                dtype=np.int32,
            ),

            target_256=
                target_256.astype(
                    np.float32
                ),

            target_480=
                target_480.astype(
                    np.float32
                ),

            source_positions=
                source_positions.astype(
                    np.int64
                ),

            src_indices=
                src_indices.astype(
                    np.int64
                ),

            rec_indices=
                rec_indices.astype(
                    np.int64
                ),

            dobs_complex=
                dobs.astype(
                    np.complex64
                ),

            # Compatibility with old C5 readers.
            dobs_saved=
                dobs.astype(
                    np.complex64
                ),

            source_file=np.array(
                [str(raw_path)]
            ),

            measurement_mode=np.array(
                [measurement_mode]
            ),

            frequency=np.array(
                [frequency],
                dtype=np.float64,
            ),

            cbs_iters=np.array(
                [cbs_iters],
                dtype=np.int32,
            ),

            boundary_width=np.array(
                [boundary_width],
                dtype=np.int32,
            ),

            boundary_strength=np.array(
                [boundary_strength],
                dtype=np.float32,
            ),

            boundary_type=np.array(
                [boundary_type]
            ),

            raw_to_physics_transform=
                np.array(
                    ["transpose"]
                ),

            processed_resize=
                np.array(
                    ["bilinear"]
                ),

            raw480_to_processed_mse=
                np.array(
                    [identity_mse],
                    dtype=np.float64,
                ),

            raw480_to_processed_mae=
                np.array(
                    [identity_mae],
                    dtype=np.float64,
                ),

            raw_speed_sha256=np.array(
                [raw_hash]
            ),
        )

        if args.save_wavefields:
            save_dict[
                "wavefields"
            ] = wavefields

        np.savez_compressed(
            out_path,
            **save_dict,
        )

        print(
            "dobs shape      =",
            dobs.shape,
        )

        print(
            "dobs |.| range  =",
            float(
                np.abs(dobs).min()
            ),
            float(
                np.abs(dobs).max()
            ),
        )

        if args.save_wavefields:
            print(
                "wavefields     =",
                wavefields.shape,
            )

        print(
            "saved           =",
            out_path,
        )

        rows.append({
            "formal_val_id":
                sid,

            "original_index":
                int(
                    mapping[sid][
                        "original_index"
                    ]
                ),

            "output":
                str(out_path),

            "raw_speed_sha256":
                raw_hash,

            "identity_mse":
                identity_mse,

            "identity_mae":
                identity_mae,

            "dobs_abs_min":
                float(
                    np.abs(dobs).min()
                ),

            "dobs_abs_max":
                float(
                    np.abs(dobs).max()
                ),

            "save_wavefields":
                bool(
                    args.save_wavefields
                ),
        })

        del wavefields
        del dobs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Run manifest
    # ---------------------------------------------------------

    manifest = {
        "start_id":
            args.start_id,

        "end_id":
            args.end_id,

        "frequency":
            frequency,

        "num_sources":
            int(
                len(src_indices)
            ),

        "num_receivers":
            int(
                len(rec_indices)
            ),

        "cbs_iters":
            cbs_iters,

        "boundary_width":
            boundary_width,

        "boundary_strength":
            boundary_strength,

        "boundary_type":
            boundary_type,

        "measurement_mode":
            measurement_mode,

        "raw_to_physics_transform":
            "transpose",

        "save_wavefields":
            bool(
                args.save_wavefields
            ),

        "samples":
            rows,
    }

    manifest_path = (
        output_root
        / (
            f"manifest_"
            f"{args.start_id:02d}_"
            f"{args.end_id:02d}.json"
        )
    )

    with open(
        manifest_path,
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
    print("DONE")
    print("=" * 100)

    print(
        "manifest =",
        manifest_path,
    )


if __name__ == "__main__":
    main()
