import argparse
import csv
import json
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import torch

from cbs_model import ConvergentBornSeries_Batch


# ============================================================
# Utilities
# ============================================================

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


def scalar_from_npz(z, key, default=None):

    if key not in z.files:

        if default is not None:
            return default

        raise KeyError(key)

    x = np.asarray(z[key])

    if x.size != 1:
        raise ValueError(
            f"{key} is not scalar: {x.shape}"
        )

    return x.reshape(-1)[0].item()


def string_scalar_from_npz(
    z,
    key,
    default=None,
):

    if key not in z.files:

        if default is not None:
            return default

        raise KeyError(key)

    x = np.asarray(z[key])

    if x.size == 0:
        return default

    return str(
        x.reshape(-1)[0]
    )


def validate_speed(x, sid):

    x = np.asarray(x)

    if x.shape != (480, 480):

        raise RuntimeError(
            f"test_{sid}: "
            f"unexpected raw shape {x.shape}"
        )

    if not np.isfinite(x).all():

        raise RuntimeError(
            f"test_{sid}: raw contains NaN/Inf"
        )

    # 宽松物理范围，仅用于抓明显错误
    if x.min() < 1200 or x.max() > 1800:

        raise RuntimeError(
            f"test_{sid}: suspicious speed range "
            f"[{x.min()}, {x.max()}]"
        )


def detect_geometry_convention(
    processed,
    reference_geometry,
):

    if (
        "x_pos" not in processed
        or "y_pos" not in processed
    ):
        return "not_available"

    x = (
        np.asarray(processed["x_pos"])
        .reshape(-1)
        .astype(np.int64)
    )

    y = (
        np.asarray(processed["y_pos"])
        .reshape(-1)
        .astype(np.int64)
    )

    if len(x) != 256 or len(y) != 256:
        return "unexpected_xy_size"

    # 256 transducers -> every 4th -> 64
    candidates = {
        "x_y_every4":
            np.stack(
                [x[::4], y[::4]],
                axis=1,
            ),

        "y_x_every4":
            np.stack(
                [y[::4], x[::4]],
                axis=1,
            ),
    }

    for name, c in candidates.items():

        if (
            c.shape == reference_geometry.shape
            and np.array_equal(
                c,
                reference_geometry,
            )
        ):
            return name

    return "NO_EXACT_MATCH"


# ============================================================
# CBS
# ============================================================

@torch.no_grad()
def run_cbs(
    target_480,
    src_selected,
    rec_indices,
    frequency,
    cbs_iters,
    boundary_width,
    boundary_strength,
    boundary_type,
    device,
    save_wavefields,
):

    sos = torch.from_numpy(
        np.ascontiguousarray(
            target_480.astype(np.float32)
        )
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

        src_loc_set=src_selected,

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

    wavefields = None

    if save_wavefields:

        wavefields = (
            u[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.complex64)
        )

    del u
    del model
    del sos

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dobs, wavefields


# ============================================================
# Main
# ============================================================

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
        default=400,
    )

    ap.add_argument(
        "--source_positions",
        default="0,8,16,24,32,40,48,56",
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


    if not raw_root.exists():
        raise FileNotFoundError(raw_root)

    if not processed_root.exists():
        raise FileNotFoundError(
            processed_root
        )


    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )


    # --------------------------------------------------------
    # Load frozen C5/C6 physics geometry
    # --------------------------------------------------------

    ref = np.load(
        args.reference_npz,
        allow_pickle=True,
    )

    src_all = (
        ref["src_indices"]
        .astype(np.int64)
    )

    rec_indices = (
        ref["rec_indices"]
        .astype(np.int64)
    )


    if rec_indices.shape != (64, 2):

        raise RuntimeError(
            f"Expected 64 receiver locations, "
            f"got {rec_indices.shape}"
        )


    requested_source_positions = np.asarray(
        [
            int(x)
            for x in
            args.source_positions.split(",")
        ],
        dtype=np.int64,
    )


    if requested_source_positions.shape != (8,):

        raise RuntimeError(
            "Formal CBS8 protocol requires "
            "exactly 8 source positions."
        )


    # --------------------------------------------------------
    # Freeze source geometry.
    #
    # Reference may contain:
    #
    # 1) full 64-source geometry
    # 2) native sparse-8 geometry
    #
    # For native sparse-8, infer each source's original
    # 64-transducer index by exact coordinate matching against
    # rec_indices. No guessing is allowed.
    # --------------------------------------------------------

    if src_all.shape == (64, 2):

        source_positions = (
            requested_source_positions.copy()
        )

        src_selected = (
            src_all[source_positions]
        )

        reference_source_mode = (
            "full64_selected8"
        )


    elif src_all.shape == (8, 2):

        inferred_positions = []

        for j, src_xy in enumerate(src_all):

            hits = np.where(
                np.all(
                    rec_indices
                    ==
                    src_xy.reshape(1, 2),
                    axis=1,
                )
            )[0]

            if len(hits) != 1:

                raise RuntimeError(
                    f"Source {j} coordinate "
                    f"{src_xy.tolist()} "
                    f"has {len(hits)} exact matches "
                    f"in rec_indices; expected exactly 1."
                )

            inferred_positions.append(
                int(hits[0])
            )


        source_positions = np.asarray(
            inferred_positions,
            dtype=np.int64,
        )


        print(
            "inferred source positions =",
            source_positions.tolist(),
        )


        if not np.array_equal(
            source_positions,
            requested_source_positions,
        ):

            raise RuntimeError(
                "Sparse-8 source geometry does not "
                "match the frozen requested protocol.\n"
                f"inferred  = "
                f"{source_positions.tolist()}\n"
                f"requested = "
                f"{requested_source_positions.tolist()}"
            )


        src_selected = (
            src_all.copy()
        )

        reference_source_mode = (
            "native_sparse8_inferred_from_rec_indices"
        )


    else:

        raise RuntimeError(
            "Unsupported reference source geometry: "
            f"{src_all.shape}. "
            "Expected (64,2) or (8,2)."
        )


    # --------------------------------------------------------
    # Frozen physics parameters
    # --------------------------------------------------------

    frequency = float(
        scalar_from_npz(
            ref,
            "frequency",
            500000.0,
        )
    )

    cbs_iters = int(
        scalar_from_npz(
            ref,
            "cbs_iters",
            80,
        )
    )

    boundary_width = int(
        scalar_from_npz(
            ref,
            "boundary_width",
            300,
        )
    )

    boundary_strength = float(
        scalar_from_npz(
            ref,
            "boundary_strength",
            225.0,
        )
    )

    boundary_type = (
        string_scalar_from_npz(
            ref,
            "boundary_type",
            "PML3",
        )
    )


    # --------------------------------------------------------
    # Freeze protocol
    # --------------------------------------------------------

    print("=" * 110)
    print("FORMAL VAL400 CBS8 GENERATOR")
    print("=" * 110)

    print("raw root           =", raw_root)
    print("processed root     =", processed_root)
    print("reference          =", args.reference_npz)
    print("output             =", output_root)

    print("IDs                =",
          args.start_id,
          "to",
          args.end_id)

    print("device             =", device)
    print("source positions   =", source_positions.tolist())
    print("src selected       =", src_selected.tolist())
    print("num receivers      =", len(rec_indices))

    print("frequency          =", frequency)
    print("CBS iters          =", cbs_iters)
    print("boundary width     =", boundary_width)
    print("boundary strength  =", boundary_strength)
    print("boundary type      =", boundary_type)

    print("raw transform      = identity")
    print("processed resize   = MATLAB bilinear")
    print("save wavefields    =", args.save_wavefields)

    print()


    # --------------------------------------------------------
    # Geometry audit using processed test_1
    # --------------------------------------------------------

    p1 = (
        processed_root /
        "val_1.mat"
    )

    proc1 = load_mat_auto(p1)

    geometry_convention = (
        detect_geometry_convention(
            proc1,
            rec_indices,
        )
    )

    print(
        "processed/reference geometry =",
        geometry_convention,
    )

    if geometry_convention == "NO_EXACT_MATCH":

        raise RuntimeError(
            "Processed x_pos/y_pos do not "
            "match frozen C5 receiver geometry."
        )


    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    rows = []

    for sid in range(
        args.start_id,
        args.end_id + 1,
    ):

        raw_path = (
            raw_root /
            f"test_{sid}.mat"
        )

        processed_path = (
            processed_root /
            f"val_{sid}.mat"
        )

        out_path = (
            output_root /
            f"val_{sid}.npz"
        )


        if out_path.exists() and not args.overwrite:

            print(
                f"[SKIP] test_{sid}: "
                f"{out_path.name}"
            )

            rows.append({
                "val_id": sid,
                "status": "skipped_existing",
                "output": str(out_path),
            })

            continue


        if not raw_path.exists():
            raise FileNotFoundError(
                raw_path
            )

        if not processed_path.exists():
            raise FileNotFoundError(
                processed_path
            )


        raw = load_mat_auto(
            raw_path
        )

        proc = load_mat_auto(
            processed_path
        )


        if "slice" not in raw:
            raise RuntimeError(
                f"test_{sid}: "
                f"raw file missing slice"
            )

        if "target_256" not in proc:
            raise RuntimeError(
                f"test_{sid}: "
                f"processed file missing target_256"
            )


        # ====================================================
        # IMPORTANT:
        # Formal TEST400 has been independently audited:
        #
        # raw slice
        #   -- identity + MATLAB bilinear -->
        # processed target_256
        #
        # 400/400 exact, MSE=0.
        #
        # Therefore NO transpose here.
        # ====================================================

        target_480 = (
            np.asarray(
                raw["slice"]
            )
            .squeeze()
            .astype(np.float32)
        )

        target_256 = (
            np.asarray(
                proc["target_256"]
            )
            .squeeze()
            .astype(np.float32)
        )


        validate_speed(
            target_480,
            sid,
        )


        if target_256.shape != (256, 256):

            raise RuntimeError(
                f"test_{sid}: "
                f"target_256 shape "
                f"{target_256.shape}"
            )


        geometry_here = (
            detect_geometry_convention(
                proc,
                rec_indices,
            )
        )


        if (
            geometry_here
            != geometry_convention
        ):

            raise RuntimeError(
                f"test_{sid}: geometry changed: "
                f"{geometry_here} vs "
                f"{geometry_convention}"
            )


        print()
        print("=" * 110)

        print(
            f"TEST {sid:03d} "
            f"[{sid - args.start_id + 1}/"
            f"{args.end_id - args.start_id + 1}]"
        )

        print(
            "speed range       =",
            float(target_480.min()),
            float(target_480.max()),
        )

        print(
            "target256 range   =",
            float(target_256.min()),
            float(target_256.max()),
        )


        t0 = time.time()

        dobs, wavefields = run_cbs(
            target_480=target_480,
            src_selected=src_selected,
            rec_indices=rec_indices,
            frequency=frequency,
            cbs_iters=cbs_iters,
            boundary_width=boundary_width,
            boundary_strength=
                boundary_strength,
            boundary_type=boundary_type,
            device=device,
            save_wavefields=
                args.save_wavefields,
        )

        elapsed = (
            time.time() - t0
        )


        expected_shape = (
            len(source_positions),
            len(rec_indices),
        )

        if dobs.shape != expected_shape:

            raise RuntimeError(
                f"test_{sid}: "
                f"unexpected dobs shape "
                f"{dobs.shape}, "
                f"expected {expected_shape}"
            )


        if not (
            np.isfinite(dobs.real).all()
            and
            np.isfinite(dobs.imag).all()
        ):

            raise RuntimeError(
                f"test_{sid}: "
                f"non-finite dobs"
            )


        abs_dobs = np.abs(dobs)


        save_dict = {

            "formal_val_id":
                np.array(
                    [sid],
                    dtype=np.int32,
                ),

            "target_256":
                target_256.astype(
                    np.float32
                ),

            "target_480":
                target_480.astype(
                    np.float32
                ),

            "source_positions":
                source_positions.astype(
                    np.int64
                ),

            "src_indices":
                src_selected.astype(
                    np.int64
                ),

            "rec_indices":
                rec_indices.astype(
                    np.int64
                ),

            "dobs_complex":
                dobs.astype(
                    np.complex64
                ),

            # Compatibility alias for
            # existing C5/C6 evaluation code.
            "dobs_saved":
                dobs.astype(
                    np.complex64
                ),

            "source_file":
                np.array([
                    str(raw_path)
                ]),

            "processed_file":
                np.array([
                    str(processed_path)
                ]),

            "measurement_mode":
                np.array([
                    "formal_val400_sparse8"
                ]),

            "frequency":
                np.array(
                    [frequency],
                    dtype=np.float64,
                ),

            "cbs_iters":
                np.array(
                    [cbs_iters],
                    dtype=np.int32,
                ),

            "boundary_width":
                np.array(
                    [boundary_width],
                    dtype=np.int32,
                ),

            "boundary_strength":
                np.array(
                    [boundary_strength],
                    dtype=np.float64,
                ),

            "boundary_type":
                np.array([
                    boundary_type
                ]),

            "raw_to_physics_transform":
                np.array([
                    "identity"
                ]),

            "raw_to_processed_transform":
                np.array([
                    "identity"
                ]),

            "processed_resize":
                np.array([
                    "MATLAB_bilinear"
                ]),

            "raw_processed_identity":
                np.array([
                    "400_of_400_exact_MSE_0"
                ]),

            "geometry_convention":
                np.array([
                    geometry_convention
                ]),
        }


        if wavefields is not None:

            save_dict[
                "wavefields"
            ] = wavefields


        np.savez_compressed(
            out_path,
            **save_dict,
        )


        print(
            "dobs shape        =",
            dobs.shape,
        )

        print(
            "dobs |.| min/max =",
            float(abs_dobs.min()),
            float(abs_dobs.max()),
        )

        print(
            "dobs |.| mean/std =",
            float(abs_dobs.mean()),
            float(abs_dobs.std()),
        )

        print(
            "elapsed sec       =",
            elapsed,
        )

        print(
            "saved             =",
            out_path,
        )


        rows.append({

            "val_id":
                sid,

            "status":
                "ok",

            "elapsed_sec":
                elapsed,

            "speed_min":
                float(target_480.min()),

            "speed_max":
                float(target_480.max()),

            "dobs_abs_min":
                float(abs_dobs.min()),

            "dobs_abs_max":
                float(abs_dobs.max()),

            "dobs_abs_mean":
                float(abs_dobs.mean()),

            "dobs_abs_std":
                float(abs_dobs.std()),

            "output":
                str(out_path),
        })


    # --------------------------------------------------------
    # Save run manifest
    # --------------------------------------------------------

    csv_path = (
        output_root /
        "run_manifest.csv"
    )

    fields = sorted(
        set().union(
            *[
                set(r.keys())
                for r in rows
            ]
        )
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        w.writeheader()
        w.writerows(rows)


    summary = {

        "protocol":
            "formal_TEST400_CBS8",

        "start_id":
            args.start_id,

        "end_id":
            args.end_id,

        "source_positions":
            source_positions.tolist(),

        "num_sources":
            int(
                len(source_positions)
            ),

        "num_receivers":
            int(
                len(rec_indices)
            ),

        "frequency":
            frequency,

        "cbs_iters":
            cbs_iters,

        "boundary_width":
            boundary_width,

        "boundary_strength":
            boundary_strength,

        "boundary_type":
            boundary_type,

        "raw_to_physics_transform":
            "identity",

        "raw_to_processed_transform":
            "identity",

        "processed_resize":
            "MATLAB bilinear",

        "raw_processed_identity":
            "400/400 exact, MSE=0",

        "geometry_convention":
            geometry_convention,

        "reference_npz":
            str(args.reference_npz),

        "save_wavefields":
            bool(
                args.save_wavefields
            ),
    }


    json_path = (
        output_root /
        "run_summary.json"
    )

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


    print()
    print("=" * 110)
    print("DONE")
    print("=" * 110)

    print(
        "manifest =",
        csv_path,
    )

    print(
        "summary  =",
        json_path,
    )


if __name__ == "__main__":
    main()
