from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PHYS = Path(
    "/home/featurize/work/USCT_repro/USCT_download/"
    "formal_val400_physics/cbs8"
)

CACHE = Path(
    "/home/featurize/work/USCT_repro/USCT_download/"
    "condition_cache/inversionnet_formal_3600_400_400_e89/val"
)

OUT = Path(
    "/home/featurize/work/USCT_repro/USCT_download/"
    "formal_val400_bridge_inversionnet"
)

OUT.mkdir(parents=True, exist_ok=True)


def upsample_256_to_480(x):
    t = torch.from_numpy(
        x.astype(np.float32)
    )[None, None]

    y = F.interpolate(
        t,
        size=(480, 480),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    return np.clip(
        y,
        1400.0,
        1605.0,
    ).astype(np.float32)


def scalar(z, key, default=None):
    if key not in z.files:
        return default

    x = np.asarray(z[key])

    if x.size == 0:
        return default

    return x.reshape(-1)[0]


max_target_mse = 0.0

for sid in range(1, 401):

    pp = PHYS / f"val_{sid}.npz"
    cp = CACHE / f"val_{sid}.npz"

    if not pp.exists():
        raise FileNotFoundError(pp)

    if not cp.exists():
        raise FileNotFoundError(cp)

    p = np.load(pp, allow_pickle=True)
    c = np.load(cp, allow_pickle=True)

    target_256_phys = (
        p["target_256"]
        .astype(np.float32)
    )

    target_256_cache = (
        c["target_speed"][0]
        .astype(np.float32)
    )

    err = (
        target_256_phys.astype(np.float64)
        -
        target_256_cache.astype(np.float64)
    )

    mse = float(np.mean(err ** 2))
    max_target_mse = max(max_target_mse, mse)

    if mse > 1e-8:
        raise RuntimeError(
            f"VAL {sid}: target mismatch MSE={mse}"
        )

    condition_256 = (
        c["condition_speed"][0]
        .astype(np.float32)
    )

    candidate_480 = (
        upsample_256_to_480(
            condition_256
        )
    )

    target_480 = (
        p["target_480"]
        .astype(np.float32)
    )

    dobs = (
        p["dobs_complex"]
        .astype(np.complex64)
    )

    source_positions = (
        p["source_positions"]
        .astype(np.int64)
    )

    src_indices = (
        p["src_indices"]
        .astype(np.int64)
    )

    rec_indices = (
        p["rec_indices"]
        .astype(np.int64)
    )

    if candidate_480.shape != (480, 480):
        raise RuntimeError(
            f"VAL {sid}: bad candidate shape"
        )

    if dobs.shape != (8, 64):
        raise RuntimeError(
            f"VAL {sid}: bad dobs shape {dobs.shape}"
        )

    out = OUT / f"val_{sid}.npz"

    np.savez_compressed(
        out,

        formal_val_id=np.array(
            [sid],
            dtype=np.int32,
        ),

        candidate_256=condition_256,
        candidate_480=candidate_480,

        target_256=target_256_phys,
        target_480=target_480,

        dobs_complex=dobs,

        source_positions=source_positions,
        src_indices=src_indices,
        rec_indices=rec_indices,

        frequency=np.array(
            [
                float(
                    scalar(
                        p,
                        "frequency",
                        500000.0,
                    )
                )
            ],
            dtype=np.float64,
        ),

        cbs_iters=np.array(
            [
                int(
                    scalar(
                        p,
                        "cbs_iters",
                        80,
                    )
                )
            ],
            dtype=np.int32,
        ),

        boundary_width=np.array(
            [
                int(
                    scalar(
                        p,
                        "boundary_width",
                        300,
                    )
                )
            ],
            dtype=np.int32,
        ),

        boundary_strength=np.array(
            [
                float(
                    scalar(
                        p,
                        "boundary_strength",
                        225.0,
                    )
                )
            ],
            dtype=np.float64,
        ),

        boundary_type=np.array(
            [
                str(
                    scalar(
                        p,
                        "boundary_type",
                        "PML3",
                    )
                )
            ]
        ),

        candidate_source=np.array(
            ["InversionNet_e89"]
        ),

        coordinate_protocol=np.array(
            [
                "formal_val400_identity_"
                "bilinear256to480"
            ]
        ),
    )

    # Compatibility alias for old C6.4A script.
    alias = OUT / f"test_{sid}.npz"

    if alias.exists() or alias.is_symlink():
        alias.unlink()

    alias.symlink_to(out.name)

    if sid <= 3 or sid % 50 == 0:
        print(
            f"VAL {sid:03d} | "
            f"target MSE={mse:.3e} | "
            f"candidate480="
            f"[{candidate_480.min():.2f},"
            f"{candidate_480.max():.2f}]"
        )

print()
print("max target MSE =", max_target_mse)
print("saved =", OUT)
print("[PASS] FORMAL VAL400 BRIDGE BUILT")
