import numpy as np
from pathlib import Path

from c5_audit_mgno_unseen64_sources import (
    solve_cbs_chunked,
)

from c5_directional_fd_audit import scalar


UROOT = Path(
    "/home/featurize/work/USCT_repro/USCT_download"
)

sample_path = (
    UROOT /
    "c6_runs/"
    "c6_2_c4_to_cbs_bridge_val10_final/"
    "test_1.npz"
)

grad_path = (
    UROOT /
    "c6_runs/"
    "c6_3a_real_c4_gradient_test1/"
    "decompose_test1.npz"
)


z = np.load(
    sample_path,
    allow_pickle=True,
)

d = np.load(
    grad_path,
    allow_pickle=True,
)


candidate = z[
    "candidate_480"
].astype(np.float32)

src_indices = z[
    "src_indices"
].astype(np.int64)

rec_indices = z[
    "rec_indices"
].astype(np.int64)

source_positions = z[
    "source_positions"
].astype(np.int64)

dobs = z[
    "dobs_complex"
].astype(np.complex64)


frequency = float(
    scalar(z["frequency"])
)

cbs_iters = int(
    scalar(z["cbs_iters"])
)

boundary_width = int(
    scalar(z["boundary_width"])
)

boundary_strength = float(
    scalar(z["boundary_strength"])
)

boundary_type = str(
    scalar(z["boundary_type"])
)


print("=" * 110)
print("C6.4A-0 TRUE8 RESIDUAL EQUIVALENCE — TEST1")
print("=" * 110)


# ------------------------------------------------------------
# Geometry identity
# ------------------------------------------------------------

expected_src = rec_indices[
    source_positions
]

print()
print(
    "src_indices == rec_indices[source_positions] =",
    np.array_equal(
        src_indices,
        expected_src,
    )
)

print(
    "source_positions =",
    source_positions.tolist(),
)

if not np.array_equal(
    src_indices,
    expected_src,
):
    raise RuntimeError(
        "Measurement-source geometry mismatch."
    )


# ------------------------------------------------------------
# Solve only the 8 physical transmitter wavefields
# ------------------------------------------------------------

print()
print(
    "Solving true CBS for only 8 measurement sources..."
)

true8 = solve_cbs_chunked(
    speed=candidate,
    source_indices=src_indices,
    frequency=frequency,
    cbs_iters=cbs_iters,
    boundary_width=boundary_width,
    boundary_strength=boundary_strength,
    boundary_type=boundary_type,
    device="cuda:0",
    chunk_size=8,
)

true8 = np.asarray(
    true8,
    dtype=np.complex64,
)


# ------------------------------------------------------------
# Compare against transmitter fields extracted from true64
# ------------------------------------------------------------

true64 = d[
    "true64"
].astype(np.complex64)

true8_from64 = true64[
    source_positions
]


def rrmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    return float(
        np.sqrt(
            np.mean(
                np.abs(a-b)**2
            )
        )
        /
        (
            np.sqrt(
                np.mean(
                    np.abs(b)**2
                )
            )
            + 1e-12
        )
    )


field_rr = rrmse(
    true8,
    true8_from64,
)

field_max = float(
    np.max(
        np.abs(
            true8 -
            true8_from64
        )
    )
)


rr = rec_indices[:, 0]
cc = rec_indices[:, 1]


meas8 = true8[
    :,
    rr,
    cc,
]

meas64 = true8_from64[
    :,
    rr,
    cc,
]


res8 = (
    meas8 -
    dobs
)

res64 = (
    meas64 -
    dobs
)


measurement_rr = rrmse(
    meas8,
    meas64,
)

residual_rr = rrmse(
    res8,
    res64,
)


print()
print(
    "true8 vs true64 TX field RRMSE =",
    f"{field_rr:.10e}",
)

print(
    "true8 vs true64 TX maxabs      =",
    f"{field_max:.10e}",
)

print(
    "measurement RRMSE              =",
    f"{measurement_rr:.10e}",
)

print(
    "residual RRMSE                 =",
    f"{residual_rr:.10e}",
)


if (
    field_rr < 1e-6
    and
    measurement_rr < 1e-6
    and
    residual_rr < 1e-6
):
    print()
    print(
        "[STRONG PASS] "
        "8-source CBS reproduces the exact "
        "measurement residual used by the "
        "64-source reference."
    )

else:
    print()
    print(
        "[FAIL] True8 and true64 transmitter "
        "solutions are not equivalent."
    )

    raise SystemExit(2)
