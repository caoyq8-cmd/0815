import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cbs_model import ConvergentBornSeries_Batch


ROOT = Path(
    "/home/featurize/work/USCT_repro"
)

PHYS = (
    ROOT
    / "USCT_download/formal_phys20/"
      "cbs_data/dev10"
)

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


def resize256(x, transpose=False):

    if transpose:
        x = x.T

    t = torch.from_numpy(
        np.ascontiguousarray(x)
    ).float()[None, None]

    y = F.interpolate(
        t,
        size=(480,480),
        mode="bilinear",
        align_corners=False,
    )

    return (
        y[0,0]
        .numpy()
        .astype(np.float32)
    )


@torch.no_grad()
def forward(
    speed480,
    src,
    rec,
    frequency,
    iters,
    bw,
    bs,
    bt,
):

    sos = torch.from_numpy(
        np.ascontiguousarray(
            speed480
        )
    ).float().view(
        1,1,480,480
    ).to(DEVICE)

    model = ConvergentBornSeries_Batch(
        f=float(frequency),
        sos=sos,
        boundary_width=[
            int(bw),
            int(bw),
        ],
        boundary_strength=float(bs),
        boundary_type=str(bt),
        src_loc_set=src,
        device=str(DEVICE),
    )

    u = model(
        max_iters=int(iters)
    )

    r = torch.from_numpy(
        rec.astype(np.int64)
    ).long().to(DEVICE)

    d = (
        u[
            0,
            :,
            r[:,0],
            r[:,1],
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.complex64)
    )

    del u, model, sos, r

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return d


def rrmse(a, b):

    return float(
        np.linalg.norm(
            (a-b).reshape(-1)
        )
        /
        np.linalg.norm(
            b.reshape(-1)
        )
    )


p = PHYS / "test_1.npz"

z = np.load(
    p,
    allow_pickle=True,
)

gt480 = (
    z["target_480"]
    .astype(np.float32)
)

gt256 = (
    z["target_256"]
    .astype(np.float32)
)

obs = (
    z["dobs_complex"]
    .astype(np.complex64)
)

src = (
    z["src_indices"]
    .astype(np.int64)
)

rec = (
    z["rec_indices"]
    .astype(np.int64)
)

freq = float(z["frequency"][0])
iters = int(z["cbs_iters"][0])
bw = int(z["boundary_width"][0])
bs = float(z["boundary_strength"][0])
bt = str(z["boundary_type"][0])


print("=" * 100)
print("FORMAL PHYS COORDINATE AUDIT")
print("=" * 100)

print("device =", DEVICE)

# ---------------------------------------------------------
# A. Exact 480 replay
# ---------------------------------------------------------

d_exact = forward(
    gt480,
    src, rec,
    freq, iters,
    bw, bs, bt,
)

rr_exact = rrmse(
    d_exact,
    obs,
)

print()
print(
    "A. exact GT480 replay"
)
print(
    "   RRMSE =",
    f"{rr_exact:.12e}"
)

# ---------------------------------------------------------
# B. GT256 -> 480, NO transpose
# ---------------------------------------------------------

gt480_direct = resize256(
    gt256,
    transpose=False,
)

d_direct = forward(
    gt480_direct,
    src, rec,
    freq, iters,
    bw, bs, bt,
)

rr_direct = rrmse(
    d_direct,
    obs,
)

print()
print(
    "B. GT256 -> 480 direct"
)
print(
    "   image MSE vs GT480 =",
    float(
        np.mean(
            (
                gt480_direct
                - gt480
            ) ** 2
        )
    )
)
print(
    "   CBS RRMSE =",
    f"{rr_direct:.12e}"
)

# ---------------------------------------------------------
# C. GT256.T -> 480
# ---------------------------------------------------------

gt480_transpose = resize256(
    gt256,
    transpose=True,
)

d_transpose = forward(
    gt480_transpose,
    src, rec,
    freq, iters,
    bw, bs, bt,
)

rr_transpose = rrmse(
    d_transpose,
    obs,
)

print()
print(
    "C. GT256 transpose -> 480"
)
print(
    "   image MSE vs GT480 =",
    float(
        np.mean(
            (
                gt480_transpose
                - gt480
            ) ** 2
        )
    )
)
print(
    "   CBS RRMSE =",
    f"{rr_transpose:.12e}"
)

print()
print("=" * 100)
print("DECISION")
print("=" * 100)

if (
    rr_exact < 1e-5
    and
    rr_direct < rr_transpose
):

    print(
        "[PASS] formal 256 arrays must be "
        "upsampled DIRECTLY to CBS 480."
    )

elif (
    rr_exact < 1e-5
    and
    rr_transpose < rr_direct
):

    print(
        "[PASS] transpose mapping is required."
    )

else:

    print(
        "[WARNING] coordinate convention "
        "still requires investigation."
    )
