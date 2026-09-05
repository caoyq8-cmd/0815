import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cbs_model import ConvergentBornSeries_Batch
from c5_mgno_paper import MgNOBackgroundWavefield


def scalar(v):
    return np.asarray(v).reshape(-1)[0].item()


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


def stats(x):
    x = np.asarray(x, dtype=np.float64)

    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def solve_cbs_chunked(
    speed,
    source_indices,
    frequency,
    cbs_iters,
    boundary_width,
    boundary_strength,
    boundary_type,
    device,
    chunk_size=8,
):
    waves = []

    for start in range(
        0,
        len(source_indices),
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            len(source_indices),
        )

        src = source_indices[
            start:end
        ]

        print(
            f"[CBS] sources "
            f"{start:02d}:{end:02d} "
            f"/ {len(source_indices)}"
        )

        sos = (
            torch
            .from_numpy(
                speed.astype(
                    np.float32
                )
            )
            .float()[
                None,
                None,
            ]
            .to(device)
        )

        solver = ConvergentBornSeries_Batch(
            f=frequency,
            sos=sos,
            boundary_width=[
                boundary_width,
                boundary_width,
            ],
            boundary_strength=
                boundary_strength,
            boundary_type=
                boundary_type,
            src_loc_set=
                src,
            device=
                device,
        )

        with torch.no_grad():
            u = solver(
                max_iters=
                    cbs_iters
            )

        u = (
            u[0]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.complex64
            )
        )

        waves.append(u)

        del solver
        del sos
        del u

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.concatenate(
        waves,
        axis=0,
    )


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

    return (
        model,
        channels,
        recurrent_iters,
    )


def neural_predict(
    model,
    speed,
    backgrounds,
    speed_mean,
    speed_std,
    wave_scale,
    device,
):
    speed_t = torch.from_numpy(
        speed.astype(
            np.float32
        )
    ).to(device)

    speed_norm = (
        speed_t
        - speed_mean
    ) / speed_std

    outputs = []

    with torch.no_grad():

        for s in range(
            len(backgrounds)
        ):
            bg = torch.from_numpy(
                backgrounds[s]
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

            pred = (
                out[0]
                + 1j * out[1]
            ) * wave_scale

            outputs.append(
                pred.cpu().numpy()
            )

            if (
                (s + 1) % 8 == 0
                or s + 1
                == len(backgrounds)
            ):
                print(
                    f"[MgNO] predicted "
                    f"{s+1}/{len(backgrounds)}"
                )

    return np.stack(
        outputs,
        axis=0,
    ).astype(
        np.complex64
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--sample",
        required=True,
    )

    ap.add_argument(
        "--mgno_ckpt",
        required=True,
    )

    ap.add_argument(
        "--existing_background",
        required=True,
    )

    ap.add_argument(
        "--rho",
        type=float,
        default=0.8,
    )

    ap.add_argument(
        "--background_speed",
        type=float,
        default=1500.0,
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
        "--chunk_size",
        type=int,
        default=8,
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

    z = np.load(
        args.sample,
        allow_pickle=True,
    )

    gt = z[
        "target_480"
    ].astype(
        np.float32
    )

    seen_src = z[
        "src_indices"
    ].astype(
        np.int64
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
        scalar(
            z["frequency"]
        )
    )

    cbs_iters = int(
        scalar(
            z["cbs_iters"]
        )
    )

    boundary_width = int(
        scalar(
            z["boundary_width"]
        )
    )

    boundary_strength = float(
        scalar(
            z["boundary_strength"]
        )
    )

    boundary_type = str(
        scalar(
            z["boundary_type"]
        )
    )

    candidate = (
        args.rho * gt
        +
        (
            1.0 - args.rho
        )
        * args.background_speed
    ).astype(
        np.float32
    )

    homogeneous = np.full_like(
        gt,
        args.background_speed,
        dtype=np.float32,
    )

    print("=" * 110)
    print("C5.4a MgNO-I 64-SOURCE LOCATION GENERALIZATION AUDIT")
    print("=" * 110)

    print(
        "sample            =",
        args.sample,
    )

    print(
        "seen sources      =",
        len(seen_src),
    )

    print(
        "all receiver srcs =",
        len(rec_indices),
    )

    print(
        "source_positions  =",
        source_positions.tolist(),
    )

    print(
        "candidate range   =",
        float(candidate.min()),
        float(candidate.max()),
    )

    print(
        "CBS config        =",
        frequency,
        cbs_iters,
        boundary_width,
        boundary_strength,
        boundary_type,
    )

    print()
    print("#" * 110)
    print("1. 64-SOURCE HOMOGENEOUS BACKGROUND")
    print("#" * 110)

    bg64 = solve_cbs_chunked(
        speed=homogeneous,
        source_indices=
            rec_indices,
        frequency=
            frequency,
        cbs_iters=
            cbs_iters,
        boundary_width=
            boundary_width,
        boundary_strength=
            boundary_strength,
        boundary_type=
            boundary_type,
        device=
            args.device,
        chunk_size=
            args.chunk_size,
    )

    # ------------------------------------------------------------
    # Verify generated 64-background against existing seen8 BG.
    # ------------------------------------------------------------

    bz = np.load(
        args.existing_background,
        allow_pickle=True,
    )

    old_bg = bz[
        "background_field"
    ].astype(
        np.complex64
    )

    old_src = bz[
        "src_indices"
    ].astype(
        np.int64
    )

    bg_align = []

    print()
    print("SEEN-8 BACKGROUND ALIGNMENT")

    for j, src in enumerate(
        old_src
    ):
        idx = np.where(
            np.all(
                rec_indices
                == src[None],
                axis=1,
            )
        )[0]

        if len(idx) != 1:
            raise RuntimeError(
                f"Could not map "
                f"source {src}"
            )

        k = int(idx[0])

        e = rrmse(
            bg64[k],
            old_bg[j],
        )

        bg_align.append(e)

        print(
            f"seen {j:02d} "
            f"-> source64 {k:02d} | "
            f"RRMSE={e:.8e}"
        )

    print(
        "background alignment "
        "mean/max =",
        f"{np.mean(bg_align):.8e}",
        f"{np.max(bg_align):.8e}",
    )

    print()
    print("#" * 110)
    print("2. EXACT 64-SOURCE CANDIDATE WAVEFIELDS")
    print("#" * 110)

    true64 = solve_cbs_chunked(
        speed=candidate,
        source_indices=
            rec_indices,
        frequency=
            frequency,
        cbs_iters=
            cbs_iters,
        boundary_width=
            boundary_width,
        boundary_strength=
            boundary_strength,
        boundary_type=
            boundary_type,
        device=
            args.device,
        chunk_size=
            args.chunk_size,
    )

    print()
    print("#" * 110)
    print("3. MgNO-I 64-SOURCE PREDICTION")
    print("#" * 110)

    (
        model,
        channels,
        vcycles,
    ) = load_mgno(
        args.mgno_ckpt,
        args.device,
    )

    print(
        "MgNO config =",
        f"C{channels}/V{vcycles}",
    )

    pred64 = neural_predict(
        model=model,
        speed=candidate,
        backgrounds=bg64,
        speed_mean=
            args.speed_mean,
        speed_std=
            args.speed_std,
        wave_scale=
            args.wave_scale,
        device=
            args.device,
    )

    rr = rec_indices[:, 0]
    cc = rec_indices[:, 1]

    seen_mask = np.zeros(
        64,
        dtype=bool,
    )

    # source_positions should be [0,8,...,56].
    seen_mask[
        source_positions
    ] = True

    unseen_mask = ~seen_mask

    full_err = []
    rec_err = []

    print()
    print("=" * 110)
    print("PER-SOURCE RESULTS")
    print("=" * 110)

    for s in range(64):

        ef = rrmse(
            pred64[s],
            true64[s],
        )

        er = rrmse(
            pred64[
                s,
                rr,
                cc,
            ],
            true64[
                s,
                rr,
                cc,
            ],
        )

        full_err.append(ef)
        rec_err.append(er)

        tag = (
            "SEEN"
            if seen_mask[s]
            else "UNSEEN"
        )

        print(
            f"source {s:02d} | "
            f"{tag:6s} | "
            f"full={ef:.6f} | "
            f"rec={er:.6f}"
        )

    full_err = np.asarray(
        full_err
    )

    rec_err = np.asarray(
        rec_err
    )

    results = {
        "channels":
            channels,

        "vcycles":
            vcycles,

        "background_alignment":
            stats(bg_align),

        "all64_full":
            stats(full_err),

        "all64_rec":
            stats(rec_err),

        "seen8_full":
            stats(
                full_err[
                    seen_mask
                ]
            ),

        "seen8_rec":
            stats(
                rec_err[
                    seen_mask
                ]
            ),

        "unseen56_full":
            stats(
                full_err[
                    unseen_mask
                ]
            ),

        "unseen56_rec":
            stats(
                rec_err[
                    unseen_mask
                ]
            ),
    }

    print()
    print("=" * 110)
    print("C5.4a FINAL SUMMARY")
    print("=" * 110)

    for group in [
        "seen8",
        "unseen56",
        "all64",
    ]:

        f = results[
            f"{group}_full"
        ]

        r = results[
            f"{group}_rec"
        ]

        print(
            f"{group:10s} | "
            f"full="
            f"{f['mean']:.6f}"
            f"±{f['std']:.6f} | "
            f"rec="
            f"{r['mean']:.6f}"
            f"±{r['std']:.6f}"
        )

    seen_full = results[
        "seen8_full"
    ]["mean"]

    unseen_full = results[
        "unseen56_full"
    ]["mean"]

    seen_rec = results[
        "seen8_rec"
    ]["mean"]

    unseen_rec = results[
        "unseen56_rec"
    ]["mean"]

    full_gap = (
        unseen_full
        / seen_full
        - 1.0
    )

    rec_gap = (
        unseen_rec
        / seen_rec
        - 1.0
    )

    results[
        "unseen_vs_seen_full_gap"
    ] = full_gap

    results[
        "unseen_vs_seen_rec_gap"
    ] = rec_gap

    print()
    print(
        "unseen/seen full gap =",
        f"{100*full_gap:+.2f}%"
    )

    print(
        "unseen/seen rec gap  =",
        f"{100*rec_gap:+.2f}%"
    )

    print()

    if (
        full_gap < 0.20
        and rec_gap < 0.20
    ):
        print(
            "[PASS] source-location "
            "generalization is adequate "
            "for a paper-style ANO pilot."
        )
    else:
        print(
            "[FAIL] unseen receiver-as-source "
            "locations degrade substantially; "
            "expand operator source training "
            "before ANO construction."
        )

    out = Path(
        args.output
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            results,
            indent=2,
        )
    )

    np.savez_compressed(
        out.with_suffix(
            ".npz"
        ),

        candidate_speed=
            candidate,

        background64=
            bg64,

        true64=
            true64,

        pred64=
            pred64,

        rec_indices=
            rec_indices,

        seen_mask=
            seen_mask,

        full_rrmse=
            full_err,

        rec_rrmse=
            rec_err,
    )

    print()
    print(
        "saved JSON =",
        out,
    )

    print(
        "saved NPZ  =",
        out.with_suffix(
            ".npz"
        ),
    )


if __name__ == "__main__":
    main()
