import argparse
import json
import re
from pathlib import Path

import numpy as np


def numeric_key(p):
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else -1


def radial_energy_quantiles(w):
    """
    w: complex [H,W]

    Return spectral radius K containing
    50/90/95/99% of Fourier energy.
    """

    h, wid = w.shape

    ft = np.fft.fftshift(
        np.fft.fft2(w)
    )

    energy = (
        np.abs(ft) ** 2
    ).astype(np.float64)

    yy = (
        np.arange(h)
        -
        h // 2
    )

    xx = (
        np.arange(wid)
        -
        wid // 2
    )

    Y, X = np.meshgrid(
        yy,
        xx,
        indexing="ij",
    )

    R = np.sqrt(
        X ** 2
        +
        Y ** 2
    )

    r = R.reshape(-1)
    e = energy.reshape(-1)

    order = np.argsort(r)

    r = r[order]
    e = e[order]

    cum = np.cumsum(e)

    total = cum[-1]

    if total <= 0:
        return {
            "k50": 0,
            "k90": 0,
            "k95": 0,
            "k99": 0,
        }

    cum /= total

    result = {}

    for q in [
        0.50,
        0.90,
        0.95,
        0.99,
    ]:

        idx = np.searchsorted(
            cum,
            q,
        )

        result[
            f"k{int(q * 100)}"
        ] = float(
            r[
                min(
                    idx,
                    len(r) - 1,
                )
            ]
        )

    return result


def low_frequency_energy(w, k_values):
    h, wid = w.shape

    ft = np.fft.fftshift(
        np.fft.fft2(w)
    )

    energy = (
        np.abs(ft) ** 2
    ).astype(np.float64)

    yy = (
        np.arange(h)
        -
        h // 2
    )

    xx = (
        np.arange(wid)
        -
        wid // 2
    )

    Y, X = np.meshgrid(
        yy,
        xx,
        indexing="ij",
    )

    R = np.sqrt(
        X ** 2
        +
        Y ** 2
    )

    total = float(
        energy.sum()
    )

    result = {}

    for k in k_values:

        retained = float(
            energy[
                R <= k
            ].sum()
        )

        result[k] = (
            retained / total
            if total > 0
            else 0.0
        )

    return result


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        required=True,
    )

    ap.add_argument(
        "--max_images",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--max_sources",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260904,
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    args = ap.parse_args()

    root = Path(args.root)

    files = sorted(
        (root / "train").glob(
            "*.npz"
        ),
        key=numeric_key,
    )

    files = files[
        :args.max_images
    ]

    rng = np.random.default_rng(
        args.seed
    )

    k_values = [
        8,
        16,
        24,
        32,
        48,
        64,
        80,
        96,
        112,
        128,
        160,
    ]

    spectra = []
    sampled_abs = []

    rms_values = []

    print("=" * 100)
    print(
        "C5 WAVEFIELD SPECTRAL AUDIT"
    )
    print("=" * 100)

    print(
        "num images =",
        len(files),
    )

    field_count = 0

    for p in files:

        with np.load(
            p,
            allow_pickle=True,
        ) as z:

            wavefields = (
                z["wavefields"]
                .astype(np.complex64)
            )

        ns = min(
            args.max_sources,
            wavefields.shape[0],
        )

        for s in range(ns):

            w = wavefields[s]

            absw = np.abs(w)

            rms = float(
                np.sqrt(
                    np.mean(
                        absw ** 2
                    )
                )
            )

            rms_values.append(rms)

            # Random sample amplitude values
            # to avoid keeping all ~37M points.
            flat = absw.reshape(-1)

            n_take = min(
                50000,
                flat.size,
            )

            idx = rng.choice(
                flat.size,
                size=n_take,
                replace=False,
            )

            sampled_abs.append(
                flat[idx]
            )

            q = (
                radial_energy_quantiles(
                    w
                )
            )

            low = (
                low_frequency_energy(
                    w,
                    k_values,
                )
            )

            spectra.append({
                **q,
                **{
                    f"E{k}": low[k]
                    for k in k_values
                },
            })

            field_count += 1

    amp = np.concatenate(
        sampled_abs
    )

    print()
    print(
        "num fields audited =",
        field_count,
    )

    print()
    print("=" * 100)
    print("AMPLITUDE |u|")
    print("=" * 100)

    for q in [
        0,
        50,
        90,
        95,
        99,
        99.5,
        99.9,
        100,
    ]:

        print(
            f"p{q:<5} = "
            f"{np.percentile(amp, q):.8e}"
        )

    rms_values = np.asarray(
        rms_values,
        dtype=np.float64,
    )

    print()
    print(
        "field RMS mean/std = "
        f"{rms_values.mean():.8e} ± "
        f"{rms_values.std():.8e}"
    )

    print(
        "field RMS min/max  = "
        f"{rms_values.min():.8e} / "
        f"{rms_values.max():.8e}"
    )

    print()
    print("=" * 100)
    print("SPECTRAL RADIUS QUANTILES")
    print("=" * 100)

    for key in [
        "k50",
        "k90",
        "k95",
        "k99",
    ]:

        a = np.asarray(
            [
                r[key]
                for r in spectra
            ]
        )

        print(
            f"{key:5s} = "
            f"{a.mean():8.2f} ± "
            f"{a.std():6.2f} "
            f"(min={a.min():.1f}, "
            f"max={a.max():.1f})"
        )

    print()
    print("=" * 100)
    print("ENERGY RETAINED INSIDE RADIAL MODE K")
    print("=" * 100)

    for k in k_values:

        a = np.asarray(
            [
                r[f"E{k}"]
                for r in spectra
            ]
        )

        print(
            f"K={k:3d}: "
            f"{100*a.mean():7.3f}% ± "
            f"{100*a.std():6.3f}%"
        )

    summary = {
        "num_images":
            len(files),

        "num_fields":
            field_count,

        "field_rms_mean":
            float(
                rms_values.mean()
            ),

        "field_rms_std":
            float(
                rms_values.std()
            ),

        "amplitude_percentiles": {
            str(q):
            float(
                np.percentile(
                    amp,
                    q,
                )
            )
            for q in [
                0,
                50,
                90,
                95,
                99,
                99.5,
                99.9,
                100,
            ]
        },

        "spectral_quantiles": {},

        "energy_retention": {},
    }

    for key in [
        "k50",
        "k90",
        "k95",
        "k99",
    ]:

        a = np.asarray(
            [
                r[key]
                for r in spectra
            ]
        )

        summary[
            "spectral_quantiles"
        ][key] = {
            "mean":
                float(a.mean()),
            "std":
                float(a.std()),
            "min":
                float(a.min()),
            "max":
                float(a.max()),
        }

    for k in k_values:

        a = np.asarray(
            [
                r[f"E{k}"]
                for r in spectra
            ]
        )

        summary[
            "energy_retention"
        ][str(k)] = {
            "mean":
                float(a.mean()),
            "std":
                float(a.std()),
        }

    out = Path(args.out)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        out,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(
        "[PASS] spectral audit complete"
    )

    print(
        "summary =",
        out,
    )


if __name__ == "__main__":
    main()
