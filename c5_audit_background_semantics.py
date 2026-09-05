import argparse
import json
import re
from pathlib import Path

import numpy as np


def numeric_key(path):
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


def rrmse(a, b, eps=1e-12):
    a = np.asarray(a)
    b = np.asarray(b)

    return float(
        np.sqrt(np.mean(np.abs(a - b) ** 2))
        /
        (
            np.sqrt(np.mean(np.abs(b) ** 2))
            + eps
        )
    )


def rms(a):
    a = np.asarray(a)
    return float(
        np.sqrt(np.mean(np.abs(a) ** 2))
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data_root",
        required=True,
    )

    ap.add_argument(
        "--background_path",
        required=True,
    )

    ap.add_argument(
        "--output",
        default="c5_background_semantics.json",
    )

    args = ap.parse_args()

    train_dir = Path(args.data_root) / "train"

    files = sorted(
        train_dir.glob("train_*.npz"),
        key=numeric_key,
    )

    if not files:
        files = sorted(
            train_dir.glob("*.npz"),
            key=numeric_key,
        )

    if not files:
        raise RuntimeError(
            f"No train npz files in {train_dir}"
        )

    print("=" * 100)
    print("C5.1j-A BACKGROUND SEMANTICS AUDIT")
    print("=" * 100)

    print("train files =", len(files))

    waves = []

    src_ref = None
    rec_ref = None

    for p in files:
        with np.load(
            p,
            allow_pickle=True,
        ) as z:

            wave = z[
                "wavefields"
            ].astype(
                np.complex64
            )

            src = z[
                "src_indices"
            ].astype(
                np.int64
            )

            rec = z[
                "rec_indices"
            ].astype(
                np.int64
            )

        if src_ref is None:
            src_ref = src
            rec_ref = rec

        else:
            assert np.array_equal(
                src,
                src_ref,
            )

            assert np.array_equal(
                rec,
                rec_ref,
            )

        waves.append(
            wave
        )

    waves = np.stack(
        waves,
        axis=0,
    )

    # [Nimage, Nsource, H, W]
    empirical_mean = waves.mean(
        axis=0
    )

    print(
        "all wavefields shape =",
        waves.shape,
    )

    print(
        "empirical mean shape =",
        empirical_mean.shape,
    )

    b = np.load(
        args.background_path,
        allow_pickle=True,
    )

    print()
    print(
        "baseline keys =",
        list(b.keys()),
    )

    if "src_indices" in b:
        print(
            "source geometry match =",
            np.array_equal(
                b["src_indices"],
                src_ref,
            ),
        )

    if "rec_indices" in b:
        print(
            "receiver geometry match =",
            np.array_equal(
                b["rec_indices"],
                rec_ref,
            ),
        )

    rr = rec_ref[:, 0]
    cc = rec_ref[:, 1]

    report = {}

    for key in [
        "mean_field",
        "background_field",
    ]:
        if key not in b:
            print()
            print(
                f"[MISSING] {key}"
            )
            continue

        field = b[
            key
        ]

        if field.shape != empirical_mean.shape:
            raise RuntimeError(
                f"{key} shape={field.shape}, "
                f"expected={empirical_mean.shape}"
            )

        field = field.astype(
            np.complex64
        )

        print()
        print("-" * 100)
        print(key)
        print("-" * 100)

        print(
            "shape =",
            field.shape,
        )

        print(
            "RMS =",
            f"{rms(field):.8e}",
        )

        em_full = rrmse(
            field,
            empirical_mean,
        )

        em_rec = rrmse(
            field[:, rr, cc],
            empirical_mean[:, rr, cc],
        )

        print(
            "vs empirical train mean:"
        )

        print(
            "  full RRMSE =",
            f"{em_full:.8e}",
        )

        print(
            "  rec  RRMSE =",
            f"{em_rec:.8e}",
        )

        # How good is this field as a raw predictor of every training wavefield?
        full_errors = []
        rec_errors = []

        for i in range(
            waves.shape[0]
        ):
            for s in range(
                waves.shape[1]
            ):
                full_errors.append(
                    rrmse(
                        field[s],
                        waves[i, s],
                    )
                )

                rec_errors.append(
                    rrmse(
                        field[s, rr, cc],
                        waves[i, s, rr, cc],
                    )
                )

        full_errors = np.asarray(
            full_errors
        )

        rec_errors = np.asarray(
            rec_errors
        )

        print(
            "as raw predictor of train wavefields:"
        )

        print(
            "  full mean/std =",
            f"{full_errors.mean():.6f}",
            f"{full_errors.std():.6f}",
        )

        print(
            "  rec  mean/std =",
            f"{rec_errors.mean():.6f}",
            f"{rec_errors.std():.6f}",
        )

        report[key] = {
            "rms": rms(field),

            "vs_empirical_mean_full_rrmse":
                em_full,

            "vs_empirical_mean_rec_rrmse":
                em_rec,

            "raw_predictor_full_mean":
                float(full_errors.mean()),

            "raw_predictor_full_std":
                float(full_errors.std()),

            "raw_predictor_rec_mean":
                float(rec_errors.mean()),

            "raw_predictor_rec_std":
                float(rec_errors.std()),
        }

    if (
        "mean_field" in b
        and "background_field" in b
    ):
        mf = b[
            "mean_field"
        ].astype(
            np.complex64
        )

        bg = b[
            "background_field"
        ].astype(
            np.complex64
        )

        diff_full = rrmse(
            mf,
            bg,
        )

        diff_rec = rrmse(
            mf[:, rr, cc],
            bg[:, rr, cc],
        )

        print()
        print("=" * 100)
        print("MEAN_FIELD vs BACKGROUND_FIELD")
        print("=" * 100)

        print(
            "full RRMSE =",
            f"{diff_full:.8e}",
        )

        print(
            "rec  RRMSE =",
            f"{diff_rec:.8e}",
        )

        report[
            "mean_vs_background"
        ] = {
            "full_rrmse":
                diff_full,

            "rec_rrmse":
                diff_rec,
        }

    out = Path(
        args.output
    )

    out.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print(
        "saved =",
        out.resolve(),
    )


if __name__ == "__main__":
    main()
