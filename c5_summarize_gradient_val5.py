import json
import argparse
from pathlib import Path

import numpy as np


def stat(x):
    x = np.asarray(x, dtype=float)
    return {
        "mean": x.mean(),
        "std": x.std(),
        "min": x.min(),
        "max": x.max(),
        "median": np.median(x),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        required=True,
    )

    args = ap.parse_args()

    root = Path(args.root)

    files = sorted(
        root.glob("test_*.json")
    )

    if not files:
        raise RuntimeError(
            f"No JSON files under {root}"
        )

    models = [
        "FNO",
        "MgNO-I",
        "MgNO-II",
    ]

    rows = {
        m: {
            "fwd_full": [],
            "fwd_rec": [],
            "cos": [],
            "crop16": [],
            "crop32": [],
            "smooth5": [],
            "smooth9": [],
        }
        for m in models
    }

    print("=" * 120)
    print("C5.3c PER-MEDIUM GRADIENT RESULTS")
    print("=" * 120)

    print(
        f"{'sample':10s}"
        f"{'model':12s}"
        f"{'fwd_full':>12s}"
        f"{'fwd_rec':>12s}"
        f"{'cos':>12s}"
        f"{'crop32':>12s}"
        f"{'smooth5':>12s}"
        f"{'smooth9':>12s}"
    )

    for path in files:

        d = json.loads(
            path.read_text()
        )

        for m in models:

            r = d[m]

            values = {
                "fwd_full":
                    r["forward_full_rrmse"],

                "fwd_rec":
                    r["forward_rec_rrmse"],

                "cos":
                    r["gradient"]["full"]["cosine"],

                "crop16":
                    r["gradient"]["crop16"]["cosine"],

                "crop32":
                    r["gradient"]["crop32"]["cosine"],

                "smooth5":
                    r["gradient"]["smooth5"]["cosine"],

                "smooth9":
                    r["gradient"]["smooth9"]["cosine"],
            }

            for k, v in values.items():
                rows[m][k].append(
                    float(v)
                )

            print(
                f"{path.stem:10s}"
                f"{m:12s}"
                f"{values['fwd_full']:12.6f}"
                f"{values['fwd_rec']:12.6f}"
                f"{values['cos']:12.6f}"
                f"{values['crop32']:12.6f}"
                f"{values['smooth5']:12.6f}"
                f"{values['smooth9']:12.6f}"
            )

    print()
    print("=" * 120)
    print("C5.3c AGGREGATE")
    print("=" * 120)

    print(
        f"{'model':12s}"
        f"{'fwd_full':>20s}"
        f"{'fwd_rec':>20s}"
        f"{'raw_cos':>20s}"
        f"{'smooth5':>20s}"
        f"{'smooth9':>20s}"
        f"{'cos>0.2':>10s}"
    )

    summary = {}

    for m in models:

        ff = stat(
            rows[m]["fwd_full"]
        )

        fr = stat(
            rows[m]["fwd_rec"]
        )

        cc = stat(
            rows[m]["cos"]
        )

        s5 = stat(
            rows[m]["smooth5"]
        )

        s9 = stat(
            rows[m]["smooth9"]
        )

        wins02 = int(
            np.sum(
                np.asarray(
                    rows[m]["cos"]
                ) > 0.2
            )
        )

        print(
            f"{m:12s}"
            f"{ff['mean']:9.4f}±{ff['std']:<9.4f}"
            f"{fr['mean']:9.4f}±{fr['std']:<9.4f}"
            f"{cc['mean']:9.4f}±{cc['std']:<9.4f}"
            f"{s5['mean']:9.4f}±{s5['std']:<9.4f}"
            f"{s9['mean']:9.4f}±{s9['std']:<9.4f}"
            f"{wins02:10d}"
        )

        summary[m] = {
            k: stat(v)
            for k, v in rows[m].items()
        }

        summary[m]["raw_cos_gt_0p2"] = (
            wins02
        )

    # Which model has best raw cosine per medium?
    raw = np.stack(
        [
            rows[m]["cos"]
            for m in models
        ],
        axis=1,
    )

    winner = np.argmax(
        raw,
        axis=1,
    )

    print()
    print("RAW GRADIENT COSINE WIN COUNTS")

    for j, m in enumerate(models):
        print(
            f"{m:12s}:",
            int(
                np.sum(
                    winner == j
                )
            ),
            "/",
            len(files),
        )

    out = (
        root
        / "aggregate_summary.json"
    )

    out.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print(
        "saved =",
        out,
    )


if __name__ == "__main__":
    main()
