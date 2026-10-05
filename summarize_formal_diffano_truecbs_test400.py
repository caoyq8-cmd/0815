import json
import csv
from pathlib import Path

import numpy as np


ROOT = Path(
    "/home/featurize/work/USCT_repro/"
    "USCT_download/formal_diffano_truecbs_test400"
)


rows = []
missing = []


for sid in range(1, 401):

    p = (
        ROOT /
        f"test_{sid}" /
        "run_summary.json"
    )

    if not p.exists():
        missing.append(sid)
        continue

    s = json.loads(
        p.read_text()
    )

    b = s["initial"]
    a = s["final"]

    row = {
        "test_id": sid,

        "mse_before":
            b["image_mse_256"],

        "mse_after":
            a["image_mse_256"],

        "mae_before":
            b["image_mae_256"],

        "mae_after":
            a["image_mae_256"],

        "rmse_before":
            b["image_rmse_256"],

        "rmse_after":
            a["image_rmse_256"],

        "psnr_before":
            b["image_psnr_256"],

        "psnr_after":
            a["image_psnr_256"],

        "ssim_before":
            b["image_ssim_256"],

        "ssim_after":
            a["image_ssim_256"],

        "dobs_abs_before":
            b["dobs_abs_loss"],

        "dobs_abs_after":
            a["dobs_abs_loss"],

        "dobs_rel_l2_before":
            b["dobs_rel_l2"],

        "dobs_rel_l2_after":
            a["dobs_rel_l2"],

        "physics_descent":
            bool(
                a["dobs_abs_loss"]
                <
                b["dobs_abs_loss"]
            ),

        "mse_win":
            bool(
                a["image_mse_256"]
                <
                b["image_mse_256"]
            ),

        "mae_win":
            bool(
                a["image_mae_256"]
                <
                b["image_mae_256"]
            ),

        "ssim_win":
            bool(
                a["image_ssim_256"]
                >
                b["image_ssim_256"]
            ),

        "runtime_seconds":
            s["runtime_seconds"],
    }

    rows.append(row)


if missing:
    print(
        "missing:",
        missing
    )

if len(rows) != 400:
    raise RuntimeError(
        f"Only {len(rows)}/400 completed"
    )


def arr(key):
    return np.asarray(
        [
            r[key]
            for r in rows
        ],
        dtype=np.float64
    )


def stat(key):
    x = arr(key)

    return {
        "mean":
            float(x.mean()),

        "std":
            float(x.std()),

        "median":
            float(np.median(x)),
    }


summary = {
    "n": 400,

    "before": {
        "mse":
            stat("mse_before"),

        "mae":
            stat("mae_before"),

        "rmse":
            stat("rmse_before"),

        "psnr":
            stat("psnr_before"),

        "ssim":
            stat("ssim_before"),

        "dobs_abs":
            stat("dobs_abs_before"),

        "dobs_rel_l2":
            stat("dobs_rel_l2_before"),
    },

    "after": {
        "mse":
            stat("mse_after"),

        "mae":
            stat("mae_after"),

        "rmse":
            stat("rmse_after"),

        "psnr":
            stat("psnr_after"),

        "ssim":
            stat("ssim_after"),

        "dobs_abs":
            stat("dobs_abs_after"),

        "dobs_rel_l2":
            stat("dobs_rel_l2_after"),
    },

    "win_counts": {
        "physics_descent":
            int(
                sum(
                    r["physics_descent"]
                    for r in rows
                )
            ),

        "mse":
            int(
                sum(
                    r["mse_win"]
                    for r in rows
                )
            ),

        "mae":
            int(
                sum(
                    r["mae_win"]
                    for r in rows
                )
            ),

        "ssim":
            int(
                sum(
                    r["ssim_win"]
                    for r in rows
                )
            ),
    },

    "relative_mean_change": {
        "mse": float(
            (
                arr("mse_before").mean()
                -
                arr("mse_after").mean()
            )
            /
            arr("mse_before").mean()
        ),

        "dobs_abs": float(
            (
                arr("dobs_abs_before").mean()
                -
                arr("dobs_abs_after").mean()
            )
            /
            arr("dobs_abs_before").mean()
        ),

        "dobs_rel_l2": float(
            (
                arr("dobs_rel_l2_before").mean()
                -
                arr("dobs_rel_l2_after").mean()
            )
            /
            arr("dobs_rel_l2_before").mean()
        ),
    },

    "runtime": {
        "mean_seconds":
            float(
                arr(
                    "runtime_seconds"
                ).mean()
            ),

        "total_seconds":
            float(
                arr(
                    "runtime_seconds"
                ).sum()
            ),
    }
}


csv_path = (
    ROOT /
    "formal_diffano_truecbs_test400.csv"
)

with csv_path.open(
    "w",
    newline=""
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        )
    )

    w.writeheader()
    w.writerows(rows)


json_path = (
    ROOT /
    "summary.json"
)

json_path.write_text(
    json.dumps(
        summary,
        indent=2
    )
)


print("=" * 100)
print("FORMAL DIFF-ANO TRUE-CBS TEST400")
print("=" * 100)

print()
print("MSE:")
print(
    summary["before"]["mse"]["mean"],
    "->",
    summary["after"]["mse"]["mean"]
)

print("MAE:")
print(
    summary["before"]["mae"]["mean"],
    "->",
    summary["after"]["mae"]["mean"]
)

print("PSNR:")
print(
    summary["before"]["psnr"]["mean"],
    "->",
    summary["after"]["psnr"]["mean"]
)

print("SSIM:")
print(
    summary["before"]["ssim"]["mean"],
    "->",
    summary["after"]["ssim"]["mean"]
)

print()
print(
    "measurement loss:",
    summary["before"]["dobs_abs"]["mean"],
    "->",
    summary["after"]["dobs_abs"]["mean"]
)

print(
    "measurement rel-L2:",
    summary["before"]["dobs_rel_l2"]["mean"],
    "->",
    summary["after"]["dobs_rel_l2"]["mean"]
)

print()
print(
    "physics descent:",
    summary["win_counts"]["physics_descent"],
    "/400"
)

print(
    "MSE wins:",
    summary["win_counts"]["mse"],
    "/400"
)

print(
    "MAE wins:",
    summary["win_counts"]["mae"],
    "/400"
)

print(
    "SSIM wins:",
    summary["win_counts"]["ssim"],
    "/400"
)

print()
print(
    "mean MSE improvement =",
    100
    * summary[
        "relative_mean_change"
    ]["mse"],
    "%"
)

print(
    "mean physics-loss improvement =",
    100
    * summary[
        "relative_mean_change"
    ]["dobs_abs"],
    "%"
)

print()
print("saved:", csv_path)
print("saved:", json_path)
