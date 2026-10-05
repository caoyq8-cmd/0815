from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# paths
# =============================================================================

METRIC_CSV = Path(
    "/home/featurize/work/USCT_repro/USCT_download/"
    "final_thesis_metrics/"
    "canonical_test400_inversionnet_naive_pgcr.csv"
)

NAIVE_ROOT = Path(
    "/home/featurize/work/USCT_repro/USCT_download/"
    "formal_diffano_truecbs_test400"
)

PGCR_ROOT = Path(
    "/home/featurize/work/USCT_repro/USCT_download/"
    "pgcr_frozen_test400"
)

OUT = Path(
    "/home/featurize/work/USCT_repro/USCT_download/"
    "final_thesis_figures/pgcr"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# load metrics
# =============================================================================

rows = list(
    csv.DictReader(
        METRIC_CSV.open(
            encoding="utf-8"
        )
    )
)

for r in rows:

    r["sid"] = int(r["sid"])

    for k in list(r.keys()):
        if k != "sid":
            r[k] = float(r[k])

    r["pgcr_mse_gain"] = (
        r["inv_mse"]
        - r["pgcr_mse"]
    )

    r["naive_mae_damage"] = (
        r["naive_mae"]
        - r["inv_mae"]
    )

    r["pgcr_mae_recovery"] = (
        r["naive_mae"]
        - r["pgcr_mae"]
    )

    r["pgcr_ssim_gain"] = (
        r["pgcr_ssim"]
        - r["inv_ssim"]
    )


# =============================================================================
# deterministic sample selection
# =============================================================================
#
# 2 samples:
#   strongest PGCR MSE gain
#
# 2 samples:
#   strongest Naive-CBS MAE damage
#
# 2 samples:
#   strongest PGCR recovery from Naive-CBS damage
#
# duplicate IDs are skipped.
# =============================================================================

groups = [
    (
        "Best PGCR MSE",
        sorted(
            rows,
            key=lambda r: r["pgcr_mse_gain"],
            reverse=True,
        ),
    ),
    (
        "Naive-CBS damage",
        sorted(
            rows,
            key=lambda r: r["naive_mae_damage"],
            reverse=True,
        ),
    ),
    (
        "PGCR recovery",
        sorted(
            rows,
            key=lambda r: r["pgcr_mae_recovery"],
            reverse=True,
        ),
    ),
]

selected = []
used = set()

for group_name, ordered in groups:

    count = 0

    for r in ordered:

        if r["sid"] in used:
            continue

        item = dict(r)
        item["selection_group"] = group_name

        selected.append(item)
        used.add(r["sid"])

        count += 1

        if count == 2:
            break


print("=" * 120)
print("SELECTED THESIS SAMPLES")
print("=" * 120)

for r in selected:

    print(
        f"{r['selection_group']:20s} "
        f"test_{r['sid']:03d} | "
        f"MSE: "
        f"{r['inv_mse']:.2f} -> "
        f"{r['naive_mse']:.2f} -> "
        f"{r['pgcr_mse']:.2f} | "
        f"MAE: "
        f"{r['inv_mae']:.3f} -> "
        f"{r['naive_mae']:.3f} -> "
        f"{r['pgcr_mae']:.3f} | "
        f"SSIM: "
        f"{r['inv_ssim']:.4f} -> "
        f"{r['naive_ssim']:.4f} -> "
        f"{r['pgcr_ssim']:.4f}"
    )

print("=" * 120)


# =============================================================================
# save selection manifest
# =============================================================================

manifest = []

for r in selected:

    manifest.append({
        "selection_group":
            r["selection_group"],

        "sid":
            r["sid"],

        "inv_mse":
            r["inv_mse"],

        "naive_mse":
            r["naive_mse"],

        "pgcr_mse":
            r["pgcr_mse"],

        "inv_mae":
            r["inv_mae"],

        "naive_mae":
            r["naive_mae"],

        "pgcr_mae":
            r["pgcr_mae"],

        "inv_ssim":
            r["inv_ssim"],

        "naive_ssim":
            r["naive_ssim"],

        "pgcr_ssim":
            r["pgcr_ssim"],
    })

(OUT / "selected_samples.json").write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    )
)


# =============================================================================
# image loader
# =============================================================================

def load_sample(sid):

    pn = np.load(
        NAIVE_ROOT
        / f"test_{sid}"
        / "final_result.npz"
    )

    pp = np.load(
        PGCR_ROOT
        / f"test_{sid}"
        / "final_result.npz"
    )

    gt = pp[
        "target_256_phys"
    ].astype(np.float32)

    inv = pp[
        "condition_256_phys"
    ].astype(np.float32)

    naive = pn[
        "corrected_256_phys"
    ].astype(np.float32)

    pgcr = pp[
        "corrected_256_phys"
    ].astype(np.float32)

    return gt, inv, naive, pgcr


# =============================================================================
# Figure A
# reconstruction comparison
# =============================================================================

n = len(selected)

fig, axes = plt.subplots(
    n,
    4,
    figsize=(10.5, 2.45 * n),
)

titles = [
    "Ground Truth",
    "InversionNet",
    "Naive-CBS8",
    "PGCR",
]

for i, r in enumerate(selected):

    sid = r["sid"]

    gt, inv, naive, pgcr = load_sample(
        sid
    )

    imgs = [
        gt,
        inv,
        naive,
        pgcr,
    ]

    for j, img in enumerate(imgs):

        ax = axes[i, j]

        im = ax.imshow(
            img,
            cmap="inferno",
            vmin=1400,
            vmax=1605,
        )

        ax.set_xticks([])
        ax.set_yticks([])

        if i == 0:
            ax.set_title(
                titles[j],
                fontsize=11,
            )

        if j == 0:
            ax.set_ylabel(
                f"Test {sid}\n"
                f"{r['selection_group']}",
                fontsize=9,
            )

        if j > 0:

            prefix = [
                None,
                "inv",
                "naive",
                "pgcr",
            ][j]

            ax.text(
                0.02,
                0.03,
                (
                    f"MSE={r[prefix+'_mse']:.1f}\n"
                    f"SSIM={r[prefix+'_ssim']:.3f}"
                ),
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="bottom",
                bbox=dict(
                    facecolor="white",
                    alpha=0.75,
                    edgecolor="none",
                    pad=2,
                ),
            )


cbar = fig.colorbar(
    im,
    ax=axes.ravel().tolist(),
    fraction=0.015,
    pad=0.012,
)

cbar.set_label(
    "Speed of sound (m/s)"
)

fig.suptitle(
    "Reconstruction comparison on representative TEST samples",
    fontsize=13,
    y=0.995,
)

fig.subplots_adjust(
    left=0.10,
    right=0.93,
    top=0.965,
    bottom=0.02,
    wspace=0.035,
    hspace=0.08,
)

fig.savefig(
    OUT / "pgcr_reconstruction_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT / "pgcr_reconstruction_comparison.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# Figure B
# absolute error comparison
# =============================================================================

# common error scale over selected images:
# use 99th percentile to prevent one outlier
# from ruining visual contrast.

all_errors = []

cache = {}

for r in selected:

    sid = r["sid"]

    gt, inv, naive, pgcr = load_sample(
        sid
    )

    e_inv = np.abs(inv - gt)
    e_naive = np.abs(naive - gt)
    e_pgcr = np.abs(pgcr - gt)

    cache[sid] = (
        e_inv,
        e_naive,
        e_pgcr,
    )

    all_errors.extend([
        e_inv.ravel(),
        e_naive.ravel(),
        e_pgcr.ravel(),
    ])

all_errors = np.concatenate(
    all_errors
)

ERR_MAX = float(
    np.percentile(
        all_errors,
        99.0,
    )
)

fig, axes = plt.subplots(
    n,
    3,
    figsize=(8.2, 2.45 * n),
)

titles = [
    "|InversionNet - GT|",
    "|Naive-CBS8 - GT|",
    "|PGCR - GT|",
]

for i, r in enumerate(selected):

    sid = r["sid"]

    imgs = cache[sid]

    for j, img in enumerate(imgs):

        ax = axes[i, j]

        im = ax.imshow(
            img,
            cmap="magma",
            vmin=0,
            vmax=ERR_MAX,
        )

        ax.set_xticks([])
        ax.set_yticks([])

        if i == 0:
            ax.set_title(
                titles[j],
                fontsize=11,
            )

        if j == 0:
            ax.set_ylabel(
                f"Test {sid}",
                fontsize=9,
            )


cbar = fig.colorbar(
    im,
    ax=axes.ravel().tolist(),
    fraction=0.02,
    pad=0.015,
)

cbar.set_label(
    "Absolute error (m/s)"
)

fig.suptitle(
    "Absolute-error comparison",
    fontsize=13,
    y=0.995,
)

fig.subplots_adjust(
    left=0.08,
    right=0.92,
    top=0.965,
    bottom=0.02,
    wspace=0.035,
    hspace=0.08,
)

fig.savefig(
    OUT / "pgcr_absolute_error_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT / "pgcr_absolute_error_comparison.pdf",
    bbox_inches="tight",
)

plt.close(fig)


print()
print("[PASS] figures saved to")
print(OUT)

print()
print("files:")
for p in sorted(OUT.iterdir()):
    print(" ", p.name)
