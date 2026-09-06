import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# basic metrics
# ============================================================

def mse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def mae(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def normalize_direction(g):
    g = np.asarray(g, dtype=np.float64)

    rms = np.sqrt(
        np.mean(g ** 2)
    )

    if not np.isfinite(rms) or rms < 1e-12:
        raise RuntimeError(
            f"invalid gradient RMS = {rms}"
        )

    return g / rms


# ============================================================
# find gradient
# ============================================================

RA_KEYS = [
    "ra_arss_gradient",
    "ra_gradient",
    "g_ra",
    "gradient_ra",
]

EXACT_KEYS = [
    "exact_gradient",
    "g_exact",
    "cbs_gradient",
]


def find_gradient_file(root, sample):
    root = Path(root)

    patterns = [
        f"test_{sample}_gradients.npz",
        f"test_{sample}.npz",
        f"*test_{sample}*.npz",
        f"*{sample}*.npz",
    ]

    hits = []

    for pat in patterns:
        for p in root.rglob(pat):
            if p not in hits:
                hits.append(p)

    for p in hits:
        try:
            z = np.load(p, allow_pickle=True)

            keys = set(z.files)

            has_ra = any(
                k in keys
                for k in RA_KEYS
            )

            if has_ra:
                return p

        except Exception:
            pass

    return None


def get_key(z, candidates):
    for k in candidates:
        if k in z.files:
            return k

    return None


# ============================================================
# accepted step
# ============================================================

def load_steps(gra_dir):
    gra_dir = Path(gra_dir)

    candidates = [
        gra_dir / "summary.csv",
        gra_dir / "gra_summary.csv",
        gra_dir / "all_steps.csv",
    ]

    csv_path = None

    for p in candidates:
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        for p in gra_dir.rglob("*.csv"):
            try:
                txt = p.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

                if (
                    "sample" in txt
                    and
                    (
                        "accepted_step_mps" in txt
                        or
                        "accepted_step" in txt
                    )
                ):
                    csv_path = p
                    break
            except Exception:
                pass

    if csv_path is None:
        raise RuntimeError(
            "Cannot find GRA summary CSV "
            "containing accepted step."
        )

    print(
        "[INFO] step CSV =",
        csv_path,
    )

    steps = {}

    with csv_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if "sample" not in row:
                continue

            sid = int(
                float(row["sample"])
            )

            if "accepted_step_mps" in row:
                step = float(
                    row["accepted_step_mps"]
                )

            elif "accepted_step" in row:
                step = float(
                    row["accepted_step"]
                )

            else:
                continue

            steps[sid] = step

    if not steps:
        raise RuntimeError(
            "No accepted steps found."
        )

    return steps


# ============================================================
# plotting
# ============================================================

def add_colorbar(fig, im, ax, label):
    cb = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cb.set_label(label)


def plot_full(
    sid,
    gt,
    hint,
    c4,
    gra,
    step,
    out,
):

    err_c4 = np.abs(
        c4 - gt
    )

    err_gra = np.abs(
        gra - gt
    )

    correction = (
        gra - c4
    )

    # Physical range is frozen globally.
    vmin = 1400.0
    vmax = 1605.0

    # Same error scale for fair comparison.
    err_lim = np.percentile(
        np.concatenate(
            [
                err_c4.ravel(),
                err_gra.ravel(),
            ]
        ),
        99.5,
    )

    diff_lim = np.percentile(
        np.abs(correction),
        99.5,
    )

    diff_lim = max(
        float(diff_lim),
        1e-6,
    )

    fig, ax = plt.subplots(
        2,
        4,
        figsize=(15, 8),
    )

    im = ax[0, 0].imshow(
        gt,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )

    ax[0, 0].set_title(
        f"GT (test {sid})"
    )

    ax[0, 1].imshow(
        hint,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )

    ax[0, 1].set_title(
        "Supervised Hint"
    )

    ax[0, 2].imshow(
        c4,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )

    ax[0, 2].set_title(
        f"C4\nMSE={mse(c4, gt):.2f}"
    )

    ax[0, 3].imshow(
        gra,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )

    ax[0, 3].set_title(
        f"GRA final, step={step:g}\n"
        f"MSE={mse(gra, gt):.2f}"
    )

    im_err = ax[1, 0].imshow(
        np.abs(hint - gt),
        cmap="magma",
        vmin=0,
        vmax=err_lim,
    )

    ax[1, 0].set_title(
        "|Hint - GT|"
    )

    ax[1, 1].imshow(
        err_c4,
        cmap="magma",
        vmin=0,
        vmax=err_lim,
    )

    ax[1, 1].set_title(
        "|C4 - GT|"
    )

    ax[1, 2].imshow(
        err_gra,
        cmap="magma",
        vmin=0,
        vmax=err_lim,
    )

    ax[1, 2].set_title(
        "|GRA - GT|"
    )

    im_diff = ax[1, 3].imshow(
        correction,
        cmap="coolwarm",
        vmin=-diff_lim,
        vmax=diff_lim,
    )

    ax[1, 3].set_title(
        "GRA - C4"
    )

    for a in ax.ravel():
        a.axis("off")

    add_colorbar(
        fig,
        im,
        ax[0, :].tolist(),
        "Sound speed (m/s)",
    )

    add_colorbar(
        fig,
        im_err,
        ax[1, :3].tolist(),
        "Absolute error (m/s)",
    )

    add_colorbar(
        fig,
        im_diff,
        ax[1, 3],
        "Correction (m/s)",
    )

    fig.suptitle(
        (
            f"test {sid}: C4 -> GRA physical refinement | "
            f"MSE gain = "
            f"{100*(mse(c4,gt)-mse(gra,gt))/mse(c4,gt):+.2f}%"
        ),
        fontsize=14,
    )

    plt.savefig(
        out,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_roi(
    sid,
    gt,
    c4,
    gra,
    step,
    out,
):

    # Same 300x300 ROI convention used
    # elsewhere in the project.
    s = np.s_[90:390, 90:390]

    gt = gt[s]
    c4 = c4[s]
    gra = gra[s]

    err_c4 = np.abs(
        c4 - gt
    )

    err_gra = np.abs(
        gra - gt
    )

    correction = (
        gra - c4
    )

    vmin = 1400.0
    vmax = 1605.0

    err_lim = np.percentile(
        np.concatenate(
            [
                err_c4.ravel(),
                err_gra.ravel(),
            ]
        ),
        99,
    )

    diff_lim = max(
        float(
            np.percentile(
                np.abs(correction),
                99,
            )
        ),
        1e-6,
    )

    fig, ax = plt.subplots(
        2,
        3,
        figsize=(11, 7),
    )

    im = ax[0, 0].imshow(
        gt,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )

    ax[0, 0].set_title(
        "GT ROI"
    )

    ax[0, 1].imshow(
        c4,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )

    ax[0, 1].set_title(
        "C4 ROI"
    )

    ax[0, 2].imshow(
        gra,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )

    ax[0, 2].set_title(
        f"GRA ROI\nstep={step:g}"
    )

    im_err = ax[1, 0].imshow(
        err_c4,
        cmap="magma",
        vmin=0,
        vmax=err_lim,
    )

    ax[1, 0].set_title(
        "|C4-GT| ROI"
    )

    ax[1, 1].imshow(
        err_gra,
        cmap="magma",
        vmin=0,
        vmax=err_lim,
    )

    ax[1, 1].set_title(
        "|GRA-GT| ROI"
    )

    im_diff = ax[1, 2].imshow(
        correction,
        cmap="coolwarm",
        vmin=-diff_lim,
        vmax=diff_lim,
    )

    ax[1, 2].set_title(
        "GRA-C4 ROI"
    )

    for a in ax.ravel():
        a.axis("off")

    add_colorbar(
        fig,
        im,
        ax[0, :].tolist(),
        "Sound speed (m/s)",
    )

    add_colorbar(
        fig,
        im_err,
        ax[1, :2].tolist(),
        "Absolute error (m/s)",
    )

    add_colorbar(
        fig,
        im_diff,
        ax[1, 2],
        "Correction (m/s)",
    )

    fig.suptitle(
        f"test {sid}: central ROI comparison",
        fontsize=14,
    )

    plt.savefig(
        out,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_profile(
    sid,
    gt,
    c4,
    gra,
    out,
):

    # Horizontal line through ROI centre.
    row = 240

    x = np.arange(
        gt.shape[1]
    )

    fig, ax = plt.subplots(
        figsize=(9, 4),
    )

    ax.plot(
        x,
        gt[row],
        label="GT",
        linewidth=2,
    )

    ax.plot(
        x,
        c4[row],
        label="C4",
        linewidth=1.5,
    )

    ax.plot(
        x,
        gra[row],
        label="GRA",
        linewidth=1.5,
    )

    ax.set_xlim(
        90,
        390,
    )

    ax.set_xlabel(
        "Pixel"
    )

    ax.set_ylabel(
        "Sound speed (m/s)"
    )

    ax.set_title(
        f"test {sid}: central horizontal profile"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        out,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# main
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--bridge_dir",
        required=True,
    )

    ap.add_argument(
        "--gradient_dir",
        required=True,
    )

    ap.add_argument(
        "--gra_dir",
        required=True,
    )

    ap.add_argument(
        "--output_dir",
        required=True,
    )

    ap.add_argument(
        "--samples",
        default="11,12,15,16,17,18,20",
    )

    args = ap.parse_args()

    bridge = Path(
        args.bridge_dir
    )

    out = Path(
        args.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    steps = load_steps(
        args.gra_dir
    )

    sample_ids = [
        int(x)
        for x in args.samples.split(",")
    ]

    print("=" * 100)
    print("C6.4B FINAL RECONSTRUCTION VISUALIZATION")
    print("=" * 100)

    for sid in sample_ids:

        bp = (
            bridge
            /
            f"test_{sid}.npz"
        )

        if not bp.exists():
            raise FileNotFoundError(
                bp
            )

        b = np.load(
            bp,
            allow_pickle=True,
        )

        gt = b[
            "target_480"
        ].astype(np.float32)

        hint = b[
            "hint_480"
        ].astype(np.float32)

        c4 = b[
            "candidate_480"
        ].astype(np.float32)

        gp = find_gradient_file(
            args.gradient_dir,
            sid,
        )

        if gp is None:
            raise RuntimeError(
                f"Cannot find RA gradient for test {sid} "
                f"under {args.gradient_dir}"
            )

        gz = np.load(
            gp,
            allow_pickle=True,
        )

        ra_key = get_key(
            gz,
            RA_KEYS,
        )

        if ra_key is None:
            raise RuntimeError(
                f"{gp}: no RA gradient key.\n"
                f"available keys = {gz.files}"
            )

        if sid not in steps:
            raise RuntimeError(
                f"No accepted step for test {sid}"
            )

        step = steps[sid]

        grad = gz[
            ra_key
        ].astype(np.float64)

        direction = normalize_direction(
            grad
        )

        if step > 0:
            gra = (
                c4
                - step * direction
            ).astype(np.float32)
        else:
            gra = c4.copy()

        print()
        print(
            f"test {sid:02d}"
        )

        print(
            "  gradient file =",
            gp,
        )

        print(
            "  gradient key  =",
            ra_key,
        )

        print(
            "  accepted step =",
            step,
        )

        print(
            "  C4 MSE        =",
            f"{mse(c4, gt):.6f}",
        )

        print(
            "  GRA MSE       =",
            f"{mse(gra, gt):.6f}",
        )

        print(
            "  MSE gain      =",
            f"{100*(mse(c4,gt)-mse(gra,gt))/mse(c4,gt):+.3f}%",
        )

        print(
            "  mean |change| =",
            f"{np.mean(np.abs(gra-c4)):.6f} m/s",
        )

        print(
            "  max  |change| =",
            f"{np.max(np.abs(gra-c4)):.6f} m/s",
        )

        plot_full(
            sid,
            gt,
            hint,
            c4,
            gra,
            step,
            out /
            f"test_{sid:02d}_full.png",
        )

        plot_roi(
            sid,
            gt,
            c4,
            gra,
            step,
            out /
            f"test_{sid:02d}_roi.png",
        )

        plot_profile(
            sid,
            gt,
            c4,
            gra,
            out /
            f"test_{sid:02d}_profile.png",
        )


if __name__ == "__main__":
    main()