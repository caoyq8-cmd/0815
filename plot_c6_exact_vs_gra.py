import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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
    r = float(np.sqrt(np.mean(g ** 2)))
    if not np.isfinite(r) or r < 1e-12:
        raise RuntimeError(f"invalid gradient RMS={r}")
    return g / r


def load_gra_steps(gra_dir):
    gra_dir = Path(gra_dir)
    candidates = [
        gra_dir / "summary.csv",
        gra_dir / "gra_summary.csv",
    ]

    path = None
    for p in candidates:
        if p.exists():
            path = p
            break

    if path is None:
        raise FileNotFoundError(
            f"Cannot find summary.csv under {gra_dir}"
        )

    steps = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "sample" not in row:
                continue
            sid = int(float(row["sample"]))
            if "accepted_step_mps" in row:
                step = float(row["accepted_step_mps"])
            elif "accepted_step" in row:
                step = float(row["accepted_step"])
            else:
                continue
            steps[sid] = step

    if not steps:
        raise RuntimeError(
            f"No accepted steps found in {path}"
        )

    print("[INFO] GRA step CSV =", path)
    return steps


def reconstruct(candidate, gradient, step):
    d = normalize_direction(gradient)
    return (
        np.asarray(candidate, dtype=np.float64)
        - float(step) * d
    ).astype(np.float32)


def hf_energy(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()

    f = np.fft.fftshift(np.fft.fft2(x))
    power = np.abs(f) ** 2

    h, w = x.shape
    yy, xx = np.ogrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0

    rr = np.sqrt(
        ((yy - cy) / max(h, 1)) ** 2
        +
        ((xx - cx) / max(w, 1)) ** 2
    )

    mask = rr >= 0.25

    return float(
        power[mask].sum()
        /
        (power.sum() + 1e-30)
    )


def gradient_energy(x):
    x = np.asarray(x, dtype=np.float64)
    gy = np.diff(x, axis=0)
    gx = np.diff(x, axis=1)

    return float(
        0.5 * (
            np.mean(gx ** 2)
            +
            np.mean(gy ** 2)
        )
    )


def add_cbar(fig, im, axes, label):
    cb = fig.colorbar(
        im,
        ax=axes,
        fraction=0.025,
        pad=0.02,
    )
    cb.set_label(label)


def plot_full(
    sid,
    gt,
    hint,
    c4,
    exact,
    gra,
    exact_step,
    gra_step,
    out_path,
):
    methods = [
        ("GT", gt),
        ("Hint", hint),
        ("C4", c4),
        ("Exact-CBS", exact),
        ("GRA", gra),
    ]

    vmin = 1400.0
    vmax = 1605.0

    errors = [
        np.abs(hint - gt),
        np.abs(c4 - gt),
        np.abs(exact - gt),
        np.abs(gra - gt),
    ]

    err_lim = float(
        np.percentile(
            np.concatenate(
                [e.ravel() for e in errors]
            ),
            99.5,
        )
    )

    exact_delta = exact - c4
    gra_delta = gra - c4

    corr_lim = max(
        float(
            np.percentile(
                np.concatenate(
                    [
                        np.abs(exact_delta).ravel(),
                        np.abs(gra_delta).ravel(),
                    ]
                ),
                99.5,
            )
        ),
        1e-6,
    )

    fig, ax = plt.subplots(
        3,
        5,
        figsize=(18, 11),
    )

    image_axes = []

    for j, (name, img) in enumerate(methods):
        im = ax[0, j].imshow(
            img,
            vmin=vmin,
            vmax=vmax,
        )
        image_axes.append(ax[0, j])

        if name == "C4":
            title = f"C4\nMSE={mse(img, gt):.2f}"
        elif name == "Exact-CBS":
            title = (
                f"Exact-CBS, step={exact_step:g}\n"
                f"MSE={mse(img, gt):.2f}"
            )
        elif name == "GRA":
            title = (
                f"GRA, step={gra_step:g}\n"
                f"MSE={mse(img, gt):.2f}"
            )
        else:
            title = name

        ax[0, j].set_title(title)
        ax[0, j].axis("off")

    err_names = [
        "|Hint-GT|",
        "|C4-GT|",
        "|Exact-GT|",
        "|GRA-GT|",
    ]

    err_axes = []

    for j in range(4):
        im_err = ax[1, j].imshow(
            errors[j],
            vmin=0,
            vmax=err_lim,
        )
        err_axes.append(ax[1, j])
        ax[1, j].set_title(err_names[j])
        ax[1, j].axis("off")

    exact_gra_diff = exact - gra
    eg_lim = max(
        float(
            np.percentile(
                np.abs(exact_gra_diff),
                99.5,
            )
        ),
        1e-6,
    )

    im_eg = ax[1, 4].imshow(
        exact_gra_diff,
        vmin=-eg_lim,
        vmax=eg_lim,
    )
    ax[1, 4].set_title("Exact - GRA")
    ax[1, 4].axis("off")

    im_exact_delta = ax[2, 0].imshow(
        exact_delta,
        vmin=-corr_lim,
        vmax=corr_lim,
    )
    ax[2, 0].set_title("Exact - C4")
    ax[2, 0].axis("off")

    im_gra_delta = ax[2, 1].imshow(
        gra_delta,
        vmin=-corr_lim,
        vmax=corr_lim,
    )
    ax[2, 1].set_title("GRA - C4")
    ax[2, 1].axis("off")

    exact_improve = (
        np.abs(c4 - gt)
        -
        np.abs(exact - gt)
    )

    gra_improve = (
        np.abs(c4 - gt)
        -
        np.abs(gra - gt)
    )

    imp_lim = max(
        float(
            np.percentile(
                np.concatenate(
                    [
                        np.abs(exact_improve).ravel(),
                        np.abs(gra_improve).ravel(),
                    ]
                ),
                99.5,
            )
        ),
        1e-6,
    )

    im_exact_imp = ax[2, 2].imshow(
        exact_improve,
        vmin=-imp_lim,
        vmax=imp_lim,
    )
    ax[2, 2].set_title(
        "C4 error - Exact error\n(+ means improvement)"
    )
    ax[2, 2].axis("off")

    im_gra_imp = ax[2, 3].imshow(
        gra_improve,
        vmin=-imp_lim,
        vmax=imp_lim,
    )
    ax[2, 3].set_title(
        "C4 error - GRA error\n(+ means improvement)"
    )
    ax[2, 3].axis("off")

    ax[2, 4].axis("off")

    text = (
        f"test {sid}\n\n"
        f"C4 MSE       = {mse(c4, gt):.4f}\n"
        f"Exact MSE    = {mse(exact, gt):.4f}\n"
        f"GRA MSE      = {mse(gra, gt):.4f}\n\n"
        f"Exact gain   = "
        f"{100*(mse(c4,gt)-mse(exact,gt))/mse(c4,gt):+.3f}%\n"
        f"GRA gain     = "
        f"{100*(mse(c4,gt)-mse(gra,gt))/mse(c4,gt):+.3f}%\n\n"
        f"mean|Exact-C4| = "
        f"{np.mean(np.abs(exact_delta)):.4f} m/s\n"
        f"mean|GRA-C4|   = "
        f"{np.mean(np.abs(gra_delta)):.4f} m/s\n\n"
        f"HF energy:\n"
        f"GT    {hf_energy(gt):.5f}\n"
        f"C4    {hf_energy(c4):.5f}\n"
        f"Exact {hf_energy(exact):.5f}\n"
        f"GRA   {hf_energy(gra):.5f}"
    )

    ax[2, 4].text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        transform=ax[2, 4].transAxes,
    )

    add_cbar(
        fig,
        im,
        image_axes,
        "Sound speed (m/s)",
    )

    add_cbar(
        fig,
        im_err,
        err_axes,
        "Absolute error (m/s)",
    )

    add_cbar(
        fig,
        im_exact_delta,
        [ax[2, 0], ax[2, 1]],
        "Correction (m/s)",
    )

    add_cbar(
        fig,
        im_exact_imp,
        [ax[2, 2], ax[2, 3]],
        "Pixel error improvement (m/s)",
    )

    fig.suptitle(
        f"test {sid}: Terminal Exact-CBS vs GRA",
        fontsize=15,
    )

    plt.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_roi(
    sid,
    gt,
    c4,
    exact,
    gra,
    out_path,
):
    s = np.s_[90:390, 90:390]

    gt = gt[s]
    c4 = c4[s]
    exact = exact[s]
    gra = gra[s]

    vmin = 1400.0
    vmax = 1605.0

    errs = [
        np.abs(c4 - gt),
        np.abs(exact - gt),
        np.abs(gra - gt),
    ]

    err_lim = float(
        np.percentile(
            np.concatenate(
                [x.ravel() for x in errs]
            ),
            99.0,
        )
    )

    fig, ax = plt.subplots(
        2,
        4,
        figsize=(14, 7),
    )

    im = ax[0, 0].imshow(
        gt,
        vmin=vmin,
        vmax=vmax,
    )
    ax[0, 0].set_title("GT ROI")

    ax[0, 1].imshow(
        c4,
        vmin=vmin,
        vmax=vmax,
    )
    ax[0, 1].set_title(
        f"C4 ROI\nMSE={mse(c4,gt):.2f}"
    )

    ax[0, 2].imshow(
        exact,
        vmin=vmin,
        vmax=vmax,
    )
    ax[0, 2].set_title(
        f"Exact ROI\nMSE={mse(exact,gt):.2f}"
    )

    ax[0, 3].imshow(
        gra,
        vmin=vmin,
        vmax=vmax,
    )
    ax[0, 3].set_title(
        f"GRA ROI\nMSE={mse(gra,gt):.2f}"
    )

    im_err = ax[1, 0].imshow(
        errs[0],
        vmin=0,
        vmax=err_lim,
    )
    ax[1, 0].set_title("|C4-GT| ROI")

    ax[1, 1].imshow(
        errs[1],
        vmin=0,
        vmax=err_lim,
    )
    ax[1, 1].set_title("|Exact-GT| ROI")

    ax[1, 2].imshow(
        errs[2],
        vmin=0,
        vmax=err_lim,
    )
    ax[1, 2].set_title("|GRA-GT| ROI")

    exact_gra = exact - gra

    eg_lim = max(
        float(
            np.percentile(
                np.abs(exact_gra),
                99,
            )
        ),
        1e-6,
    )

    im_diff = ax[1, 3].imshow(
        exact_gra,
        vmin=-eg_lim,
        vmax=eg_lim,
    )
    ax[1, 3].set_title("Exact-GRA ROI")

    for a in ax.ravel():
        a.axis("off")

    add_cbar(
        fig,
        im,
        ax[0, :].tolist(),
        "Sound speed (m/s)",
    )

    add_cbar(
        fig,
        im_err,
        ax[1, :3].tolist(),
        "Absolute error (m/s)",
    )

    add_cbar(
        fig,
        im_diff,
        ax[1, 3],
        "Exact-GRA difference (m/s)",
    )

    fig.suptitle(
        f"test {sid}: Exact-CBS vs GRA in central ROI",
        fontsize=15,
    )

    plt.savefig(
        out_path,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_profile(
    sid,
    gt,
    c4,
    exact,
    gra,
    out_path,
):
    row = 240
    x = np.arange(gt.shape[1])

    fig, ax = plt.subplots(
        figsize=(10, 4.5),
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
        linewidth=1.4,
    )

    ax.plot(
        x,
        exact[row],
        label="Exact-CBS",
        linewidth=1.4,
    )

    ax.plot(
        x,
        gra[row],
        label="GRA",
        linewidth=1.4,
    )

    ax.set_xlim(90, 390)
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Sound speed (m/s)")
    ax.set_title(
        f"test {sid}: horizontal profile through row 240"
    )
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


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
        default="15,19,20",
    )

    ap.add_argument(
        "--exact_step",
        type=float,
        default=1.5,
    )

    args = ap.parse_args()

    bridge_dir = Path(args.bridge_dir)
    gradient_dir = Path(args.gradient_dir)
    out_dir = Path(args.output_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    steps = load_gra_steps(
        args.gra_dir
    )

    samples = [
        int(x)
        for x in args.samples.split(",")
    ]

    rows = []

    print("=" * 110)
    print("C6 EXACT-CBS VS GRA VISUAL AUDIT")
    print("=" * 110)

    for sid in samples:
        bp = (
            bridge_dir
            /
            f"test_{sid}.npz"
        )

        gp = (
            gradient_dir
            /
            f"test_{sid}_gradients.npz"
        )

        if not bp.exists():
            raise FileNotFoundError(bp)

        if not gp.exists():
            raise FileNotFoundError(gp)

        b = np.load(
            bp,
            allow_pickle=True,
        )

        g = np.load(
            gp,
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

        if "exact_gradient" not in g.files:
            raise RuntimeError(
                f"{gp}: exact_gradient missing. "
                f"keys={g.files}"
            )

        if "ra_arss_gradient" not in g.files:
            raise RuntimeError(
                f"{gp}: ra_arss_gradient missing. "
                f"keys={g.files}"
            )

        exact_grad = g[
            "exact_gradient"
        ].astype(np.float64)

        ra_grad = g[
            "ra_arss_gradient"
        ].astype(np.float64)

        if sid not in steps:
            raise RuntimeError(
                f"No GRA accepted step for sample {sid}"
            )

        gra_step = steps[sid]

        exact = reconstruct(
            c4,
            exact_grad,
            args.exact_step,
        )

        gra = reconstruct(
            c4,
            ra_grad,
            gra_step,
        )

        c4_mse = mse(c4, gt)
        exact_mse = mse(exact, gt)
        gra_mse = mse(gra, gt)

        row = {
            "sample": sid,
            "exact_step": args.exact_step,
            "gra_step": gra_step,
            "c4_mse": c4_mse,
            "exact_mse": exact_mse,
            "gra_mse": gra_mse,
            "exact_mse_gain": (
                (c4_mse - exact_mse)
                /
                max(c4_mse, 1e-30)
            ),
            "gra_mse_gain": (
                (c4_mse - gra_mse)
                /
                max(c4_mse, 1e-30)
            ),
            "c4_mae": mae(c4, gt),
            "exact_mae": mae(exact, gt),
            "gra_mae": mae(gra, gt),
            "gt_hf_energy": hf_energy(gt),
            "c4_hf_energy": hf_energy(c4),
            "exact_hf_energy": hf_energy(exact),
            "gra_hf_energy": hf_energy(gra),
            "gt_gradient_energy": gradient_energy(gt),
            "c4_gradient_energy": gradient_energy(c4),
            "exact_gradient_energy": gradient_energy(exact),
            "gra_gradient_energy": gradient_energy(gra),
            "mean_abs_exact_c4": float(
                np.mean(np.abs(exact - c4))
            ),
            "mean_abs_gra_c4": float(
                np.mean(np.abs(gra - c4))
            ),
        }

        rows.append(row)

        print()
        print(f"test {sid:02d}")
        print(
            f"  C4 MSE     = {c4_mse:.6f}"
        )
        print(
            f"  Exact MSE  = {exact_mse:.6f} "
            f"({100*row['exact_mse_gain']:+.3f}%)"
        )
        print(
            f"  GRA MSE    = {gra_mse:.6f} "
            f"({100*row['gra_mse_gain']:+.3f}%)"
        )
        print(
            f"  Exact step = {args.exact_step:g}"
        )
        print(
            f"  GRA step   = {gra_step:g}"
        )
        print(
            "  HF energy  | "
            f"GT={row['gt_hf_energy']:.6f} "
            f"C4={row['c4_hf_energy']:.6f} "
            f"Exact={row['exact_hf_energy']:.6f} "
            f"GRA={row['gra_hf_energy']:.6f}"
        )

        plot_full(
            sid,
            gt,
            hint,
            c4,
            exact,
            gra,
            args.exact_step,
            gra_step,
            out_dir /
            f"test_{sid:02d}_exact_vs_gra_full.png",
        )

        plot_roi(
            sid,
            gt,
            c4,
            exact,
            gra,
            out_dir /
            f"test_{sid:02d}_exact_vs_gra_roi.png",
        )

        plot_profile(
            sid,
            gt,
            c4,
            exact,
            gra,
            out_dir /
            f"test_{sid:02d}_exact_vs_gra_profile.png",
        )

    csv_path = (
        out_dir
        /
        "exact_vs_gra_metrics.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 110)
    print("SAVED")
    print("=" * 110)
    print("metrics =", csv_path)
    print("figures =", out_dir)


if __name__ == "__main__":
    main()
