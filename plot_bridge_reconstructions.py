# plot_bridge_reconstructions.py
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_npz(path):
    return np.load(path, allow_pickle=True)


def save_panel(sample_id, gt, hint, c4, out_dir, vmin=None, vmax=None):
    err_hint = np.abs(hint - gt)
    err_c4 = np.abs(c4 - gt)

    if vmin is None:
        vmin = min(gt.min(), hint.min(), c4.min())
    if vmax is None:
        vmax = max(gt.max(), hint.max(), c4.max())

    emax = max(err_hint.max(), err_c4.max())

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    ims = []
    ims.append(axes[0, 0].imshow(gt, cmap="inferno", vmin=vmin, vmax=vmax))
    axes[0, 0].set_title(f"GT (test_{sample_id})")
    axes[0, 0].axis("off")

    ims.append(axes[0, 1].imshow(hint, cmap="inferno", vmin=vmin, vmax=vmax))
    axes[0, 1].set_title("Hint")
    axes[0, 1].axis("off")

    ims.append(axes[0, 2].imshow(c4, cmap="inferno", vmin=vmin, vmax=vmax))
    axes[0, 2].set_title("C4 candidate")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(err_hint, cmap="inferno", vmin=0, vmax=emax)
    axes[1, 0].set_title("|Hint - GT|")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(err_c4, cmap="inferno", vmin=0, vmax=emax)
    axes[1, 1].set_title("|C4 - GT|")
    axes[1, 1].axis("off")

    diff = c4 - hint
    dmax = np.max(np.abs(diff))
    axes[1, 2].imshow(diff, cmap="bwr", vmin=-dmax, vmax=dmax)
    axes[1, 2].set_title("C4 - Hint")
    axes[1, 2].axis("off")

    plt.tight_layout()
    out_path = out_dir / f"bridge_panel_test_{sample_id}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge_dir", required=True)
    ap.add_argument("--samples", default="11,12,15,16,17,18,20")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    bridge_dir = Path(args.bridge_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = [int(x) for x in args.samples.split(",")]

    # first pass: global color range
    all_vals = []
    for sid in sample_ids:
        z = load_npz(bridge_dir / f"test_{sid}.npz")
        all_vals.append(z["target_480"].astype(np.float32))
        all_vals.append(z["hint_480"].astype(np.float32))
        all_vals.append(z["candidate_480"].astype(np.float32))

    vmin = min(x.min() for x in all_vals)
    vmax = max(x.max() for x in all_vals)

    for sid in sample_ids:
        z = load_npz(bridge_dir / f"test_{sid}.npz")
        gt = z["target_480"].astype(np.float32)
        hint = z["hint_480"].astype(np.float32)
        c4 = z["candidate_480"].astype(np.float32)
        save_panel(sid, gt, hint, c4, out_dir, vmin=vmin, vmax=vmax)


if __name__ == "__main__":
    main()