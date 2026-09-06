# plot_c6_final_inverse_results.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sqrt(np.mean(x ** 2))


def normalize_direction(g):
    r = rms(g)
    if r < 1e-12:
        raise RuntimeError("Gradient RMS too small.")
    return np.asarray(g, dtype=np.float64) / r


def load_npz(path):
    return np.load(path, allow_pickle=True)


def find_sample_npz(result_dir, sid):
    pats = [
        f"test_{sid}.npz",
        f"sample_{sid}.npz",
        f"*test_{sid}*.npz",
        f"*sample_{sid}*.npz",
    ]
    for pat in pats:
        hits = list(Path(result_dir).rglob(pat))
        if hits:
            return hits[0]
    return None


def find_summary_csv(result_dir):
    hits = list(Path(result_dir).rglob("summary.csv"))
    if hits:
        return hits[0]
    hits = list(Path(result_dir).rglob("all_steps.csv"))
    if hits:
        return hits[0]
    return None


def pick_key(keys, candidates):
    lower = {k.lower(): k for k in keys}
    for cand in candidates:
        for lk, ok in lower.items():
            if cand in lk:
                return ok
    return None


def reconstruct_trial(candidate, grad, step):
    direction = normalize_direction(grad)
    return (candidate - step * direction).astype(np.float32)


def save_panel(sample_id, panels, out_path):
    n = len(panels)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    value_imgs = [p["img"] for p in panels if not p.get("error", False)]
    vmin = min(img.min() for img in value_imgs)
    vmax = max(img.max() for img in value_imgs)

    error_imgs = [p["img"] for p in panels if p.get("error", False)]
    emax = max(img.max() for img in error_imgs) if error_imgs else 1.0

    for ax in axes.flat:
        ax.axis("off")

    for ax, p in zip(axes.flat, panels):
        img = p["img"]
        if p.get("error", False):
            ax.imshow(img, cmap="inferno", vmin=0, vmax=emax)
        elif p.get("signed", False):
            dmax = np.max(np.abs(img))
            ax.imshow(img, cmap="bwr", vmin=-dmax, vmax=dmax)
        else:
            ax.imshow(img, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(p["title"])
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge_dir", required=True)
    ap.add_argument("--result_dir", required=True,
                    help="C6.4B or C6.6 result directory containing summary.csv and per-sample npz.")
    ap.add_argument("--samples", default="15,16,17,18,20")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    bridge_dir = Path(args.bridge_dir)
    result_dir = Path(args.result_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = [int(x) for x in args.samples.split(",")]

    summary_csv = find_summary_csv(result_dir)
    summary = None
    if summary_csv is not None:
        summary = pd.read_csv(summary_csv)
        print("summary csv =", summary_csv)
    else:
        print("[WARN] summary.csv not found; will use default step=1.5 if needed.")

    step_map = {}
    if summary is not None and "sample" in summary.columns:
        if "accepted_step_mps" in summary.columns:
            for _, row in summary.iterrows():
                step_map[int(row["sample"])] = float(row["accepted_step_mps"])
        elif "step_mps" in summary.columns:
            for _, row in summary.iterrows():
                step_map[int(row["sample"])] = float(row["step_mps"])

    for sid in sample_ids:
        bz = load_npz(bridge_dir / f"test_{sid}.npz")
        gt = bz["target_480"].astype(np.float32)
        hint = bz["hint_480"].astype(np.float32)
        c4 = bz["candidate_480"].astype(np.float32)

        panels = [
            {"title": f"GT (test_{sid})", "img": gt},
            {"title": "Hint", "img": hint},
            {"title": "C4", "img": c4},
            {"title": "|C4-GT|", "img": np.abs(c4 - gt), "error": True},
        ]

        rp = find_sample_npz(result_dir, sid)
        if rp is not None:
            rz = load_npz(rp)
            keys = list(rz.files)

            exact_key = pick_key(keys, ["exact_gradient", "cbs_gradient"])
            full_key = pick_key(keys, ["full_arss_gradient", "full_ano_gradient", "arss_gradient"])
            ra_key = pick_key(keys, ["ra_arss_gradient", "gra_gradient", "ra_gradient"])

            step = step_map.get(sid, 1.5)

            if exact_key is not None:
                exact = reconstruct_trial(c4, rz[exact_key], step if sid in step_map else 1.5)
                panels.append({"title": "Exact-CBS final", "img": exact})
                panels.append({"title": "|Exact-GT|", "img": np.abs(exact - gt), "error": True})

            if full_key is not None:
                full = reconstruct_trial(c4, rz[full_key], step if sid in step_map else 1.5)
                panels.append({"title": "Full-ARSS final", "img": full})
                panels.append({"title": "|Full-ARSS-GT|", "img": np.abs(full - gt), "error": True})

            if ra_key is not None:
                gra = reconstruct_trial(c4, rz[ra_key], step)
                panels.append({"title": f"GRA/RA final (step={step:g})", "img": gra})
                panels.append({"title": "|GRA-GT|", "img": np.abs(gra - gt), "error": True})
                panels.append({"title": "GRA - C4", "img": gra - c4, "signed": True})

            print(f"sample {sid}: npz={rp}")
            print("keys:", keys)
        else:
            print(f"[WARN] No per-sample npz found for sample {sid} under {result_dir}")

        out_path = out_dir / f"final_inverse_panel_test_{sid}.png"
        save_panel(sid, panels, out_path)


if __name__ == "__main__":
    main()