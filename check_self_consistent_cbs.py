import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from cbs_model import ConvergentBornSeries_Batch


@torch.no_grad()
def rerun_cbs(target_480, src_indices, rec_indices, args, device):
    sos = torch.from_numpy(target_480.astype(np.float32))[None, None].to(device)

    model = ConvergentBornSeries_Batch(
        f=args.frequency,
        sos=sos,
        boundary_width=[args.boundary_width, args.boundary_width],
        boundary_strength=args.boundary_strength,
        boundary_type=args.boundary_type,
        src_loc_set=src_indices.astype(np.int64),
        device=device,
    )

    u = model(max_iters=args.cbs_iters)

    rec_t = torch.from_numpy(rec_indices.astype(np.int64)).to(device)
    dobs_new = u[0, :, rec_t[:, 0], rec_t[:, 1]]

    dobs_new = dobs_new.detach().cpu().numpy().astype(np.complex64)

    del model, u, sos
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dobs_new


def rel_l1(a, b):
    return float(np.mean(np.abs(a - b)) / (np.mean(np.abs(b)) + 1e-12))


def rel_l2(a, b):
    return float(np.sqrt(np.mean(np.abs(a - b) ** 2)) / (np.sqrt(np.mean(np.abs(b) ** 2)) + 1e-12))


def save_complex_vis(dobs_old, dobs_new, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    items = [
        ("saved_real", dobs_old.real, "seismic"),
        ("saved_imag", dobs_old.imag, "seismic"),
        ("rerun_real", dobs_new.real, "seismic"),
        ("rerun_imag", dobs_new.imag, "seismic"),
        ("abs_diff", np.abs(dobs_new - dobs_old), "viridis"),
    ]

    for name, img, cmap in items:
        plt.figure(figsize=(5, 4))
        plt.imshow(img, cmap=cmap)
        plt.colorbar()
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./self_consistent_check")

    parser.add_argument("--frequency", type=float, default=500e3)
    parser.add_argument("--cbs_iters", type=int, default=80)

    parser.add_argument("--boundary_width", type=int, default=300)
    parser.add_argument("--boundary_strength", type=float, default=225.0)
    parser.add_argument("--boundary_type", type=str, default="PML3")

    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    data = np.load(args.sample_path)

    target_480 = data["target_480"]
    dobs_saved = data["dobs_complex"]
    src_indices = data["src_indices"]
    rec_indices = data["rec_indices"]

    print("sample_path =", args.sample_path)
    print("target_480 shape =", target_480.shape)
    print("dobs_saved shape =", dobs_saved.shape)
    print("src/rec shape    =", src_indices.shape, rec_indices.shape)

    dobs_new = rerun_cbs(target_480, src_indices, rec_indices, args, device)

    abs_diff = np.abs(dobs_new - dobs_saved)

    metrics = {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rel_l1": rel_l1(dobs_new, dobs_saved),
        "rel_l2": rel_l2(dobs_new, dobs_saved),
        "saved_abs_mean": float(np.abs(dobs_saved).mean()),
        "rerun_abs_mean": float(np.abs(dobs_new).mean()),
    }

    print("=" * 80)
    print("[Self-consistency metrics]")
    print("=" * 80)
    for k, v in metrics.items():
        print(f"{k}: {v:.12e}")

    save_complex_vis(dobs_saved, dobs_new, args.output_dir)

    with open(os.path.join(args.output_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.12e}\n")

    print("saved visuals to:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()