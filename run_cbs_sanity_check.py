import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.io import loadmat

from cbs_model import ConvergentBornSeries_Batch
from utils_cbs import normalize as cbs_normalize
from utils_cbs import denormalize as cbs_denormalize
from train_inversionnet_baseline import InversionNetBaseline


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_inversion_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    inv_cfg = ckpt["config"]
    model = InversionNetBaseline(
        in_channels=2,
        out_channels=1,
        base_ch=inv_cfg["base_ch"],
        bottleneck_blocks=inv_cfg["bottleneck_blocks"],
        dropout=inv_cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def build_geometry(aux_dir: str, measurement_mode: str, device: str):
    x_pos = loadmat(os.path.join(aux_dir, "x_pos.mat"))["x_pos256"]
    y_pos = loadmat(os.path.join(aux_dir, "y_pos.mat"))["y_pos256"]

    all_indices = torch.cat(
        (
            torch.tensor(x_pos.astype(np.int64)),
            torch.tensor(y_pos.astype(np.int64)),
        ),
        dim=1
    )

    if measurement_mode == "sparse":
        num_keep = 64
    elif measurement_mode == "sparse_2":
        num_keep = 32
    else:
        raise ValueError(measurement_mode)

    if all_indices.shape[0] == num_keep:
        transmitter_indices = all_indices.contiguous()
    else:
        subsample = all_indices.shape[0] // num_keep
        transmitter_indices = all_indices[::subsample].contiguous()

    receiver_indices = transmitter_indices.clone()
    receiver_mask = torch.ones((num_keep, num_keep), dtype=torch.float32, device=device)
    return transmitter_indices, receiver_indices, receiver_mask


def extract_receiver_data(u_batch, receiver_indices, receiver_mask=None):
    rec = u_batch[:, :, receiver_indices[:, 0], receiver_indices[:, 1]]
    if receiver_mask is not None:
        rec = rec * receiver_mask
    return rec


def make_inversion_input_from_dobs(dobs_complex: np.ndarray, target_size: int = 256):
    inp = np.stack([dobs_complex.real, dobs_complex.imag], axis=0).astype(np.float32)
    inp_t = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
    if inp_t.shape[-1] != target_size or inp_t.shape[-2] != target_size:
        inp_t = F.interpolate(inp_t, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return inp_t


def save_img(x, path, title=None, cmap="inferno"):
    plt.figure(figsize=(5, 5))
    plt.imshow(x, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def summarize_tensor(x):
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inversion_ckpt", type=str, required=True)
    parser.add_argument("--base_dir_dobs_eval", type=str, required=True)
    parser.add_argument("--base_dir_speed_eval", type=str, required=True)
    parser.add_argument("--aux_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eval_index", type=int, default=0)
    parser.add_argument("--measurement_mode", type=str, default="sparse")
    parser.add_argument("--frequency_hz", type=float, default=500e3)
    parser.add_argument("--boundary_strength", type=float, default=225.0)
    parser.add_argument("--boundary_type", type=str, default="PML3")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.output_dir)

    file_id = args.eval_index + 6601
    file_name = f"train_{file_id}.npy"

    dobs_path = os.path.join(args.base_dir_dobs_eval, file_name)
    speed_path = os.path.join(args.base_dir_speed_eval, file_name)

    dobs_complex = np.load(dobs_path)
    speed_full = np.load(speed_path)

    gt_sos = torch.tensor(speed_full, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    dobs_obs = torch.tensor(dobs_complex, dtype=torch.complex64, device=device).unsqueeze(0)

    transmitter_indices, receiver_indices, receiver_mask = build_geometry(
        args.aux_dir, args.measurement_mode, device
    )

    # A. GT -> CBS forward
    cbs_model = ConvergentBornSeries_Batch(
        f=args.frequency_hz,
        sos=gt_sos,
        boundary_width=[300, 300],
        boundary_strength=args.boundary_strength,
        boundary_type=args.boundary_type,
        device=device,
        src_loc_set=transmitter_indices.cpu().numpy(),
    )
    with torch.no_grad():
        u_gt = cbs_model.forward()
        dobs_from_gt = extract_receiver_data(u_gt, receiver_indices, receiver_mask)

    dobs_l1 = torch.mean(torch.abs(dobs_from_gt - dobs_obs)).item()
    dobs_rel = (
        torch.mean(torch.abs(dobs_from_gt - dobs_obs)) /
        (torch.mean(torch.abs(dobs_obs)) + 1e-12)
    ).item()

    # B. GT normalize-denormalize roundtrip
    with torch.no_grad():
        gt_norm = cbs_normalize(gt_sos)
        gt_roundtrip = cbs_denormalize(gt_norm)

    roundtrip_mse = F.mse_loss(gt_roundtrip, gt_sos).item()
    roundtrip_mae = F.l1_loss(gt_roundtrip, gt_sos).item()

    # C. InversionNet 480 bridge stats
    inversion_model = load_inversion_model(args.inversion_ckpt, device)
    inversion_input = make_inversion_input_from_dobs(dobs_complex, target_size=256).to(device)

    with torch.no_grad():
        inv_pred_norm = inversion_model(inversion_input)
        inv_sos_480 = cbs_denormalize(inv_pred_norm)

    inv_vs_gt_mse = F.mse_loss(inv_sos_480, gt_sos).item()
    inv_vs_gt_mae = F.l1_loss(inv_sos_480, gt_sos).item()

    roi = slice(90, 390)
    gt_roi = gt_sos[:, :, roi, roi]
    inv_roi = inv_sos_480[:, :, roi, roi]

    bg_mask = torch.ones_like(gt_sos)
    bg_mask[:, :, roi, roi] = 0.0
    bg_count = bg_mask.sum()

    gt_bg_mean = float((gt_sos * bg_mask).sum().item() / bg_count.item())
    inv_bg_mean = float((inv_sos_480 * bg_mask).sum().item() / bg_count.item())

    save_img(gt_sos[0, 0].detach().cpu().numpy(), os.path.join(args.output_dir, "gt_480.png"), "GT 480")
    save_img(inv_sos_480[0, 0].detach().cpu().numpy(), os.path.join(args.output_dir, "inv_480.png"), "Inversion 480")
    save_img((inv_sos_480[0, 0] - gt_sos[0, 0]).detach().cpu().numpy(), os.path.join(args.output_dir, "inv_minus_gt.png"), "Inv - GT", cmap="bwr")

    summary = {
        "file_name": file_name,
        "dobs_shape_disk": list(dobs_complex.shape),
        "geometry_num_src": int(transmitter_indices.shape[0]),
        "geometry_num_rec": int(receiver_indices.shape[0]),
        "gt_forward_check": {
            "dobs_l1": dobs_l1,
            "dobs_relative_l1": dobs_rel,
        },
        "normalize_denormalize_roundtrip": {
            "roundtrip_mse": roundtrip_mse,
            "roundtrip_mae": roundtrip_mae,
        },
        "inversion_bridge_stats": {
            "inv_vs_gt_mse": inv_vs_gt_mse,
            "inv_vs_gt_mae": inv_vs_gt_mae,
            "gt_full": summarize_tensor(gt_sos),
            "inv_full": summarize_tensor(inv_sos_480),
            "gt_roi": summarize_tensor(gt_roi),
            "inv_roi": summarize_tensor(inv_roi),
            "gt_bg_mean": gt_bg_mean,
            "inv_bg_mean": inv_bg_mean,
        }
    }

    with open(os.path.join(args.output_dir, "sanity_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()