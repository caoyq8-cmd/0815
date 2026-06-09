import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from cbs_model import ConvergentBornSeries_Batch


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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
        raise ValueError(f"Unsupported measurement_mode: {measurement_mode}")

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


def load_eval_sample(base_dir_dobs_eval: str, base_dir_speed_eval: str, eval_index: int):
    file_id = eval_index + 6601
    file_name = f"train_{file_id}.npy"

    dobs_path = os.path.join(base_dir_dobs_eval, file_name)
    speed_path = os.path.join(base_dir_speed_eval, file_name)

    if not os.path.exists(dobs_path):
        raise FileNotFoundError(f"Cannot find dobs file: {dobs_path}")
    if not os.path.exists(speed_path):
        raise FileNotFoundError(f"Cannot find speed file: {speed_path}")

    dobs_complex = np.load(dobs_path)
    speed_full = np.load(speed_path)
    return file_name, dobs_complex, speed_full


def complex_relative_l1(pred: torch.Tensor, obs: torch.Tensor):
    return (
        torch.mean(torch.abs(pred - obs)) /
        (torch.mean(torch.abs(obs)) + 1e-12)
    ).item()


def magnitude_relative_l1(pred: torch.Tensor, obs: torch.Tensor):
    return (
        torch.mean(torch.abs(torch.abs(pred) - torch.abs(obs))) /
        (torch.mean(torch.abs(obs)) + 1e-12)
    ).item()


def best_complex_scalar(pred: torch.Tensor, obs: torch.Tensor):
    """
    求最优复数标量 c，使 || c*pred - obs ||_2 最小
    c = <pred, obs> / <pred, pred>
    """
    pred_flat = pred.reshape(-1)
    obs_flat = obs.reshape(-1)

    denom = torch.sum(torch.conj(pred_flat) * pred_flat)
    if torch.abs(denom) < 1e-12:
        return torch.tensor(1.0 + 0.0j, device=pred.device, dtype=pred.dtype)

    c = torch.sum(torch.conj(pred_flat) * obs_flat) / denom
    return c


def evaluate_variant(pred: torch.Tensor, obs_variant: torch.Tensor):
    raw_rel_l1 = complex_relative_l1(pred, obs_variant)
    mag_rel_l1 = magnitude_relative_l1(pred, obs_variant)

    c = best_complex_scalar(pred, obs_variant)
    pred_scaled = c * pred
    scaled_rel_l1 = complex_relative_l1(pred_scaled, obs_variant)
    scaled_mag_rel_l1 = magnitude_relative_l1(pred_scaled, obs_variant)

    return {
        "raw_relative_l1": raw_rel_l1,
        "raw_magnitude_relative_l1": mag_rel_l1,
        "best_complex_scale_real": float(torch.real(c).item()),
        "best_complex_scale_imag": float(torch.imag(c).item()),
        "scaled_relative_l1": scaled_rel_l1,
        "scaled_magnitude_relative_l1": scaled_mag_rel_l1,
        "pred_mean_abs": float(torch.mean(torch.abs(pred)).item()),
        "obs_mean_abs": float(torch.mean(torch.abs(obs_variant)).item()),
        "scaled_pred_mean_abs": float(torch.mean(torch.abs(pred_scaled)).item()),
    }


def build_obs_variants(obs: torch.Tensor):
    """
    obs: [1, M, L] complex
    返回不同定义变体
    """
    out = {}
    out["original"] = obs
    out["transpose"] = obs.transpose(1, 2)
    out["conj"] = torch.conj(obs)
    out["conj_transpose"] = torch.conj(obs).transpose(1, 2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir_dobs_eval", type=str, required=True)
    parser.add_argument("--base_dir_speed_eval", type=str, required=True)
    parser.add_argument("--aux_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--eval_indices", type=str, default="0,1,2")
    parser.add_argument("--measurement_mode", type=str, default="sparse", choices=["sparse", "sparse_2"])
    parser.add_argument("--frequency_hz", type=float, default=500000.0)
    parser.add_argument("--boundary_strength", type=float, default=100.0)
    parser.add_argument("--boundary_type", type=str, default="PML2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.output_dir)

    eval_indices = [int(x) for x in args.eval_indices.split(",") if x.strip()]
    transmitter_indices, receiver_indices, receiver_mask = build_geometry(
        args.aux_dir, args.measurement_mode, device
    )

    all_results = []

    for eval_index in eval_indices:
        file_name, dobs_complex_np, speed_full_np = load_eval_sample(
            args.base_dir_dobs_eval,
            args.base_dir_speed_eval,
            eval_index
        )

        gt_sos = torch.tensor(speed_full_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        dobs_obs = torch.tensor(dobs_complex_np, dtype=torch.complex64, device=device).unsqueeze(0)

        if dobs_obs.shape[1] != transmitter_indices.shape[0] or dobs_obs.shape[2] != receiver_indices.shape[0]:
            row = {
                "file_name": file_name,
                "eval_index": eval_index,
                "status": "skipped_shape_mismatch",
                "disk_dobs_shape": list(dobs_complex_np.shape),
                "expected_shape": [int(transmitter_indices.shape[0]), int(receiver_indices.shape[0])],
            }
            all_results.append(row)
            print(row)
            continue

        model = ConvergentBornSeries_Batch(
            f=args.frequency_hz,
            sos=gt_sos,
            boundary_width=[300, 300],
            boundary_strength=args.boundary_strength,
            boundary_type=args.boundary_type,
            device=device,
            src_loc_set=transmitter_indices.cpu().numpy(),
        )

        with torch.no_grad():
            u_gt = model.forward()
            dobs_pred = extract_receiver_data(u_gt, receiver_indices, receiver_mask)

        obs_variants = build_obs_variants(dobs_obs)

        row = {
            "file_name": file_name,
            "eval_index": eval_index,
            "measurement_mode": args.measurement_mode,
            "frequency_hz": args.frequency_hz,
            "boundary_strength": args.boundary_strength,
            "boundary_type": args.boundary_type,
            "disk_dobs_shape": list(dobs_complex_np.shape),
            "status": "ok",
            "variants": {},
        }

        for variant_name, obs_variant in obs_variants.items():
            row["variants"][variant_name] = evaluate_variant(dobs_pred, obs_variant)

        all_results.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False))

    summary_path = os.path.join(args.output_dir, "dobs_definition_check_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 自动挑出每个样本 scaled_relative_l1 最小的 variant
    best_rows = []
    for row in all_results:
        if row.get("status") != "ok":
            continue
        best_variant_name = None
        best_value = None
        for k, v in row["variants"].items():
            val = v["scaled_relative_l1"]
            if best_value is None or val < best_value:
                best_value = val
                best_variant_name = k
        best_rows.append({
            "file_name": row["file_name"],
            "eval_index": row["eval_index"],
            "best_variant": best_variant_name,
            "best_scaled_relative_l1": best_value,
            "best_variant_detail": row["variants"][best_variant_name],
        })

    best_path = os.path.join(args.output_dir, "best_variants.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_rows, f, indent=2, ensure_ascii=False)

    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved best variants to: {best_path}")


if __name__ == "__main__":
    main()