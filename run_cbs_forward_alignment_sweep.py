import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import argparse
import itertools
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


def evaluate_one_setting(
    gt_sos: torch.Tensor,
    dobs_obs: torch.Tensor,
    transmitter_indices: torch.Tensor,
    receiver_indices: torch.Tensor,
    receiver_mask: torch.Tensor,
    frequency_hz: float,
    boundary_strength: float,
    boundary_type: str,
    device: str,
):
    model = ConvergentBornSeries_Batch(
        f=frequency_hz,
        sos=gt_sos,
        boundary_width=[300, 300],
        boundary_strength=boundary_strength,
        boundary_type=boundary_type,
        device=device,
        src_loc_set=transmitter_indices.cpu().numpy(),
    )

    with torch.no_grad():
        u_gt = model.forward()
        dobs_pred = extract_receiver_data(u_gt, receiver_indices, receiver_mask)

    l1 = torch.mean(torch.abs(dobs_pred - dobs_obs)).item()
    rel_l1 = (
        torch.mean(torch.abs(dobs_pred - dobs_obs)) /
        (torch.mean(torch.abs(dobs_obs)) + 1e-12)
    ).item()

    pred_mean_abs = torch.mean(torch.abs(dobs_pred)).item()
    obs_mean_abs = torch.mean(torch.abs(dobs_obs)).item()

    return {
        "dobs_l1": l1,
        "dobs_relative_l1": rel_l1,
        "pred_mean_abs": pred_mean_abs,
        "obs_mean_abs": obs_mean_abs,
    }


def parse_float_list(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def parse_str_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir_dobs_eval", type=str, required=True)
    parser.add_argument("--base_dir_speed_eval", type=str, required=True)
    parser.add_argument("--aux_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--eval_indices", type=str, default="0,1,2")
    parser.add_argument("--measurement_modes", type=str, default="sparse,sparse_2")
    parser.add_argument("--frequencies_hz", type=str, default="500000")
    parser.add_argument("--boundary_strengths", type=str, default="100,150,225,300")
    parser.add_argument("--boundary_types", type=str, default="PML3,PML2")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.output_dir)

    eval_indices = parse_int_list(args.eval_indices)
    measurement_modes = parse_str_list(args.measurement_modes)
    frequencies_hz = parse_float_list(args.frequencies_hz)
    boundary_strengths = parse_float_list(args.boundary_strengths)
    boundary_types = parse_str_list(args.boundary_types)

    all_results = []

    for eval_index in eval_indices:
        file_name, dobs_complex_np, speed_full_np = load_eval_sample(
            args.base_dir_dobs_eval,
            args.base_dir_speed_eval,
            eval_index
        )

        gt_sos = torch.tensor(speed_full_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        for measurement_mode in measurement_modes:
            transmitter_indices, receiver_indices, receiver_mask = build_geometry(
                args.aux_dir, measurement_mode, device
            )

            expected_size = transmitter_indices.shape[0]
            if dobs_complex_np.shape != (expected_size, expected_size):
                # 尺寸不匹配就跳过这组
                all_results.append({
                    "file_name": file_name,
                    "eval_index": eval_index,
                    "measurement_mode": measurement_mode,
                    "status": "skipped_shape_mismatch",
                    "disk_dobs_shape": list(dobs_complex_np.shape),
                    "expected_shape": [expected_size, expected_size],
                })
                continue

            dobs_obs = torch.tensor(dobs_complex_np, dtype=torch.complex64, device=device).unsqueeze(0)

            for frequency_hz, boundary_strength, boundary_type in itertools.product(
                frequencies_hz, boundary_strengths, boundary_types
            ):
                try:
                    metrics = evaluate_one_setting(
                        gt_sos=gt_sos,
                        dobs_obs=dobs_obs,
                        transmitter_indices=transmitter_indices,
                        receiver_indices=receiver_indices,
                        receiver_mask=receiver_mask,
                        frequency_hz=frequency_hz,
                        boundary_strength=boundary_strength,
                        boundary_type=boundary_type,
                        device=device,
                    )

                    row = {
                        "file_name": file_name,
                        "eval_index": eval_index,
                        "measurement_mode": measurement_mode,
                        "frequency_hz": frequency_hz,
                        "boundary_strength": boundary_strength,
                        "boundary_type": boundary_type,
                        "disk_dobs_shape": list(dobs_complex_np.shape),
                        "status": "ok",
                        **metrics,
                    }
                except Exception as e:
                    row = {
                        "file_name": file_name,
                        "eval_index": eval_index,
                        "measurement_mode": measurement_mode,
                        "frequency_hz": frequency_hz,
                        "boundary_strength": boundary_strength,
                        "boundary_type": boundary_type,
                        "disk_dobs_shape": list(dobs_complex_np.shape),
                        "status": "error",
                        "error": str(e),
                    }

                all_results.append(row)
                print(row)

    summary_path = os.path.join(args.output_dir, "forward_alignment_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    ok_rows = [r for r in all_results if r.get("status") == "ok"]
    if len(ok_rows) > 0:
        best_row = min(ok_rows, key=lambda r: r["dobs_relative_l1"])
        best_path = os.path.join(args.output_dir, "best_setting.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best_row, f, indent=2, ensure_ascii=False)
        print("\nBest setting by relative L1:")
        print(json.dumps(best_row, indent=2, ensure_ascii=False))

    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()