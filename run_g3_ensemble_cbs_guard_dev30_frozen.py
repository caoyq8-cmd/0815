#!/usr/bin/env python3
"""One-shot frozen DEV30 evaluation of G3-Ensemble and its true-CBS guard.

All method hyperparameters were frozen on VAL20 and are constants in this file.
DEV30=test21..50 is evaluation-only: this script performs no search or tuning.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from screen_ddpm_sdedit_val20 import load_ddpm, safe_paired, sdedit_ddim
from train_icrfm_oof_v2 import (
    ConditionCacheDataset,
    denormalize_speed,
    file_manifest,
    integrate_flow,
    json_dump,
    load_model,
    seed_everything,
    write_csv,
)
from validate_g3_ensemble_true_cbs_val20 import (
    complex_mae,
    complex_mse,
    complex_rrmse,
    find_alpha,
    forward_cbs,
    image_metrics,
    load_groups,
    numeric_key,
    resize_np,
    to_2d,
)


FROZEN = {
    "eval_split": "test",
    "eval_start": 21,
    "eval_end": 50,
    "flow_steps": 4,
    "flow_solver": "heun",
    "ddpm_t_start": 25,
    "ddpm_steps": 5,
    "alpha_flow": 0.03,
    "alpha_prior": 0.15,
    "seeds": [20260902, 20260903, 20260904, 20260905, 20260906],
    "frequency": 500000.0,
    "forward_iters": 80,
    "boundary_width": 300,
    "boundary_strength": 225.0,
    "boundary_type": "PML3",
    "speed_min": 1400.0,
    "speed_max": 1605.0,
    "alignment_rrmse_tol": 1e-3,
    "min_image_mse_gain": 0.05,
    "min_image_mse_wins": 15,
    "image_mae_tolerance": 0.0,
    "image_ssim_guard": 0.0,
    "min_raw_physics_wins": 15,
}


def mean_std(values: Sequence[float]) -> Dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def paired_bundle(rows: Sequence[Dict], method: str, seed: int, n_boot: int) -> Dict:
    output = {"physics": {}, "image": {}}
    for offset, metric in enumerate(("mse", "mae", "rrmse")):
        delta = np.asarray(
            [
                row[f"original_physics_{metric}"]
                - row[f"{method}_physics_{metric}"]
                for row in rows
            ],
            dtype=np.float64,
        )
        output["physics"][metric] = safe_paired(delta, seed + offset, n_boot)
    for offset, metric in enumerate(("mse", "mae", "psnr", "ssim"), start=10):
        if metric in ("mse", "mae"):
            delta = np.asarray(
                [
                    row[f"original_image_{metric}"]
                    - row[f"{method}_image_{metric}"]
                    for row in rows
                ],
                dtype=np.float64,
            )
        else:
            delta = np.asarray(
                [
                    row[f"{method}_image_{metric}"]
                    - row[f"original_image_{metric}"]
                    for row in rows
                ],
                dtype=np.float64,
            )
        output["image"][metric] = safe_paired(delta, seed + offset, n_boot)
    return output


def summarize(rows: Sequence[Dict]) -> Dict:
    result: Dict = {"num_samples": len(rows)}
    for method in ("original", "raw_g3", "guarded"):
        for metric in ("mse", "mae", "rrmse"):
            result[f"{method}_physics_{metric}"] = mean_std(
                [row[f"{method}_physics_{metric}"] for row in rows]
            )
        for metric in ("mse", "mae", "rmse", "psnr", "ssim"):
            result[f"{method}_image_{metric}"] = mean_std(
                [row[f"{method}_image_{metric}"] for row in rows]
            )
    result["raw_g3_physics_mse_wins"] = int(
        sum(row["raw_g3_physics_mse"] < row["original_physics_mse"] for row in rows)
    )
    result["raw_g3_image_mse_wins"] = int(
        sum(row["raw_g3_image_mse"] < row["original_image_mse"] for row in rows)
    )
    result["guarded_image_mse_wins"] = int(
        sum(row["guarded_image_mse"] < row["original_image_mse"] for row in rows)
    )
    result["guarded_selected_g3"] = int(sum(row["guard_selected_g3"] for row in rows))
    result["guarded_selected_original"] = len(rows) - result["guarded_selected_g3"]
    result["generation_runtime_sec"] = mean_std(
        [row["generation_runtime_sec"] for row in rows]
    )
    result["cbs_runtime_sec"] = mean_std([row["cbs_runtime_sec"] for row in rows])
    result["total_runtime_sec"] = mean_std([row["total_runtime_sec"] for row in rows])
    return result


def method_summary_rows(summary: Dict) -> List[Dict]:
    rows = []
    labels = {
        "original": "Original",
        "raw_g3": "G3-Ensemble",
        "guarded": "G3-Ensemble-CBSGuard",
    }
    for key, label in labels.items():
        # Keep an identical schema for every method.  The repository's shared
        # write_csv helper takes its field names from the first row, so the
        # Original row must also declare the two comparison-only columns.
        row = {
            "method": label,
            "num_samples": summary["num_samples"],
            "physics_mse_wins_vs_original": "",
            "image_mse_wins_vs_original": "",
        }
        for domain, metrics in (
            ("image", ("mse", "mae", "rmse", "psnr", "ssim")),
            ("physics", ("mse", "mae", "rrmse")),
        ):
            for metric in metrics:
                stats = summary[f"{key}_{domain}_{metric}"]
                row[f"{domain}_{metric}_mean"] = stats["mean"]
                row[f"{domain}_{metric}_std"] = stats["std"]
        if key == "raw_g3":
            row["physics_mse_wins_vs_original"] = summary["raw_g3_physics_mse_wins"]
            row["image_mse_wins_vs_original"] = summary["raw_g3_image_mse_wins"]
        elif key == "guarded":
            row["physics_mse_wins_vs_original"] = summary["guarded_selected_g3"]
            row["image_mse_wins_vs_original"] = summary["guarded_image_mse_wins"]
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--condition_root", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--flow_ckpt", required=True)
    ap.add_argument("--ddpm_ckpt", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--bootstrap_samples", type=int, default=10000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    prediction_dir = out / "ensemble_predictions"
    cache_dir = out / "forward_cache"
    guarded_dir = out / "guarded_predictions"
    for directory in (prediction_dir, cache_dir, guarded_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config = {
        **FROZEN,
        "condition_root": str(Path(args.condition_root).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "flow_ckpt": str(Path(args.flow_ckpt).resolve()),
        "ddpm_ckpt": str(Path(args.ddpm_ckpt).resolve()),
    }
    config_path = out / "config_frozen.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        changed = [key for key, value in config.items() if old.get(key) != value]
        if changed:
            raise RuntimeError(
                f"frozen DEV30 resume guard: protected settings changed: {changed}"
            )
    else:
        json_dump(config_path, config)

    seed_everything(FROZEN["seeds"][0])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    flow_model, flow_cfg, flow_ckpt = load_model(args.flow_ckpt, device)
    ddpm_model, diffusion, ddpm_cfg, ddpm_ckpt = load_ddpm(args.ddpm_ckpt, device)
    speed_min = FROZEN["speed_min"]
    speed_max = FROZEN["speed_max"]
    data_range = speed_max - speed_min
    flow_min = float(flow_cfg.get("speed_min", speed_min))
    flow_max = float(flow_cfg.get("speed_max", speed_max))
    ddpm_min = float(ddpm_cfg.get("norm_min", speed_min))
    ddpm_max = float(ddpm_cfg.get("norm_max", speed_max))
    if any(
        abs(left - right) > 1e-6
        for left, right in (
            (flow_min, speed_min),
            (flow_max, speed_max),
            (ddpm_min, speed_min),
            (ddpm_max, speed_max),
        )
    ):
        raise RuntimeError("checkpoint normalization does not match the frozen range")

    dataset = ConditionCacheDataset(
        args.condition_root,
        FROZEN["eval_split"],
        FROZEN["eval_start"],
        FROZEN["eval_end"],
        -1,
        False,
    )
    ids = [numeric_key(path) for path in dataset.files]
    if ids != list(range(21, 51)):
        raise RuntimeError(f"DEV30 guard expected test21..50, got {ids}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("=" * 126)
    print("Frozen G3-Ensemble generation on DEV30")
    print("IDs                = test21..50")
    print("flow               = 4-step Heun, alpha=0.03")
    print("DDPM               = t25, NFE5, five-seed mean, alpha=0.15")
    print("selection/tuning   = disabled")
    print("=" * 126)

    generation_runtime: Dict[int, float] = {}
    flow_model.eval()
    ddpm_model.eval()
    with torch.no_grad():
        for batch in loader:
            start_time = time.perf_counter()
            condition = batch["condition"].to(device, non_blocking=True)
            sample_ids = batch["sample_id"]
            flow_endpoint = integrate_flow(
                flow_model,
                condition,
                FROZEN["flow_steps"],
                FROZEN["flow_solver"],
                noise_scale=0.0,
            )
            prior_endpoints = []
            for noise_seed in FROZEN["seeds"]:
                endpoint, _ = sdedit_ddim(
                    ddpm_model,
                    diffusion,
                    condition,
                    sample_ids,
                    FROZEN["ddpm_t_start"],
                    FROZEN["ddpm_steps"],
                    noise_seed,
                )
                prior_endpoints.append(endpoint)
            prior_mean = torch.stack(prior_endpoints, dim=0).mean(dim=0)
            prediction = torch.clamp(
                condition
                + FROZEN["alpha_flow"] * (flow_endpoint - condition)
                + FROZEN["alpha_prior"] * (prior_mean - condition),
                -1.0,
                1.0,
            )
            prediction_speed = denormalize_speed(
                prediction, speed_min, speed_max
            ).cpu().numpy()
            elapsed_per_sample = (time.perf_counter() - start_time) / condition.shape[0]
            for index, sample_id in enumerate(sample_ids.numpy()):
                sample_id = int(sample_id)
                generation_runtime[sample_id] = float(elapsed_per_sample)
                np.savez_compressed(
                    prediction_dir / f"test_{sample_id}.npz",
                    sample_id=np.int64(sample_id),
                    seeds=np.asarray(FROZEN["seeds"], dtype=np.int64),
                    condition_norm=condition[index].cpu().numpy().astype(np.float32),
                    target_speed=batch["target_speed"][index].numpy().astype(np.float32),
                    flow_endpoint_norm=flow_endpoint[index].cpu().numpy().astype(np.float32),
                    ddpm_endpoint_mean_norm=prior_mean[index].cpu().numpy().astype(np.float32),
                    hybrid_norm=prediction[index].cpu().numpy().astype(np.float32),
                    hybrid_speed=prediction_speed[index].astype(np.float32),
                )
                print(f"[generated] test_{sample_id}")

    groups_all = load_groups(args.data_root, "test")
    expected_bases = [f"test_{index}" for index in range(21, 51)]
    groups = [(base, groups_all[base]) for base in expected_bases if base in groups_all]
    if [base for base, _ in groups] != expected_bases:
        raise RuntimeError(
            f"local-alpha DEV30 must contain {expected_bases}; got {[x[0] for x in groups]}"
        )

    cbs_args = SimpleNamespace(
        frequency=FROZEN["frequency"],
        forward_iters=FROZEN["forward_iters"],
        boundary_width=FROZEN["boundary_width"],
        boundary_strength=FROZEN["boundary_strength"],
        boundary_type=FROZEN["boundary_type"],
    )
    rows: List[Dict] = []
    for position, (base, records) in enumerate(groups, start=1):
        sample_id = numeric_key(base)
        init_path = find_alpha(records, 0.0)
        gt_path = find_alpha(records, 1.0)
        prediction_path = prediction_dir / f"{base}.npz"
        with np.load(init_path, allow_pickle=True) as init_data, np.load(
            gt_path, allow_pickle=True
        ) as gt_data, np.load(prediction_path) as pred_data:
            original_480 = init_data["target_480"].astype(np.float32)
            stored_original_dobs = init_data["dobs_complex"].astype(np.complex64)
            target_dobs = gt_data["dobs_complex"].astype(np.complex64)
            src_indices = init_data["src_indices"].astype(np.int64)
            rec_indices = init_data["rec_indices"].astype(np.int64)
            condition_image = to_2d(pred_data["condition_norm"])
            condition_image = condition_image * (0.5 * data_range) + 0.5 * (
                speed_min + speed_max
            )
            target_image = to_2d(pred_data["target_speed"])
            raw_g3_image = to_2d(pred_data["hybrid_speed"])
            raw_g3_480 = np.clip(
                resize_np(raw_g3_image.T, 480), speed_min, speed_max
            ).astype(np.float32)

        cache_path = cache_dir / f"{base}.npz"
        cbs_start = time.perf_counter()
        if cache_path.exists() and not args.no_resume:
            with np.load(cache_path) as cache:
                original_dobs = cache["original_dobs"].astype(np.complex64)
                raw_g3_dobs = cache["raw_g3_dobs"].astype(np.complex64)
        else:
            original_dobs = forward_cbs(
                original_480, src_indices, rec_indices, cbs_args, device
            )
            raw_g3_dobs = forward_cbs(
                raw_g3_480, src_indices, rec_indices, cbs_args, device
            )
            np.savez_compressed(
                cache_path,
                original_dobs=original_dobs,
                raw_g3_dobs=raw_g3_dobs,
                target_dobs=target_dobs,
            )
        cbs_runtime = time.perf_counter() - cbs_start

        original_physics = {
            "mse": complex_mse(original_dobs, target_dobs),
            "mae": complex_mae(original_dobs, target_dobs),
            "rrmse": complex_rrmse(original_dobs, target_dobs),
        }
        raw_physics = {
            "mse": complex_mse(raw_g3_dobs, target_dobs),
            "mae": complex_mae(raw_g3_dobs, target_dobs),
            "rrmse": complex_rrmse(raw_g3_dobs, target_dobs),
        }
        choose_g3 = raw_physics["mse"] < original_physics["mse"]
        guarded_physics = raw_physics if choose_g3 else original_physics
        guarded_image = raw_g3_image if choose_g3 else condition_image
        original_image_metrics = image_metrics(condition_image, target_image, data_range)
        raw_image_metrics = image_metrics(raw_g3_image, target_image, data_range)
        guarded_image_metrics = image_metrics(guarded_image, target_image, data_range)
        alignment = complex_rrmse(original_dobs, stored_original_dobs)
        row: Dict = {
            "sample_id": sample_id,
            "base_sample": base,
            "alignment_rrmse": alignment,
            "guard_selected_g3": bool(choose_g3),
            "generation_runtime_sec": generation_runtime[sample_id],
            "cbs_runtime_sec": float(cbs_runtime),
            "total_runtime_sec": float(generation_runtime[sample_id] + cbs_runtime),
            "neural_flow_nfe": 8,
            "neural_ddpm_nfe": 25,
            "cbs_forward_calls": 2,
            "cbs_adjoint_calls": 0,
        }
        for name, metrics in (
            ("original_physics", original_physics),
            ("raw_g3_physics", raw_physics),
            ("guarded_physics", guarded_physics),
            ("original_image", original_image_metrics),
            ("raw_g3_image", raw_image_metrics),
            ("guarded_image", guarded_image_metrics),
        ):
            row.update({f"{name}_{key}": value for key, value in metrics.items()})
        rows.append(row)
        np.savez_compressed(
            guarded_dir / f"{base}.npz",
            sample_id=np.int64(sample_id),
            selected_g3=np.bool_(choose_g3),
            selected_speed_image=guarded_image.astype(np.float32),
            original_speed_image=condition_image.astype(np.float32),
            raw_g3_speed_image=raw_g3_image.astype(np.float32),
            target_speed_image=target_image.astype(np.float32),
            original_physics_mse=np.float64(original_physics["mse"]),
            raw_g3_physics_mse=np.float64(raw_physics["mse"]),
        )
        print(
            f"[{position:02d}/30] {base} | align={alignment:.2e} | "
            f"CBS {original_physics['mse']:.6e}->{raw_physics['mse']:.6e} | "
            f"guard={'G3' if choose_g3 else 'Original'}"
        )

    summary = summarize(rows)
    alignment_max = max(row["alignment_rrmse"] for row in rows)
    alignment_pass = alignment_max <= FROZEN["alignment_rrmse_tol"]

    def image_guard(method: str) -> Dict:
        mse_gain = (
            summary["original_image_mse"]["mean"]
            - summary[f"{method}_image_mse"]["mean"]
        )
        mae_gain = (
            summary["original_image_mae"]["mean"]
            - summary[f"{method}_image_mae"]["mean"]
        )
        ssim_gain = (
            summary[f"{method}_image_ssim"]["mean"]
            - summary["original_image_ssim"]["mean"]
        )
        mse_wins = summary[f"{method}_image_mse_wins"]
        return {
            "mse_gain": mse_gain,
            "mae_gain": mae_gain,
            "ssim_gain": ssim_gain,
            "mse_wins": mse_wins,
            "pass": bool(
                mse_gain >= FROZEN["min_image_mse_gain"]
                and mse_wins >= FROZEN["min_image_mse_wins"]
                and mae_gain >= -FROZEN["image_mae_tolerance"]
                and ssim_gain >= -FROZEN["image_ssim_guard"]
            ),
        }

    raw_image_guard = image_guard("raw_g3")
    guarded_image_guard = image_guard("guarded")
    raw_physics_gain = (
        summary["original_physics_mse"]["mean"]
        - summary["raw_g3_physics_mse"]["mean"]
    )
    raw_physics_pass = bool(
        raw_physics_gain >= 0
        and summary["raw_g3_physics_mse_wins"] >= FROZEN["min_raw_physics_wins"]
    )
    if not alignment_pass:
        status = "ALIGNMENT_FAIL"
    elif guarded_image_guard["pass"] and summary["guarded_selected_g3"] > 0:
        status = "DEV30_CBS_GUARDED_PASS"
    elif raw_image_guard["pass"] and raw_physics_pass:
        status = "DEV30_RAW_G3_PASS"
    else:
        status = "DEV30_REJECT"

    paired = {
        "raw_g3_vs_original": paired_bundle(
            rows, "raw_g3", FROZEN["seeds"][0] + 700000, args.bootstrap_samples
        ),
        "guarded_vs_original": paired_bundle(
            rows, "guarded", FROZEN["seeds"][0] + 800000, args.bootstrap_samples
        ),
    }
    write_csv(out / "dev30_per_sample.csv", rows)
    write_csv(out / "dev30_method_summary.csv", method_summary_rows(summary))
    json_dump(out / "paired_statistics.json", paired)
    json_dump(
        out / "dev30_evaluation.json",
        {
            "status": status,
            "note": "frozen one-shot DEV30 evaluation; no retuning is permitted",
            "alignment_pass": alignment_pass,
            "alignment_rrmse_max": alignment_max,
            "raw_physics_gain": raw_physics_gain,
            "raw_physics_pass": raw_physics_pass,
            "raw_image_guard": raw_image_guard,
            "guarded_image_guard": guarded_image_guard,
            "summary": summary,
            "frozen_config": config,
            "flow_checkpoint_epoch": int(flow_ckpt.get("epoch", -1)),
            "ddpm_checkpoint_epoch": int(ddpm_ckpt.get("epoch", -1)),
            "eval_files": file_manifest(dataset.files),
        },
    )

    print("=" * 126)
    print("status                    =", status)
    print("alignment max RRMSE       =", f"{alignment_max:.3e}")
    print(
        "raw physics MSE           =",
        f"{summary['original_physics_mse']['mean']:.6e}",
        "->",
        f"{summary['raw_g3_physics_mse']['mean']:.6e}",
        f"wins={summary['raw_g3_physics_mse_wins']}/30",
    )
    print("CBS Guard selected G3    =", f"{summary['guarded_selected_g3']}/30")
    print("raw image guard          =", raw_image_guard)
    print("guarded image guard      =", guarded_image_guard)
    print("saved to:", out.resolve())


if __name__ == "__main__":
    main()
