#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E3.5: multi-step structure-preserving Local-AA on DEV30 (old test21-test50).

This runner extends ``run_adjoint_aware_multistep_correction_v2.py`` without
changing its neural gradient, CBS forward solver, step sizes, clipping, or
fallback rule.  At every iteration it:

1. obtains the Local-AA neural gradient;
2. evaluates the same zero / multi-scale candidate pool with true CBS;
3. keeps candidates that retain at least ``rho`` of the best available physics
   improvement at that iteration; and
4. among those candidates, selects the smallest drift from the fixed
   InversionNet initialization.

The selection rule uses only the measurement and the InversionNet condition.
Ground truth is loaded solely for post-selection metrics.

``rho=1`` reproduces pure best-physics Local-AA selection (up to exact ties).
Smaller ``rho`` trades some of the best available physics gain for lower image
and finite-difference-gradient drift. ``gamma`` weights gradient drift relative
to value drift after candidate-pool normalization.

The script deliberately guards the development split.  By default it selects
numeric base IDs 21 through 50 and requires all 30 IDs to be present.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def rms(x: np.ndarray) -> float:
    x64 = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x64 * x64)) + 1e-12)


def normalize_rms(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) / rms(x)


def finite_difference_drift(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute finite-difference drift in m/s per parameter-grid pixel."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    dx = np.mean(np.abs((a[:, 1:] - a[:, :-1]) - (b[:, 1:] - b[:, :-1])))
    dy = np.mean(np.abs((a[1:, :] - a[:-1, :]) - (b[1:, :] - b[:-1, :])))
    return float(0.5 * (dx + dy))


def add_structure_scores(
    candidates: List[Dict[str, Any]],
    condition: np.ndarray,
    gamma: float,
) -> None:
    """Attach condition-relative, pool-normalized structure scores in place."""
    max_value = 0.0
    max_grad = 0.0
    for candidate in candidates:
        speed = candidate["speed"]
        value = float(np.mean(np.abs(speed - condition)))
        grad = finite_difference_drift(speed, condition)
        candidate["value_drift"] = value
        candidate["grad_drift"] = grad
        max_value = max(max_value, value)
        max_grad = max(max_grad, grad)

    for candidate in candidates:
        value_norm = candidate["value_drift"] / (max_value + 1e-12)
        grad_norm = candidate["grad_drift"] / (max_grad + 1e-12)
        candidate["value_drift_norm"] = float(value_norm)
        candidate["grad_drift_norm"] = float(grad_norm)
        candidate["structure_score"] = float(value_norm + gamma * grad_norm)


def choose_candidate(
    candidates: List[Dict[str, Any]],
    current_physics: float,
    rho: float,
    gamma: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Choose without GT: physics-retention constraint, then structure drift."""
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must lie in (0, 1].")
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative.")

    add_structure_scores(candidates, candidates[0]["condition"], gamma)
    best_physics = min(float(c["physics_loss"]) for c in candidates)
    best_gain = max(float(current_physics) - best_physics, 0.0)

    if best_gain <= max(abs(float(current_physics)), 1.0) * 1e-12:
        zero = next(c for c in candidates if c["direction"] == "zero")
        return zero, {
            "best_physics": best_physics,
            "best_gain": 0.0,
            "required_gain": 0.0,
            "num_eligible": 1,
        }

    required_gain = float(rho) * best_gain
    tolerance = max(abs(float(current_physics)), 1.0) * 1e-12
    eligible = [
        c
        for c in candidates
        if float(current_physics) - float(c["physics_loss"])
        >= required_gain - tolerance
    ]
    if not eligible:
        eligible = [min(candidates, key=lambda c: float(c["physics_loss"]))]

    # Physics loss is the deterministic tie breaker.  The final factor term
    # prefers the smaller step only if both preceding quantities are equal.
    selected = min(
        eligible,
        key=lambda c: (
            float(c["structure_score"]),
            float(c["physics_loss"]),
            float(c["factor"]),
        ),
    )
    return selected, {
        "best_physics": best_physics,
        "best_gain": best_gain,
        "required_gain": required_gain,
        "num_eligible": len(eligible),
    }


def float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def setting_tag(rho: float, gamma: float) -> str:
    return f"rho{float_tag(rho)}_gamma{float_tag(gamma)}"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric_summary(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": math.nan, "sample_std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(arr)),
        "sample_std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def bootstrap_mean_ci(values: Sequence[float], seed: int, draws: int = 10000) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    chunk = 1000
    offset = 0
    while offset < draws:
        count = min(chunk, draws - offset)
        indices = rng.integers(0, arr.size, size=(count, arr.size))
        means[offset : offset + count] = arr[indices].mean(axis=1)
        offset += count
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def aggregate_rows(rows: Sequence[Mapping[str, Any]], seed: int) -> Dict[str, Any]:
    if not rows:
        return {"num_samples": 0}

    metric_keys = [
        "init_true_cbs_mse",
        "final_true_cbs_mse",
        "true_cbs_mse_rel_reduction",
        "init_true_cbs_rrmse",
        "final_true_cbs_rrmse",
        "init_mse_240",
        "final_mse_240",
        "mse240_rel_reduction",
        "init_mae_240",
        "final_mae_240",
        "init_psnr_240",
        "final_psnr_240",
        "init_ssim_240",
        "final_ssim_240",
        "final_value_drift",
        "final_grad_drift",
        "cbs_forward_calls_algorithm",
        "neural_grad_calls",
        "num_updates",
        "runtime_sec",
    ]
    summary: Dict[str, Any] = {"num_samples": len(rows), "metrics": {}}
    for key in metric_keys:
        if all(key in row for row in rows):
            summary["metrics"][key] = numeric_summary(float(row[key]) for row in rows)

    improvements = {
        "physics_mse": [
            float(r["init_true_cbs_mse"]) - float(r["final_true_cbs_mse"])
            for r in rows
        ],
        "image_mse_240": [
            float(r["init_mse_240"]) - float(r["final_mse_240"])
            for r in rows
        ],
        "image_mae_240": [
            float(r["init_mae_240"]) - float(r["final_mae_240"])
            for r in rows
        ],
        "image_psnr_240": [
            float(r["final_psnr_240"]) - float(r["init_psnr_240"])
            for r in rows
        ],
        "image_ssim_240": [
            float(r["final_ssim_240"]) - float(r["init_ssim_240"])
            for r in rows
        ],
    }
    summary["paired_improvements"] = {}
    for index, (name, values) in enumerate(improvements.items()):
        item: Dict[str, Any] = numeric_summary(values)
        item["bootstrap_mean_95ci"] = bootstrap_mean_ci(values, seed + index)
        item["wins"] = int(np.sum(np.asarray(values) > 0.0))
        item["ties"] = int(np.sum(np.asarray(values) == 0.0))
        item["losses"] = int(np.sum(np.asarray(values) < 0.0))
        try:
            from scipy.stats import wilcoxon

            nonzero = np.asarray(values, dtype=np.float64)
            if np.any(np.abs(nonzero) > 0.0):
                item["wilcoxon_two_sided_p"] = float(
                    wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox").pvalue
                )
                item["wilcoxon_improvement_p"] = float(
                    wilcoxon(nonzero, alternative="greater", zero_method="wilcox").pvalue
                )
        except Exception:
            pass
        summary["paired_improvements"][name] = item
    return summary


def load_core(repo_root: Path):
    repo_root = repo_root.resolve()
    required = [
        repo_root / "run_adjoint_aware_multistep_correction_v2.py",
        repo_root / "run_neural_measurement_correction.py",
        repo_root / "cbs_model.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "E3.5 must be placed in the latest 0815 experiment repository root. "
            "Missing:\n  " + "\n  ".join(missing)
        )
    sys.path.insert(0, str(repo_root))
    return importlib.import_module("run_adjoint_aware_multistep_correction_v2")


def select_dev_groups(core, data_root: Path, split: str, start_id: int, end_id: int):
    groups_all = core.load_groups(str(data_root), split)
    selected = {
        base: records
        for base, records in groups_all.items()
        if start_id <= int(core.numeric_key(base)) <= end_id
    }
    return groups_all, selected


def load_completed_row(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def run_setting(
    *,
    args,
    core,
    rho: float,
    gamma: float,
    groups,
    model,
    model_context: Mapping[str, Any],
    device,
    setting_dir: Path,
) -> List[Dict[str, Any]]:
    setting_dir.mkdir(parents=True, exist_ok=True)
    setting_config = dict(vars(args))
    setting_config.update(
        {
            "rho": rho,
            "gamma": gamma,
            "selection_uses_gt": False,
            "condition_reference": "fixed alpha=0 InversionNet initialization",
            "candidate_validation": "true CBS",
        }
    )
    config_path = setting_dir / "config.json"
    if args.resume and config_path.is_file():
        with config_path.open("r", encoding="utf-8") as stream:
            previous_config = json.load(stream)
        protected_keys = [
            "data_root", "split", "ckpt_path", "mean_dobs_path",
            # sample_id_start/end and max_base intentionally are not protected:
            # smoke3 -> screen10 -> DEV30 expands the evaluated set while safely
            # reusing per-sample rows computed with identical algorithm settings.
            "init_alpha", "gt_alpha",
            "rho", "gamma", "num_steps", "base_step_mps", "step_factors",
            "speed_min", "speed_max", "fallback_opposite", "stop_on_no_update",
            "frequency", "forward_iters", "boundary_width",
            "boundary_strength", "boundary_type",
        ]
        changed = [
            key
            for key in protected_keys
            if previous_config.get(key) != setting_config.get(key)
        ]
        if changed:
            raise RuntimeError(
                f"Resume guard failed for {setting_dir}: protected settings changed: "
                f"{changed}. Use a new output_dir, or pass --no_resume to recompute."
            )
    with config_path.open("w", encoding="utf-8") as stream:
        json.dump(setting_config, stream, indent=2, ensure_ascii=False)

    mean_real = model_context["mean_real"]
    mean_imag = model_context["mean_imag"]
    residual_scale = model_context["residual_scale"]
    speed_center = model_context["speed_center"]
    speed_scale = model_context["speed_scale"]
    y_map = model_context["y_map"]
    x_map = model_context["x_map"]
    image_size = int(model_context["image_size"])

    rows: List[Dict[str, Any]] = []
    for position, (base, records) in enumerate(groups.items(), start=1):
        sample_dir = setting_dir / base
        row_path = sample_dir / "result_row.json"
        if args.resume:
            cached = load_completed_row(row_path)
            if cached is not None:
                print(f"[{position:02d}/{len(groups):02d}] {base} resume: completed")
                rows.append(cached)
                continue

        started = time.time()
        init_path = core.find_alpha_record(records, args.init_alpha)
        gt_path = core.find_alpha_record(records, args.gt_alpha)
        zi = np.load(init_path, allow_pickle=True)
        zg = np.load(gt_path, allow_pickle=True)

        init_480_stored = zi["target_480"].astype(np.float32)
        gt_480 = zg["target_480"].astype(np.float32)
        target_dobs = zg["dobs_complex"].astype(np.complex64)
        src_indices = zi["src_indices"].astype(np.int64)
        rec_indices = zi["rec_indices"].astype(np.int64)

        condition = core.resize_np(init_480_stored, image_size).astype(np.float32)
        current = condition.copy()
        gt_240 = core.resize_np(gt_480, image_size).astype(np.float32)

        current_480 = core.resize_np(current, 480).astype(np.float32)
        pred_true = core.forward_cbs_dobs(
            current_480, src_indices, rec_indices, args, device
        )
        cbs_forward_calls = 1
        neural_grad_calls = 0

        current_mse = core.complex_mse(pred_true, target_dobs)
        current_mae = core.complex_mae(pred_true, target_dobs)
        current_rrmse = core.complex_rrmse(pred_true, target_dobs)
        init_m240 = core.image_metrics(current, gt_240)
        init_m480 = core.image_metrics(current_480, gt_480)

        history: List[Dict[str, Any]] = [
            {
                "iter": 0,
                "selected_direction": "init",
                "selected_factor": 0.0,
                "true_cbs_mse": current_mse,
                "true_cbs_mae": current_mae,
                "true_cbs_rrmse": current_rrmse,
                "image_mse_240": init_m240["mse"],
                "image_mae_240": init_m240["mae"],
                "image_psnr_240": init_m240["psnr"],
                "image_ssim_240": init_m240["ssim"],
                "value_drift": 0.0,
                "grad_drift": 0.0,
                "candidate_records": [],
            }
        ]

        print(
            f"[{position:02d}/{len(groups):02d}] {base} INIT | "
            f"CBS-MSE={current_mse:.6e} | imgMSE240={init_m240['mse']:.3f}"
        )

        for iteration in range(1, args.num_steps + 1):
            neural_loss, raw_grad, _ = core.compute_neural_grad(
                model=model,
                speed_np=current,
                target_dobs=target_dobs,
                mean_real=mean_real,
                mean_imag=mean_imag,
                residual_scale=residual_scale,
                speed_center=speed_center,
                speed_scale=speed_scale,
                y_map=y_map,
                x_map=x_map,
                device=device,
                lambda_l1=0.0,
                lambda_mse=1.0,
            )
            neural_grad_calls += 1
            direction = normalize_rms(raw_grad)

            candidates: List[Dict[str, Any]] = [
                {
                    "direction": "zero",
                    "factor": 0.0,
                    "speed": current.copy(),
                    "physics_loss": float(current_mse),
                    "true_pred": pred_true,
                    "condition": condition,
                }
            ]

            def evaluate(sign: float, label: str) -> List[Dict[str, Any]]:
                nonlocal cbs_forward_calls
                result = []
                for factor in args.step_factors:
                    step = args.base_step_mps * float(factor)
                    candidate_speed = np.clip(
                        current + sign * step * direction,
                        args.speed_min,
                        args.speed_max,
                    ).astype(np.float32)
                    candidate_480 = core.resize_np(candidate_speed, 480).astype(np.float32)
                    candidate_pred = core.forward_cbs_dobs(
                        candidate_480, src_indices, rec_indices, args, device
                    )
                    cbs_forward_calls += 1
                    result.append(
                        {
                            "direction": label,
                            "factor": float(factor),
                            "speed": candidate_speed,
                            "physics_loss": core.complex_mse(candidate_pred, target_dobs),
                            "true_pred": candidate_pred,
                            "condition": condition,
                        }
                    )
                return result

            candidates.extend(evaluate(-1.0, "-"))
            best_minus = min(candidates, key=lambda c: float(c["physics_loss"]))
            if args.fallback_opposite and best_minus["direction"] == "zero":
                candidates.extend(evaluate(+1.0, "+"))

            physics_before_update = float(current_mse)
            selected, selection = choose_candidate(
                candidates, physics_before_update, rho=rho, gamma=gamma
            )
            updated = selected["direction"] != "zero"
            if updated:
                current = selected["speed"].astype(np.float32)
                pred_true = selected["true_pred"]
                current_mse = float(selected["physics_loss"])

            current_mae = core.complex_mae(pred_true, target_dobs)
            current_rrmse = core.complex_rrmse(pred_true, target_dobs)
            current_480 = core.resize_np(current, 480).astype(np.float32)
            metrics_240 = core.image_metrics(current, gt_240)
            metrics_480 = core.image_metrics(current_480, gt_480)

            candidate_records = [
                {
                    "direction": candidate["direction"],
                    "factor": candidate["factor"],
                    "physics_loss": candidate["physics_loss"],
                    "value_drift": candidate["value_drift"],
                    "grad_drift": candidate["grad_drift"],
                    "structure_score": candidate["structure_score"],
                    "eligible": int(
                        physics_before_update - float(candidate["physics_loss"])
                        >= selection["required_gain"] - 1e-12
                    ),
                }
                for candidate in candidates
            ]
            history.append(
                {
                    "iter": iteration,
                    "neural_loss": float(neural_loss),
                    "raw_grad_rms": rms(raw_grad),
                    "rho": rho,
                    "gamma": gamma,
                    "selected_direction": selected["direction"],
                    "selected_factor": selected["factor"],
                    "updated": int(updated),
                    "best_available_physics_gain": selection["best_gain"],
                    "required_physics_gain": selection["required_gain"],
                    "num_eligible": selection["num_eligible"],
                    "true_cbs_mse": current_mse,
                    "true_cbs_mae": current_mae,
                    "true_cbs_rrmse": current_rrmse,
                    "image_mse_240": metrics_240["mse"],
                    "image_mae_240": metrics_240["mae"],
                    "image_psnr_240": metrics_240["psnr"],
                    "image_ssim_240": metrics_240["ssim"],
                    "image_mse_480": metrics_480["mse"],
                    "value_drift": selected["value_drift"],
                    "grad_drift": selected["grad_drift"],
                    "structure_score": selected["structure_score"],
                    "candidate_records": candidate_records,
                }
            )
            print(
                f"    iter={iteration:02d} | choose={selected['direction']}"
                f"{selected['factor']:.2g} | eligible={selection['num_eligible']} | "
                f"CBS-MSE={current_mse:.6e} | imgMSE240={metrics_240['mse']:.3f} | "
                f"SSIM={metrics_240['ssim']:.4f}"
            )
            if not updated and args.stop_on_no_update:
                print("    early stop: no true-CBS improving candidate")
                break

        final_480 = core.resize_np(current, 480).astype(np.float32)
        final_m240 = core.image_metrics(current, gt_240)
        final_m480 = core.image_metrics(final_480, gt_480)
        sample_dir.mkdir(parents=True, exist_ok=True)
        with (sample_dir / "history.json").open("w", encoding="utf-8") as stream:
            json.dump(history, stream, indent=2, ensure_ascii=False)
        np.savez_compressed(
            sample_dir / "final_result.npz",
            init_speed_240=condition.astype(np.float32),
            final_speed_240=current.astype(np.float32),
            gt_speed_240=gt_240.astype(np.float32),
            final_speed_480=final_480.astype(np.float32),
            gt_speed_480=gt_480.astype(np.float32),
            target_dobs=target_dobs.astype(np.complex64),
            final_true_cbs_dobs=pred_true.astype(np.complex64),
            rho=np.asarray(rho, dtype=np.float32),
            gamma=np.asarray(gamma, dtype=np.float32),
        )

        init_true_mse = float(history[0]["true_cbs_mse"])
        final_true_mse = float(history[-1]["true_cbs_mse"])
        final_state = history[-1]
        row: Dict[str, Any] = {
            "base_sample": base,
            "sample_id": int(core.numeric_key(base)),
            "rho": rho,
            "gamma": gamma,
            "num_updates": int(sum(int(h.get("updated", 0)) for h in history[1:])),
            "num_iters_run": len(history) - 1,
            "init_true_cbs_mse": init_true_mse,
            "final_true_cbs_mse": final_true_mse,
            "true_cbs_mse_rel_reduction": float(
                (init_true_mse - final_true_mse) / (init_true_mse + 1e-20)
            ),
            "init_true_cbs_rrmse": float(history[0]["true_cbs_rrmse"]),
            "final_true_cbs_rrmse": float(history[-1]["true_cbs_rrmse"]),
            "init_mse_240": init_m240["mse"],
            "final_mse_240": final_m240["mse"],
            "mse240_rel_reduction": float(
                (init_m240["mse"] - final_m240["mse"]) / (init_m240["mse"] + 1e-20)
            ),
            "init_mae_240": init_m240["mae"],
            "final_mae_240": final_m240["mae"],
            "init_psnr_240": init_m240["psnr"],
            "final_psnr_240": final_m240["psnr"],
            "init_ssim_240": init_m240["ssim"],
            "final_ssim_240": final_m240["ssim"],
            "init_mse_480": init_m480["mse"],
            "final_mse_480": final_m480["mse"],
            "final_value_drift": float(final_state.get("value_drift", 0.0)),
            "final_grad_drift": float(final_state.get("grad_drift", 0.0)),
            "cbs_forward_calls_algorithm": cbs_forward_calls,
            "cbs_adjoint_calls": 0,
            "neural_grad_calls": neural_grad_calls,
            "runtime_sec": float(time.time() - started),
        }
        with row_path.open("w", encoding="utf-8") as stream:
            json.dump(row, stream, indent=2, ensure_ascii=False)
        rows.append(row)
        print(
            f"[{base}] FINAL | physics={100*row['true_cbs_mse_rel_reduction']:.2f}% | "
            f"MSE240={100*row['mse240_rel_reduction']:.2f}% | "
            f"CBS-fw={cbs_forward_calls} | time={row['runtime_sec']:.1f}s"
        )

        # Keep partial aggregate files current so an interrupted long run is auditable.
        write_csv(setting_dir / "sample_results.partial.csv", rows)

    rows = sorted(rows, key=lambda row: int(row["sample_id"]))
    write_csv(setting_dir / "sample_results.csv", rows)
    summary = aggregate_rows(rows, seed=args.seed)
    summary["rho"] = rho
    summary["gamma"] = gamma
    with (setting_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return rows


def compare_with_baseline(
    structure_rows: Sequence[Mapping[str, Any]], baseline_csv: Path, seed: int
) -> Dict[str, Any]:
    with baseline_csv.open("r", encoding="utf-8-sig") as stream:
        baseline_rows = list(csv.DictReader(stream))
    baseline = {str(row["base_sample"]): row for row in baseline_rows}
    pairs = [(row, baseline[str(row["base_sample"])]) for row in structure_rows if str(row["base_sample"]) in baseline]
    result: Dict[str, Any] = {
        "baseline_csv": str(baseline_csv.resolve()),
        "num_pairs": len(pairs),
        "positive_delta_means_structure_better": True,
        "metrics": {},
    }
    definitions = {
        "final_true_cbs_mse": -1.0,
        "final_mse_240": -1.0,
        "final_mae_240": -1.0,
        "final_psnr_240": +1.0,
        "final_ssim_240": +1.0,
    }
    for index, (key, sign) in enumerate(definitions.items()):
        values = [sign * (float(s[key]) - float(b[key])) for s, b in pairs]
        item: Dict[str, Any] = numeric_summary(values)
        item["bootstrap_mean_95ci"] = bootstrap_mean_ci(values, seed + 100 + index)
        item["wins"] = int(np.sum(np.asarray(values) > 0.0))
        item["losses"] = int(np.sum(np.asarray(values) < 0.0))
        try:
            from scipy.stats import wilcoxon

            if values and np.any(np.abs(values) > 0.0):
                item["wilcoxon_two_sided_p"] = float(
                    wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue
                )
        except Exception:
            pass
        result["metrics"][key] = item
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Multi-step E3.5 structure-preserving Local-AA on old test21-test50.",
    )
    parser.add_argument("--repo_root", default=".")
    parser.add_argument(
        "--data_root",
        default="./dev30_test21_50/local_alpha01",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--ckpt_path",
        default="./adjoint_aware_runs/clean_reverse_train20_val10_lf1_lg3/checkpoints/epoch_001.pth",
    )
    parser.add_argument(
        "--mean_dobs_path",
        default="./dobs_mean_baseline/local_alpha_train100_test20/mean_dobs_train.npz",
    )
    parser.add_argument("--output_dir", default="./e35_structure_preserving_dev30")
    parser.add_argument(
        "--baseline_csv",
        default="./dev30_test21_50/results/local_aa/sample_results.csv",
    )

    parser.add_argument("--sample_id_start", type=int, default=21)
    parser.add_argument("--sample_id_end", type=int, default=50)
    parser.add_argument("--max_base", type=int, default=-1)
    parser.add_argument("--allow_non_dev30", action="store_true")
    parser.add_argument("--init_alpha", type=float, default=0.0)
    parser.add_argument("--gt_alpha", type=float, default=1.0)

    parser.add_argument("--rhos", nargs="+", type=float, default=[0.5])
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.5])
    parser.add_argument("--num_steps", type=int, default=8)
    parser.add_argument("--base_step_mps", type=float, default=0.5)
    parser.add_argument("--step_factors", nargs="+", type=float, default=[1.0, 0.5, 0.25, 0.1])
    parser.add_argument("--speed_min", type=float, default=1400.0)
    parser.add_argument("--speed_max", type=float, default=1605.0)
    parser.add_argument("--fallback_opposite", action="store_true", default=True)
    parser.add_argument("--no_fallback_opposite", dest="fallback_opposite", action="store_false")
    parser.add_argument("--stop_on_no_update", action="store_true", default=True)
    parser.add_argument("--no_stop_on_no_update", dest="stop_on_no_update", action="store_false")

    parser.add_argument("--frequency", type=float, default=500000.0)
    parser.add_argument("--forward_iters", type=int, default=80)
    parser.add_argument("--boundary_width", type=int, default=300)
    parser.add_argument("--boundary_strength", type=float, default=225.0)
    parser.add_argument("--boundary_type", default="PML3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for rho in args.rhos:
        if not 0.0 < rho <= 1.0:
            raise ValueError("Every --rhos value must lie in (0, 1].")
    for gamma in args.gammas:
        if gamma < 0.0:
            raise ValueError("Every --gammas value must be non-negative.")

    repo_root = Path(args.repo_root).resolve()
    core = load_core(repo_root)
    data_root = (repo_root / args.data_root).resolve() if not Path(args.data_root).is_absolute() else Path(args.data_root)
    ckpt_path = (repo_root / args.ckpt_path).resolve() if not Path(args.ckpt_path).is_absolute() else Path(args.ckpt_path)
    mean_path = (repo_root / args.mean_dobs_path).resolve() if not Path(args.mean_dobs_path).is_absolute() else Path(args.mean_dobs_path)
    output_dir = (repo_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    groups_all, groups = select_dev_groups(
        core, data_root, args.split, args.sample_id_start, args.sample_id_end
    )
    expected_ids = list(range(args.sample_id_start, args.sample_id_end + 1))
    found_ids = sorted(int(core.numeric_key(base)) for base in groups)
    if not args.allow_non_dev30 and found_ids != expected_ids:
        raise RuntimeError(
            "DEV30 guard failed. Expected numeric IDs "
            f"{expected_ids[0]}-{expected_ids[-1]} ({len(expected_ids)} samples), "
            f"found {found_ids}. Use --allow_non_dev30 only for a deliberate smoke/screen run."
        )
    if args.max_base > 0:
        groups = dict(list(groups.items())[: args.max_base])
    if not groups:
        raise RuntimeError("No matching alpha groups were found.")

    print("=" * 108)
    print("E3.5 structure-preserving Local-AA")
    print("=" * 108)
    print("repo_root       =", repo_root)
    print("data_root       =", data_root)
    print("all groups      =", len(groups_all))
    print("selected bases  =", list(groups.keys()))
    print("rhos            =", args.rhos)
    print("gammas          =", args.gammas)
    print("frozen steps    =", args.num_steps, args.base_step_mps, args.step_factors)
    print("CBS             =", args.frequency, args.forward_iters, args.boundary_type,
          args.boundary_width, args.boundary_strength)
    print("selection GT    = False")
    print("=" * 108)

    if args.dry_run:
        print("[DRY RUN PASS] DEV split and script dependencies are available.")
        return
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Local-AA checkpoint not found: {ckpt_path}")
    if not mean_path.is_file():
        raise FileNotFoundError(f"mean_dobs file not found: {mean_path}")

    import torch

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, ckpt_args = core.load_residual_measurement_model(str(ckpt_path), device)
    image_size = int(ckpt_args.get("image_size", 240))
    mean_dobs = np.load(mean_path)["mean_dobs"].astype(np.complex64)
    model_context = {
        "image_size": image_size,
        "speed_center": float(ckpt_args.get("speed_center", 1500.0)),
        "speed_scale": float(ckpt_args.get("speed_scale", 100.0)),
        "residual_scale": float(ckpt_args.get("residual_scale", 0.02)),
        "mean_real": torch.from_numpy(mean_dobs.real.astype(np.float32)).to(device),
        "mean_imag": torch.from_numpy(mean_dobs.imag.astype(np.float32)).to(device),
    }
    model_context["y_map"], model_context["x_map"] = core.make_coord_maps(image_size, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    top_summary_rows: List[Dict[str, Any]] = []
    for rho in args.rhos:
        for gamma in args.gammas:
            tag = setting_tag(rho, gamma)
            print("\n" + "#" * 108)
            print("SETTING", tag)
            print("#" * 108)
            rows = run_setting(
                args=args,
                core=core,
                rho=float(rho),
                gamma=float(gamma),
                groups=groups,
                model=model,
                model_context=model_context,
                device=device,
                setting_dir=output_dir / tag,
            )
            all_rows[tag] = rows
            summary = aggregate_rows(rows, seed=args.seed)
            metric = summary["metrics"]
            wins = summary["paired_improvements"]
            top_summary_rows.append(
                {
                    "setting": tag,
                    "rho": rho,
                    "gamma": gamma,
                    "n": len(rows),
                    "final_mse_240_mean": metric["final_mse_240"]["mean"],
                    "final_mse_240_sample_std": metric["final_mse_240"]["sample_std"],
                    "final_mae_240_mean": metric["final_mae_240"]["mean"],
                    "final_psnr_240_mean": metric["final_psnr_240"]["mean"],
                    "final_ssim_240_mean": metric["final_ssim_240"]["mean"],
                    "physics_rel_reduction_mean": metric["true_cbs_mse_rel_reduction"]["mean"],
                    "image_mse_rel_reduction_mean": metric["mse240_rel_reduction"]["mean"],
                    "physics_wins_vs_init": wins["physics_mse"]["wins"],
                    "mse_wins_vs_init": wins["image_mse_240"]["wins"],
                    "mae_wins_vs_init": wins["image_mae_240"]["wins"],
                    "ssim_wins_vs_init": wins["image_ssim_240"]["wins"],
                    "cbs_forward_mean": metric["cbs_forward_calls_algorithm"]["mean"],
                    "runtime_mean_sec": metric["runtime_sec"]["mean"],
                }
            )

            if args.baseline_csv:
                baseline_path = Path(args.baseline_csv)
                if not baseline_path.is_absolute():
                    baseline_path = repo_root / baseline_path
                comparison = compare_with_baseline(rows, baseline_path, args.seed)
                with (output_dir / tag / "paired_vs_local_aa.json").open(
                    "w", encoding="utf-8"
                ) as stream:
                    json.dump(comparison, stream, indent=2, ensure_ascii=False)

    write_csv(output_dir / "dev30_summary.csv", top_summary_rows)
    with (output_dir / "experiment_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "dataset_role": "DEV30; development results, not strict independent holdout",
                "sample_ids": sorted(int(core.numeric_key(base)) for base in groups),
                "settings": list(all_rows),
                "frozen_comparison_parameters": {
                    "num_steps": args.num_steps,
                    "base_step_mps": args.base_step_mps,
                    "step_factors": args.step_factors,
                    "frequency": args.frequency,
                    "forward_iters": args.forward_iters,
                    "boundary_width": args.boundary_width,
                    "boundary_strength": args.boundary_strength,
                    "boundary_type": args.boundary_type,
                    "fallback_opposite": args.fallback_opposite,
                    "stop_on_no_update": args.stop_on_no_update,
                },
                "selection_uses_gt": False,
            },
            stream,
            indent=2,
            ensure_ascii=False,
        )
    print("\n[DONE]", output_dir)
    print("Key table:", output_dir / "dev30_summary.csv")


if __name__ == "__main__":
    main()
