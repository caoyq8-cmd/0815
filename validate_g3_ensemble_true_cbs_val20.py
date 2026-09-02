#!/usr/bin/env python3
"""True-CBS validation of the frozen G3-Ensemble on VAL20.

Two preregistered methods are reported:

1. Raw G3-Ensemble.
2. G3-Ensemble + CBS Guard, which selects Original or G3 using only the
   observed-data true-CBS loss.  Ground truth is used only for evaluation.

The test split is locked to test1..20 and no weights are tuned here.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cbs_model import ConvergentBornSeries_Batch
from screen_ddpm_sdedit_val20 import safe_paired

try:
    from skimage.metrics import structural_similarity
except ImportError as exc:
    raise RuntimeError("scikit-image is required for SSIM evaluation") from exc


def numeric_key(value) -> int:
    numbers = re.findall(r"\d+", str(value))
    return int(numbers[-1]) if numbers else -1


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, payload: Dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def load_groups(data_root: str, split: str) -> Dict[str, List[Tuple[float, Path]]]:
    files = sorted((Path(data_root) / split).glob(f"{split}_*.npz"), key=numeric_key)
    groups = defaultdict(list)
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            base = str(data["base_sample"])
            alpha = float(data["alpha"])
        groups[base].append((alpha, path))
    for base in groups:
        groups[base].sort(key=lambda item: item[0])
    return dict(sorted(groups.items(), key=lambda item: numeric_key(item[0])))


def find_alpha(records: Sequence[Tuple[float, Path]], target: float) -> Path:
    matches = [(abs(alpha - target), path) for alpha, path in records]
    if not matches:
        raise RuntimeError("empty local-alpha group")
    distance, path = min(matches, key=lambda item: item[0])
    if distance > 1e-6:
        raise RuntimeError(f"missing alpha={target}")
    return path


def to_2d(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    while value.ndim > 2:
        value = value[0]
    return value.astype(np.float32)


def resize_np(array: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(array, np.float32))[None, None]
    resized = F.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False
    )
    return resized[0, 0].numpy().astype(np.float32)


def complex_mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target) ** 2))


def complex_mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def complex_rrmse(pred: np.ndarray, target: np.ndarray) -> float:
    numerator = np.sqrt(np.mean(np.abs(pred - target) ** 2))
    denominator = np.sqrt(np.mean(np.abs(target) ** 2)) + 1e-12
    return float(numerator / denominator)


def image_metrics(pred: np.ndarray, target: np.ndarray, data_range: float) -> Dict:
    diff = np.asarray(pred, np.float32) - np.asarray(target, np.float32)
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    psnr = 99.0 if mse <= 1e-16 else float(20 * np.log10(data_range / rmse))
    ssim = float(structural_similarity(target, pred, data_range=data_range))
    return {"mse": mse, "mae": mae, "rmse": rmse, "psnr": psnr, "ssim": ssim}


@torch.no_grad()
def forward_cbs(
    speed_480: np.ndarray,
    src_indices: np.ndarray,
    rec_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    speed_tensor = torch.from_numpy(speed_480.astype(np.float32))[None, None].to(device)
    model = ConvergentBornSeries_Batch(
        f=args.frequency,
        sos=speed_tensor,
        boundary_width=[args.boundary_width, args.boundary_width],
        boundary_strength=args.boundary_strength,
        boundary_type=args.boundary_type,
        src_loc_set=src_indices.astype(np.int64),
        device=device,
    )
    wavefield = model(max_iters=args.forward_iters)
    receivers = torch.from_numpy(rec_indices.astype(np.int64)).long().to(device)
    dobs = wavefield[0, :, receivers[:, 0], receivers[:, 1]]
    result = dobs.detach().cpu().numpy().astype(np.complex64)
    del model, wavefield, speed_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def mean_std(values: Sequence[float]) -> Dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def improvement(rows: Sequence[Dict], prefix: str, metric: str) -> np.ndarray:
    if metric in ("mse", "mae", "rrmse"):
        return np.asarray(
            [r[f"original_{prefix}_{metric}"] - r[f"{prefix}_{metric}"] for r in rows],
            dtype=np.float64,
        )
    return np.asarray(
        [r[f"{prefix}_{metric}"] - r[f"original_{prefix}_{metric}"] for r in rows],
        dtype=np.float64,
    )


def summarize(rows: Sequence[Dict]) -> Dict:
    summary: Dict = {"num_samples": len(rows)}
    for method in ("original", "raw_g3", "guarded"):
        for metric in ("mse", "mae", "rrmse"):
            summary[f"{method}_physics_{metric}"] = mean_std(
                [r[f"{method}_physics_{metric}"] for r in rows]
            )
        for metric in ("mse", "mae", "psnr", "ssim"):
            summary[f"{method}_image_{metric}"] = mean_std(
                [r[f"{method}_image_{metric}"] for r in rows]
            )
    summary["raw_g3_physics_mse_wins"] = int(
        sum(r["raw_g3_physics_mse"] < r["original_physics_mse"] for r in rows)
    )
    summary["guarded_selected_g3"] = int(sum(r["guard_selected_g3"] for r in rows))
    summary["guarded_selected_original"] = len(rows) - summary["guarded_selected_g3"]
    return summary


def paired_bundle(rows: Sequence[Dict], method: str, seed: int, n_boot: int) -> Dict:
    output = {"physics": {}, "image": {}}
    for offset, metric in enumerate(("mse", "mae", "rrmse")):
        if metric in ("mse", "mae", "rrmse"):
            delta = np.asarray(
                [
                    r[f"original_physics_{metric}"] - r[f"{method}_physics_{metric}"]
                    for r in rows
                ],
                dtype=np.float64,
            )
        output["physics"][metric] = safe_paired(delta, seed + offset, n_boot)
    for offset, metric in enumerate(("mse", "mae", "psnr", "ssim"), start=10):
        if metric in ("mse", "mae"):
            delta = np.asarray(
                [
                    r[f"original_image_{metric}"] - r[f"{method}_image_{metric}"]
                    for r in rows
                ],
                dtype=np.float64,
            )
        else:
            delta = np.asarray(
                [
                    r[f"{method}_image_{metric}"] - r[f"original_image_{metric}"]
                    for r in rows
                ],
                dtype=np.float64,
            )
        output["image"][metric] = safe_paired(delta, seed + offset, n_boot)
    return output


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--prediction_root", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--base_start", type=int, default=1)
    ap.add_argument("--base_end", type=int, default=20)
    ap.add_argument("--frequency", type=float, default=500000.0)
    ap.add_argument("--forward_iters", type=int, default=80)
    ap.add_argument("--boundary_width", type=int, default=300)
    ap.add_argument("--boundary_strength", type=float, default=225.0)
    ap.add_argument("--boundary_type", default="PML3")
    ap.add_argument("--speed_min", type=float, default=1400.0)
    ap.add_argument("--speed_max", type=float, default=1605.0)
    ap.add_argument("--alignment_rrmse_tol", type=float, default=1e-3)
    ap.add_argument("--min_image_mse_gain", type=float, default=0.05)
    ap.add_argument("--image_mae_tolerance", type=float, default=0.0)
    ap.add_argument("--image_ssim_guard", type=float, default=0.0)
    ap.add_argument("--min_image_mse_wins", type=int, default=10)
    ap.add_argument("--min_raw_physics_wins", type=int, default=10)
    ap.add_argument("--bootstrap_samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()

    if args.split != "test" or args.base_start != 1 or args.base_end != 20:
        raise RuntimeError("Final G3 physics validation is locked to VAL20=test1..20")
    out = Path(args.output_dir)
    cache_dir = out / "forward_cache"
    guarded_dir = out / "guarded_predictions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    guarded_dir.mkdir(parents=True, exist_ok=True)

    protected = {
        "prediction_root": str(Path(args.prediction_root).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "split": args.split,
        "base_start": args.base_start,
        "base_end": args.base_end,
        "frequency": args.frequency,
        "forward_iters": args.forward_iters,
        "boundary_width": args.boundary_width,
        "boundary_strength": args.boundary_strength,
        "boundary_type": args.boundary_type,
        "speed_min": args.speed_min,
        "speed_max": args.speed_max,
    }
    config_path = out / "config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        changed = [key for key, value in protected.items() if old.get(key) != value]
        if changed:
            raise RuntimeError(f"resume guard: protected settings changed: {changed}")
    else:
        json_dump(config_path, protected)

    groups = load_groups(args.data_root, args.split)
    bases = list(groups.items())
    bases = bases[args.base_start - 1 : args.base_end]
    expected = [f"test_{index}" for index in range(1, 21)]
    actual = [base for base, _ in bases]
    if actual != expected:
        raise RuntimeError(f"expected bases {expected}, got {actual}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_range = args.speed_max - args.speed_min
    rows: List[Dict] = []

    print("=" * 120)
    print("Frozen G3-Ensemble true-CBS validation")
    print("bases             =", actual[0], "..", actual[-1])
    print("CBS               =", args.frequency, args.forward_iters, args.boundary_type)
    print("candidate policy  = Original vs G3 by true-CBS MSE; GT is evaluation-only")
    print("=" * 120)

    for position, (base, records) in enumerate(bases, start=1):
        init_path = find_alpha(records, 0.0)
        gt_path = find_alpha(records, 1.0)
        prediction_path = Path(args.prediction_root) / f"{base}.npz"
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)

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
                args.speed_min + args.speed_max
            )
            target_image = to_2d(pred_data["target_speed"])
            raw_g3_image = to_2d(pred_data["hybrid_speed"])
            raw_g3_480 = resize_np(raw_g3_image.T, 480)
            raw_g3_480 = np.clip(
                raw_g3_480, args.speed_min, args.speed_max
            ).astype(np.float32)

        cache_path = cache_dir / f"{base}.npz"
        if cache_path.exists() and not args.no_resume:
            with np.load(cache_path) as cache:
                original_dobs = cache["original_dobs"].astype(np.complex64)
                raw_g3_dobs = cache["raw_g3_dobs"].astype(np.complex64)
        else:
            original_dobs = forward_cbs(
                original_480, src_indices, rec_indices, args, device
            )
            raw_g3_dobs = forward_cbs(
                raw_g3_480, src_indices, rec_indices, args, device
            )
            np.savez_compressed(
                cache_path,
                original_dobs=original_dobs,
                raw_g3_dobs=raw_g3_dobs,
                target_dobs=target_dobs,
            )

        alignment_rrmse = complex_rrmse(original_dobs, stored_original_dobs)
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
        row: Dict = {
            "sample_id": numeric_key(base),
            "base_sample": base,
            "alignment_rrmse": alignment_rrmse,
            "guard_selected_g3": bool(choose_g3),
            "physics_mse_improvement_raw": original_physics["mse"] - raw_physics["mse"],
            "physics_mse_improvement_guarded": original_physics["mse"]
            - guarded_physics["mse"],
        }
        for method, metrics in (
            ("original_physics", original_physics),
            ("raw_g3_physics", raw_physics),
            ("guarded_physics", guarded_physics),
            ("original_image", original_image_metrics),
            ("raw_g3_image", raw_image_metrics),
            ("guarded_image", guarded_image_metrics),
        ):
            row.update({f"{method}_{key}": value for key, value in metrics.items()})
        rows.append(row)

        np.savez_compressed(
            guarded_dir / f"{base}.npz",
            sample_id=np.int64(numeric_key(base)),
            selected_g3=np.bool_(choose_g3),
            selected_speed_image=guarded_image.astype(np.float32),
            original_speed_image=condition_image.astype(np.float32),
            raw_g3_speed_image=raw_g3_image.astype(np.float32),
            target_speed_image=target_image.astype(np.float32),
            original_physics_mse=np.float64(original_physics["mse"]),
            raw_g3_physics_mse=np.float64(raw_physics["mse"]),
        )
        print(
            f"[{position:02d}/20] {base} | align={alignment_rrmse:.3e} | "
            f"CBS {original_physics['mse']:.6e}->{raw_physics['mse']:.6e} | "
            f"guard={'G3' if choose_g3 else 'Original'}"
        )

    summary = summarize(rows)
    alignment_max = max(r["alignment_rrmse"] for r in rows)
    alignment_pass = alignment_max <= args.alignment_rrmse_tol
    raw_physics_improvement = (
        summary["original_physics_mse"]["mean"]
        - summary["raw_g3_physics_mse"]["mean"]
    )
    raw_physics_pass = bool(
        raw_physics_improvement >= 0
        and summary["raw_g3_physics_mse_wins"] >= args.min_raw_physics_wins
    )

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
        mse_wins = int(
            sum(r[f"{method}_image_mse"] < r["original_image_mse"] for r in rows)
        )
        return {
            "mse_gain": mse_gain,
            "mae_gain": mae_gain,
            "ssim_gain": ssim_gain,
            "mse_wins": mse_wins,
            "pass": bool(
                mse_gain >= args.min_image_mse_gain
                and mse_wins >= args.min_image_mse_wins
                and mae_gain >= -args.image_mae_tolerance
                and ssim_gain >= -args.image_ssim_guard
            ),
        }

    raw_image_guard = image_guard("raw_g3")
    guarded_image_guard = image_guard("guarded")
    if not alignment_pass:
        status = "ALIGNMENT_FAIL"
        reason = "recomputed Original dobs does not match the stored alpha=0 CBS data"
    elif guarded_image_guard["pass"] and summary["guarded_selected_g3"] > 0:
        status = "CBS_GUARDED_PASS"
        reason = "CBS-guarded G3 preserves image guards and guarantees nonworsening physics"
    elif raw_physics_pass and raw_image_guard["pass"]:
        status = "RAW_G3_PASS"
        reason = "raw G3 passes both image and true-CBS physics guards"
    else:
        status = "REJECT"
        reason = "neither raw nor CBS-guarded G3 satisfies the frozen combined criteria"

    paired = {
        "raw_g3_vs_original": paired_bundle(
            rows, "raw_g3", args.seed + 500000, args.bootstrap_samples
        ),
        "guarded_vs_original": paired_bundle(
            rows, "guarded", args.seed + 600000, args.bootstrap_samples
        ),
    }
    write_csv(out / "physics_per_sample.csv", rows)
    json_dump(out / "paired_statistics.json", paired)
    json_dump(
        out / "physics_validation.json",
        {
            "status": status,
            "reason": reason,
            "alignment_pass": alignment_pass,
            "alignment_rrmse_max": alignment_max,
            "raw_physics_pass": raw_physics_pass,
            "raw_physics_mse_improvement_mean": raw_physics_improvement,
            "raw_image_guard": raw_image_guard,
            "guarded_image_guard": guarded_image_guard,
            "summary": summary,
            "config": protected,
        },
    )

    print("=" * 120)
    print("status                    =", status)
    print("alignment max RRMSE       =", f"{alignment_max:.3e}")
    print(
        "raw G3 physics MSE        =",
        f"{summary['original_physics_mse']['mean']:.6e}",
        "->",
        f"{summary['raw_g3_physics_mse']['mean']:.6e}",
        f"wins={summary['raw_g3_physics_mse_wins']}/20",
    )
    print(
        "CBS Guard selected G3    =",
        f"{summary['guarded_selected_g3']}/20",
    )
    print("raw image guard           =", raw_image_guard)
    print("guarded image guard       =", guarded_image_guard)
    print("saved to:", out.resolve())


if __name__ == "__main__":
    main()
