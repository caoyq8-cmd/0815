#!/usr/bin/env python3
"""Build a harmonized 240x240 DEV30 comparison table.

This script recomputes the G3 image metrics on the same 240x240 grid used by
the existing Original / Local-AA / Full-CBS / E3.5 CSV files.  It never tunes
or selects a G3 hyperparameter.  The already-frozen true-CBS decisions in
``dev30_per_sample.csv`` are reused verbatim.

Outputs
-------
* dev30_all_methods_240_per_sample.csv
* dev30_all_methods_240_summary.csv
* dev30_all_methods_240_statistics.json
* dev30_all_methods_240_tradeoff.png
* g3_guarded_per_sample_gains_240.png
* harmonization_audit.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from skimage.metrics import structural_similarity


IDS = tuple(range(21, 51))
SPEED_MIN = 1400.0
SPEED_MAX = 1605.0
DATA_RANGE = SPEED_MAX - SPEED_MIN

ORIGINAL_ALIASES = (
    "condition_speed", "condition", "condition_256", "cond", "x_init", "init", "original",
    "reconstruction", "speed_pred", "pred",
)
TARGET_ALIASES = (
    "target_speed", "target", "target_256", "gt", "ground_truth", "speed_gt", "truth",
)
G3_ALIASES = (
    "hybrid_speed", "ensemble_prediction_speed", "ensemble_speed",
    "raw_g3_speed_image", "raw_g3_speed", "g3_speed",
    "prediction_speed", "ensemble_prediction", "ensemble", "raw_g3", "g3", "prediction",
    "pred", "speed", "sample", "x_pred",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--condition_root", type=Path,
                   default=Path("condition_cache/inversionnet_oof5_b32_blocks2_e67"))
    p.add_argument("--local_alpha_root", type=Path,
                   default=Path("dev30_test21_50/local_alpha01"),
                   help="Frozen alpha=0/1 records containing the authoritative 480-grid images")
    p.add_argument("--g3_output_root", type=Path,
                   default=Path("dev30_test21_50/results/g3_ensemble_cbs_guard_frozen"))
    p.add_argument("--g3_per_sample_csv", type=Path, default=None)
    p.add_argument("--original_csv", type=Path,
                   default=Path("dev30_test21_50/results/original/sample_results.csv"))
    p.add_argument("--local_aa_csv", type=Path,
                   default=Path("dev30_test21_50/results/local_aa/sample_results.csv"))
    p.add_argument("--full_cbs_csv", type=Path,
                   default=Path("dev30_test21_50/results/full_cbs/sample_results.csv"))
    p.add_argument("--e35_csv", type=Path,
                   default=Path("dev30_test21_50/results/structure_preserving_e35/rho0p5_gamma0p5/sample_results.csv"))
    p.add_argument("--output_dir", type=Path,
                   default=Path("dev30_test21_50/results/all_methods_harmonized_240"))
    p.add_argument("--interpolation", choices=("bilinear", "bicubic"), default="bilinear")
    p.add_argument("--align_corners", action="store_true")
    p.add_argument("--original_alignment_mse_tol", type=float, default=1e-4,
                   help="Maximum absolute difference from frozen Original MSE240")
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20260902)
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def sample_id_from_text(value: object) -> int:
    m = re.search(r"(?:test[_-]?)?(\d+)", str(value))
    if not m:
        raise ValueError(f"Cannot parse sample id from {value!r}")
    return int(m.group(1))


def rows_by_id(rows: Sequence[Mapping[str, str]]) -> Dict[int, Mapping[str, str]]:
    out: Dict[int, Mapping[str, str]] = {}
    for row in rows:
        source = row.get("sample_id") or row.get("base_sample") or row.get("base")
        sid = sample_id_from_text(source)
        if sid in out:
            raise RuntimeError(f"Duplicate sample id {sid}")
        out[sid] = row
    return out


def scalar(row: Mapping[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        return default
    return float(value)


def squeeze_image(x: np.ndarray, *, source: Path, key: str) -> np.ndarray:
    x = np.asarray(x)
    x = np.squeeze(x)
    if x.ndim != 2:
        raise RuntimeError(f"{source}:{key} must reduce to 2-D, got {x.shape}")
    if not np.isfinite(x).all():
        raise RuntimeError(f"Non-finite values in {source}:{key}")
    return x.astype(np.float32, copy=False)


def pick_array(z: np.lib.npyio.NpzFile, aliases: Sequence[str], source: Path,
               role: str) -> Tuple[np.ndarray, str]:
    keys = list(z.files)
    lower = {k.lower(): k for k in keys}
    for alias in aliases:
        if alias.lower() in lower:
            key = lower[alias.lower()]
            return squeeze_image(z[key], source=source, key=key), key
    candidates = []
    for key in keys:
        arr = np.asarray(z[key])
        if arr.size >= 128 * 128 and np.squeeze(arr).ndim == 2:
            candidates.append(key)
    if len(candidates) == 1:
        key = candidates[0]
        return squeeze_image(z[key], source=source, key=key), key
    raise RuntimeError(
        f"Cannot identify {role} in {source}. keys={keys}, 2-D candidates={candidates}. "
        f"Add its exact key to the {role} alias list at the top of this script."
    )


def resize_image(x: np.ndarray, size: int, mode: str, align_corners: bool) -> np.ndarray:
    t = torch.from_numpy(x).float()[None, None]
    kwargs = {"size": (size, size), "mode": mode}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = align_corners
    y = F.interpolate(t, **kwargs)[0, 0]
    return y.cpu().numpy().astype(np.float32, copy=False)


def load_local_alpha_map(root: Path) -> Dict[int, Dict[int, Path]]:
    split = root / "test" if (root / "test").is_dir() else root
    files = sorted(split.glob("test_*.npz"))
    groups: Dict[int, Dict[int, Path]] = {}
    for path in files:
        with np.load(path, allow_pickle=True) as z:
            if not {"base_sample", "alpha", "target_480"}.issubset(z.files):
                continue
            base = np.asarray(z["base_sample"]).item()
            alpha = float(np.asarray(z["alpha"]).item())
        sid = sample_id_from_text(base)
        if sid not in IDS:
            continue
        if abs(alpha) < 1e-7:
            ai = 0
        elif abs(alpha - 1.0) < 1e-7:
            ai = 1
        else:
            continue
        if ai in groups.setdefault(sid, {}):
            raise RuntimeError(f"Duplicate alpha={ai} record for test_{sid}")
        groups[sid][ai] = path
    missing = [sid for sid in IDS if sid not in groups or set(groups[sid]) != {0, 1}]
    if missing:
        raise RuntimeError(
            f"Missing frozen alpha=0/1 local-alpha records for IDs {missing} under {split}"
        )
    return groups


def image_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    # Match the frozen Local-AA / Full-CBS metric implementation exactly.
    # Those scripts operate in float32 before reduction; using float64 here
    # creates tiny non-zero differences for CBS-guard fallback samples and
    # incorrectly turns exact ties into wins/losses in paired tests.
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    err = pred - target
    mse = float(np.mean(err * err))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    psnr = float(10.0 * np.log10((DATA_RANGE ** 2) / max(mse, 1e-30)))
    ssim = float(structural_similarity(
        target, pred, data_range=DATA_RANGE
    ))
    return {"mse": mse, "mae": mae, "rmse": rmse, "psnr": psnr, "ssim": ssim}


def condition_file(root: Path, sid: int) -> Path:
    candidates = [
        root / "test" / f"test_{sid}.npz",
        root / f"test_{sid}.npz",
    ]
    for p in candidates:
        if p.is_file():
            return p
    found = list(root.rglob(f"test_{sid}.npz"))
    if len(found) == 1:
        return found[0]
    raise FileNotFoundError(f"Expected one condition file for test_{sid}; found {found}")


def prediction_files(root: Path, sid: int, condition_path: Path) -> List[Path]:
    patterns = (f"test_{sid}.npz", f"*test_{sid}*.npz", f"*_{sid}.npz")
    found: List[Path] = []
    for pat in patterns:
        found.extend(root.rglob(pat))
    unique = []
    for p in found:
        if p.resolve() == condition_path.resolve() or p in unique:
            continue
        unique.append(p)
    def score(p: Path) -> Tuple[int, int, str]:
        s = str(p).lower()
        positive = sum(token in s for token in ("ensemble", "prediction", "g3"))
        negative = sum(token in s for token in ("physics", "cbs_cache", "dobs"))
        return (-positive + negative, len(p.parts), str(p))
    return sorted(unique, key=score)


def load_g3_prediction(root: Path, sid: int, condition_path: Path) -> Tuple[np.ndarray, Path, str]:
    errors = []
    for path in prediction_files(root, sid, condition_path):
        try:
            with np.load(path, allow_pickle=False) as z:
                arr, key = pick_array(z, G3_ALIASES, path, "G3")
            return arr, path, key
        except Exception as exc:  # retain diagnostics for safe failure
            errors.append(f"{path}: {exc}")
    message = "\n".join(errors[:10]) if errors else "No matching .npz files"
    raise RuntimeError(f"Cannot load G3 prediction for test_{sid} under {root}:\n{message}")


def bootstrap_ci(values: np.ndarray, n: int, rng: np.random.Generator) -> List[float]:
    if n <= 0:
        return [math.nan, math.nan]
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [float(lo), float(hi)]


def paired_stats(gains: np.ndarray, n_boot: int, rng: np.random.Generator) -> Dict[str, object]:
    gains = np.asarray(gains, dtype=np.float64)
    nz = gains[np.abs(gains) > 1e-12]
    if len(nz) == 0:
        p_two = p_greater = 1.0
    else:
        p_two = float(wilcoxon(nz, alternative="two-sided").pvalue)
        p_greater = float(wilcoxon(nz, alternative="greater").pvalue)
    return {
        "mean_gain": float(gains.mean()),
        "median_gain": float(np.median(gains)),
        "bootstrap_mean_95ci": bootstrap_ci(gains, n_boot, rng),
        "wilcoxon_two_sided_p": p_two,
        "wilcoxon_improvement_p": p_greater,
        "wins": int(np.sum(gains > 1e-12)),
        "losses": int(np.sum(gains < -1e-12)),
        "ties": int(np.sum(np.abs(gains) <= 1e-12)),
    }


def method_rows_from_frozen_csv(method: str, path: Path, use_init: bool = False) -> List[Dict[str, object]]:
    rows = rows_by_id(read_csv(path))
    prefix = "init" if use_init else "final"
    result = []
    for sid in IDS:
        if sid not in rows:
            raise RuntimeError(f"{path} lacks test_{sid}")
        r = rows[sid]
        result.append({
            "sample_id": sid,
            "base_sample": f"test_{sid}",
            "method": method,
            "image_mse_240": scalar(r, f"{prefix}_mse_240"),
            "image_mae_240": scalar(r, f"{prefix}_mae_240"),
            "image_psnr_240": scalar(r, f"{prefix}_psnr_240"),
            "image_ssim_240": scalar(r, f"{prefix}_ssim_240"),
            "physics_mse": scalar(r, f"{prefix}_true_cbs_mse"),
            "physics_mse_baseline": scalar(r, "init_true_cbs_mse"),
            "physics_rrmse": scalar(r, f"{prefix}_true_cbs_rrmse"),
            "cbs_forward_calls": 0.0 if use_init else scalar(r, "cbs_forward_calls_algorithm", 0.0),
            "cbs_adjoint_calls": 0.0 if use_init else scalar(r, "cbs_adjoint_calls", 0.0),
            "neural_calls": 0.0 if use_init else scalar(r, "neural_grad_calls", 0.0),
            "runtime_sec": 0.0 if use_init else scalar(r, "runtime_sec"),
            "runtime_valid": True,
            "evaluation_grid": "240x240",
        })
    return result


def recompute_g3(args: argparse.Namespace,
                 original_reference: Mapping[int, Mapping[str, str]],
                 local_alpha: Mapping[int, Mapping[int, Path]]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    g3_csv = args.g3_per_sample_csv or args.g3_output_root / "dev30_per_sample.csv"
    frozen = rows_by_id(read_csv(g3_csv))
    rows: List[Dict[str, object]] = []
    audit_files = []
    max_original_mse_error = 0.0
    max_saved_guard_error = 0.0
    max_saved_raw_g3_error = 0.0
    max_condition_480_error = 0.0
    for sid in IDS:
        cond_path = condition_file(args.condition_root, sid)
        with np.load(cond_path, allow_pickle=False) as z:
            condition_image, condition_key = pick_array(
                z, ORIGINAL_ALIASES, cond_path, "original"
            )

        alpha0_path = local_alpha[sid][0]
        alpha1_path = local_alpha[sid][1]
        with np.load(alpha0_path, allow_pickle=True) as z0:
            original_480 = squeeze_image(
                z0["target_480"], source=alpha0_path, key="target_480"
            )
        with np.load(alpha1_path, allow_pickle=True) as z1:
            target_480 = squeeze_image(
                z1["target_480"], source=alpha1_path, key="target_480"
            )

        # Reproduce generate_local_alpha_cbs_dataset.py exactly.  The condition
        # cache is image-coordinate, whereas CBS/local-alpha arrays are physics-
        # coordinate.  The stored alpha=0 target_480 is authoritative.
        condition_480_rebuilt = resize_image(
            condition_image.T, 480, args.interpolation, args.align_corners
        )
        condition_480_rebuilt = np.clip(
            condition_480_rebuilt, SPEED_MIN, SPEED_MAX
        ).astype(np.float32)
        condition_480_error = float(np.max(np.abs(
            condition_480_rebuilt - original_480
        )))
        max_condition_480_error = max(max_condition_480_error, condition_480_error)
        if condition_480_error > 1e-4:
            raise RuntimeError(
                f"test_{sid} condition→stored-480 alignment failed: "
                f"max_abs={condition_480_error:.3e}"
            )

        raw_g3, pred_path, pred_key = load_g3_prediction(args.g3_output_root, sid, cond_path)

        original_240 = resize_image(
            original_480, 240, args.interpolation, args.align_corners
        )
        target_240 = resize_image(
            target_480, 240, args.interpolation, args.align_corners
        )

        # G3 predictions are saved in image coordinates.  Convert them through
        # the same physics-grid parameterization before the 240-grid metrics.
        raw_g3_480 = resize_image(
            raw_g3.T, 480, args.interpolation, args.align_corners
        )
        raw_g3_480 = np.clip(raw_g3_480, SPEED_MIN, SPEED_MAX).astype(np.float32)
        raw_g3_240 = resize_image(
            raw_g3_480, 240, args.interpolation, args.align_corners
        )
        selected = str(frozen[sid].get("guard_selected_g3", "")).lower() in ("1", "true", "yes")
        guarded_240 = raw_g3_240 if selected else original_240

        # Independent consistency check against the explicitly saved guarded
        # prediction.  This verifies both the frozen CSV decision and the raw
        # hybrid array chosen from ensemble_predictions.
        guarded_path = args.g3_output_root / "guarded_predictions" / f"test_{sid}.npz"
        saved_guard_error = math.nan
        saved_raw_error = math.nan
        if guarded_path.is_file():
            with np.load(guarded_path, allow_pickle=False) as gz:
                saved_selected = bool(np.asarray(gz["selected_g3"]).item())
                saved_guard = squeeze_image(
                    gz["selected_speed_image"], source=guarded_path,
                    key="selected_speed_image",
                )
                saved_raw = squeeze_image(
                    gz["raw_g3_speed_image"], source=guarded_path,
                    key="raw_g3_speed_image",
                )
            if saved_selected != selected:
                raise RuntimeError(
                    f"test_{sid} guard decision mismatch: CSV={selected}, "
                    f"guarded_predictions={saved_selected}"
                )
            saved_guard_error = float(np.max(np.abs(
                resize_image(
                    np.clip(
                        resize_image(
                            saved_guard.T, 480, args.interpolation,
                            args.align_corners,
                        ),
                        SPEED_MIN, SPEED_MAX,
                    ).astype(np.float32),
                    240, args.interpolation, args.align_corners,
                )
                - guarded_240
            )))
            saved_raw_error = float(np.max(np.abs(saved_raw - raw_g3)))
            max_saved_guard_error = max(max_saved_guard_error, saved_guard_error)
            max_saved_raw_g3_error = max(max_saved_raw_g3_error, saved_raw_error)
            if saved_guard_error > 1e-5 or saved_raw_error > 1e-5:
                raise RuntimeError(
                    f"test_{sid} saved G3 consistency failed: "
                    f"guard_max_abs={saved_guard_error:.3e}, "
                    f"raw_max_abs={saved_raw_error:.3e}"
                )
        m = image_metrics(guarded_240, target_240)
        orig_m = image_metrics(original_240, target_240)

        reference_mse = scalar(original_reference[sid], "init_mse_240")
        alignment_error = abs(orig_m["mse"] - reference_mse)
        max_original_mse_error = max(max_original_mse_error, alignment_error)
        if alignment_error > args.original_alignment_mse_tol:
            raise RuntimeError(
                f"test_{sid} Original MSE240 alignment failed: recomputed={orig_m['mse']:.9f}, "
                f"frozen={reference_mse:.9f}, abs_error={alignment_error:.3e}. "
                "Do not merge tables until interpolation/array keys are corrected."
            )

        fr = frozen[sid]
        rows.append({
            "sample_id": sid,
            "base_sample": f"test_{sid}",
            "method": "G3-Ensemble-CBSGuard",
            "image_mse_240": m["mse"],
            "image_mae_240": m["mae"],
            "image_psnr_240": m["psnr"],
            "image_ssim_240": m["ssim"],
            "physics_mse": scalar(fr, "guarded_physics_mse"),
            "physics_mse_baseline": scalar(fr, "original_physics_mse"),
            "physics_rrmse": scalar(fr, "guarded_physics_rrmse"),
            "cbs_forward_calls": scalar(fr, "cbs_forward_calls", 2.0),
            "cbs_adjoint_calls": scalar(fr, "cbs_adjoint_calls", 0.0),
            "neural_calls": scalar(fr, "neural_flow_nfe", 8.0) + scalar(fr, "neural_ddpm_nfe", 25.0),
            "runtime_sec": math.nan,
            "runtime_valid": False,
            "evaluation_grid": "240x240",
            "guard_selected_g3": selected,
        })
        audit_files.append({
            "sample_id": sid,
            "condition_file": str(cond_path.resolve()),
            "condition_original_key": condition_key,
            "alpha0_file": str(alpha0_path.resolve()),
            "alpha1_file": str(alpha1_path.resolve()),
            "authoritative_original_key": "target_480",
            "authoritative_target_key": "target_480",
            "condition_to_stored_480_max_abs_error": condition_480_error,
            "g3_prediction_file": str(pred_path.resolve()),
            "g3_prediction_key": pred_key,
            "saved_guarded_file": str(guarded_path.resolve()) if guarded_path.is_file() else None,
            "saved_guard_max_abs_error": saved_guard_error,
            "saved_raw_g3_max_abs_error": saved_raw_error,
            "original_mse_240_recomputed": orig_m["mse"],
            "original_mse_240_frozen": reference_mse,
            "absolute_alignment_error": alignment_error,
        })
    return rows, {
        "status": "PASS",
        "interpolation": args.interpolation,
        "align_corners": args.align_corners,
        "original_alignment_mse_tolerance": args.original_alignment_mse_tol,
        "max_original_mse_alignment_error": max_original_mse_error,
        "max_saved_guard_error": max_saved_guard_error,
        "max_saved_raw_g3_error": max_saved_raw_g3_error,
        "max_condition_to_stored_480_error": max_condition_480_error,
        "coordinate_note": "G3 image coordinates are transposed to physics coordinates, resized 256→480, clipped, then resized 480→240.",
        "runtime_note": "G3 runtime intentionally omitted: the provided rerun reused CBS cache.",
        "files": audit_files,
    }


def summarize(all_rows: Sequence[Mapping[str, object]], n_boot: int,
              seed: int) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    methods = ["Original", "Local-AA", "Full-CBS", "E3.5", "G3-Ensemble-CBSGuard"]
    by_method = {m: [r for r in all_rows if r["method"] == m] for m in methods}
    orig = {int(r["sample_id"]): r for r in by_method["Original"]}
    summary_rows = []
    stats: Dict[str, object] = {}
    rng = np.random.default_rng(seed)
    for method in methods:
        rows = by_method[method]
        if len(rows) != len(IDS):
            raise RuntimeError(f"{method} has {len(rows)} rows; expected {len(IDS)}")
        item: Dict[str, object] = {"method": method, "n": len(rows), "evaluation_grid": "240x240"}
        for key in ("image_mse_240", "image_mae_240", "image_psnr_240", "image_ssim_240",
                    "physics_mse", "physics_rrmse", "cbs_forward_calls", "cbs_adjoint_calls", "neural_calls"):
            vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            item[f"{key}_mean"] = float(finite.mean()) if len(finite) else math.nan
            item[f"{key}_sample_std"] = float(finite.std(ddof=1)) if len(finite) > 1 else math.nan
        valid_runtime = [float(r["runtime_sec"]) for r in rows if bool(r["runtime_valid"]) and np.isfinite(float(r["runtime_sec"]))]
        item["runtime_sec_mean"] = float(np.mean(valid_runtime)) if valid_runtime else math.nan
        item["runtime_valid"] = bool(valid_runtime)
        if method == "Original":
            item.update({"image_mse_gain_mean": 0.0, "image_mse_gain_percent": 0.0,
                         "physics_mse_gain_mean": 0.0, "physics_mse_gain_percent": 0.0,
                         "image_mse_wins": 0, "physics_mse_wins": 0})
        else:
            mse_gain = np.asarray([float(orig[int(r["sample_id"])]["image_mse_240"]) - float(r["image_mse_240"]) for r in rows])
            # Physics baselines from the legacy 240 pipeline and the newer G3
            # pipeline differ slightly.  Compare every method with the exact
            # baseline evaluated in its own frozen physics run, never with a
            # mismatched global baseline.
            phy_gain = np.asarray([
                float(r["physics_mse_baseline"]) - float(r["physics_mse"])
                for r in rows
            ])
            phy_rel = np.asarray([
                (float(r["physics_mse_baseline"]) - float(r["physics_mse"]))
                / max(float(r["physics_mse_baseline"]), 1e-30)
                for r in rows
            ])
            item.update({
                "image_mse_gain_mean": float(mse_gain.mean()),
                "image_mse_gain_percent": float(100 * mse_gain.mean() / np.mean([float(x["image_mse_240"]) for x in orig.values()])),
                "physics_mse_gain_mean": float(phy_gain.mean()),
                "physics_mse_gain_percent": float(100 * phy_rel.mean()),
                "image_mse_wins": int(np.sum(mse_gain > 1e-12)),
                "physics_mse_wins": int(np.sum(phy_gain > 1e-12)),
            })
            metric_stats = {}
            for metric, sign in (("image_mse_240", 1), ("image_mae_240", 1),
                                 ("image_psnr_240", -1), ("image_ssim_240", -1)):
                gains = np.asarray([sign * (
                    float(orig[int(r["sample_id"])][metric]) - float(r[metric])
                ) for r in rows])
                if np.isfinite(gains).all():
                    metric_stats[metric] = paired_stats(gains, n_boot, rng)
            if np.isfinite(phy_gain).all():
                metric_stats["physics_mse"] = paired_stats(phy_gain, n_boot, rng)
            stats[f"{method}_vs_Original"] = metric_stats
        summary_rows.append(item)
    return summary_rows, stats


def make_figures(summary_rows: Sequence[Mapping[str, object]], all_rows: Sequence[Mapping[str, object]], out: Path) -> None:
    import matplotlib.pyplot as plt

    methods = [r for r in summary_rows if r["method"] != "Original"]
    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=160)
    for r in methods:
        calls = float(r["cbs_forward_calls_mean"]) + float(r["cbs_adjoint_calls_mean"])
        size = 80 + 8 * min(calls, 60)
        ax.scatter(float(r["physics_mse_gain_percent"]), float(r["image_mse_gain_percent"]), s=size, alpha=.78)
        ax.annotate(str(r["method"]), (float(r["physics_mse_gain_percent"]), float(r["image_mse_gain_percent"])),
                    xytext=(6, 5), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="0.7", lw=1)
    ax.axvline(0, color="0.7", lw=1)
    ax.set_xlabel("Physics MSE improvement vs Original (%)")
    ax.set_ylabel("Image MSE improvement vs Original (%)")
    ax.set_title("DEV30 accuracy-physics-cost trade-off (240x240)")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "dev30_all_methods_240_tradeoff.png", bbox_inches="tight")
    plt.close(fig)

    orig = {int(r["sample_id"]): r for r in all_rows if r["method"] == "Original"}
    g3 = sorted((r for r in all_rows if r["method"] == "G3-Ensemble-CBSGuard"), key=lambda x: int(x["sample_id"]))
    gains = [float(orig[int(r["sample_id"])]["image_mse_240"]) - float(r["image_mse_240"]) for r in g3]
    colors = ["#2b8cbe" if bool(r.get("guard_selected_g3")) else "#bdbdbd" for r in g3]
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=160)
    ax.bar([int(r["sample_id"]) for r in g3], gains, color=colors)
    ax.axhline(0, color="black", lw=.8)
    ax.set_xlabel("DEV30 sample ID")
    ax.set_ylabel("Original MSE240 - G3-Guard MSE240")
    ax.set_title("Frozen G3-CBSGuard per-sample image gains")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "g3_guarded_per_sample_gains_240.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original_csv_rows = rows_by_id(read_csv(args.original_csv))
    local_alpha = load_local_alpha_map(args.local_alpha_root)
    original = method_rows_from_frozen_csv("Original", args.original_csv, use_init=True)
    local = method_rows_from_frozen_csv("Local-AA", args.local_aa_csv)
    full = method_rows_from_frozen_csv("Full-CBS", args.full_cbs_csv)
    e35 = method_rows_from_frozen_csv("E3.5", args.e35_csv)
    g3, audit = recompute_g3(args, original_csv_rows, local_alpha)
    all_rows = original + local + full + e35 + g3
    all_rows.sort(key=lambda r: (int(r["sample_id"]), str(r["method"])))

    summary_rows, statistics = summarize(all_rows, args.bootstrap, args.seed)
    write_csv(args.output_dir / "dev30_all_methods_240_per_sample.csv", all_rows)
    write_csv(args.output_dir / "dev30_all_methods_240_summary.csv", summary_rows)
    (args.output_dir / "dev30_all_methods_240_statistics.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "harmonization_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_figures(summary_rows, all_rows, args.output_dir)

    print("=" * 120)
    print("Harmonized DEV30 comparison: 240x240")
    print("=" * 120)
    print(f"{'method':28s} {'MSE240':>12s} {'MAE240':>10s} {'SSIM':>10s} {'phy gain %':>12s} {'CBS-F':>8s} {'CBS-A':>8s}")
    for r in summary_rows:
        print(f"{str(r['method']):28s} {float(r['image_mse_240_mean']):12.4f} "
              f"{float(r['image_mae_240_mean']):10.4f} {float(r['image_ssim_240_mean']):10.6f} "
              f"{float(r['physics_mse_gain_percent']):12.4f} {float(r['cbs_forward_calls_mean']):8.2f} "
              f"{float(r['cbs_adjoint_calls_mean']):8.2f}")
    print(f"alignment max abs MSE error = {audit['max_original_mse_alignment_error']:.3e}")
    print(f"saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
