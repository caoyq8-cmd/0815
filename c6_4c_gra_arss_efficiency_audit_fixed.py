#!/usr/bin/env python3
"""
C6.4C — Real Efficiency Audit for Exact-CBS / Full-ARSS / RA-ARSS / GRA-ARSS

This script is intentionally derived from C6.4B without changing the frozen
GRA-ARSS algorithm.

What C6.4C adds
---------------
1. CUDA-synchronized wall-clock timing.
2. Full-method warmup + repeated timing.
3. Peak GPU memory accounting.
4. Separation of method-core cost from evaluation-only true-CBS validation.
5. Stage-wise timing:
      - exact64 true CBS
      - 8-source anchor true CBS
      - neural 64-source MgNO
      - gradient + normalized update
      - GRA guarded line-search true CBS
6. Per-run, per-sample, and aggregate method summaries.
7. True-source-equivalent accounting.

Fair-cost convention
--------------------
Exact-CBS core:
    64 true-source CBS + exact gradient/update.
    Final 8-source objective evaluation is validation-only.

Full-ARSS core:
    64-source MgNO + neural residual + gradient/update.
    Current/final true-CBS objectives are validation-only.

RA-ARSS core:
    8 true-source CBS anchor/residual + 64-source MgNO + gradient/update.
    Final objective evaluation is validation-only.

GRA-ARSS core:
    8 true-source CBS anchor/residual
    + 64-source MgNO
    + gradient/update
    + every 8-source backtracking objective evaluation.
    These line-search evaluations ARE method cost, because the monotone guard
    needs them online.

The frozen GRA protocol remains:
    alpha_0 = 1.5 m/s
    alpha_{k+1} = 0.5 alpha_k
    accept iff J_trial < J_current
    max_backtracks = 5
No GT / MSE / SSIM / cosine is used for acceptance.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from c5_audit_mgno_unseen64_sources import (
    solve_cbs_chunked,
    load_mgno,
    neural_predict,
)

from c5_paper_ano_val5 import (
    ano_gradient,
)

from c5_directional_fd_audit import (
    measurement_loss,
    normalize_direction,
    scalar,
)


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def sha256_array(x):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    return hashlib.sha256(x.tobytes()).hexdigest()


def rrmse(a, b, eps=1e-12):
    a = np.asarray(a)
    b = np.asarray(b)
    num = np.sqrt(np.mean(np.abs(a - b) ** 2))
    den = np.sqrt(np.mean(np.abs(b) ** 2)) + eps
    return float(num / den)


def mse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def mae(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def objective_from_measurement(pred, dobs):
    residual = pred - dobs
    J = 0.5 * float(
        np.sum(
            np.abs(residual) ** 2,
            dtype=np.float64,
        )
    )
    rr = rrmse(pred, dobs)
    return J, rr, residual


def stat(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
        }

    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def pct_drop(J0, J1):
    return float(
        (J0 - J1)
        / max(abs(J0), 1e-30)
    )


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# CUDA-safe timing and memory
# ---------------------------------------------------------------------

def resolve_device(device_str):
    device = torch.device(device_str)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {device_str}, but CUDA is unavailable."
            )
        # Force context creation before benchmark.
        torch.empty(1, device=device)
        torch.cuda.synchronize(device)

    return device


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(fn, device):
    """
    Return (output, synchronized wall-clock seconds).
    """
    cuda_sync(device)
    t0 = time.perf_counter()
    out = fn()
    cuda_sync(device)
    return out, float(time.perf_counter() - t0)


def reset_peak_memory(device):
    if device.type != "cuda":
        return {
            "baseline_allocated_mb": 0.0,
            "baseline_reserved_mb": 0.0,
        }

    cuda_sync(device)
    torch.cuda.reset_peak_memory_stats(device)

    return {
        "baseline_allocated_mb":
            float(torch.cuda.memory_allocated(device) / 1024**2),

        "baseline_reserved_mb":
            float(torch.cuda.memory_reserved(device) / 1024**2),
    }


def read_peak_memory(device, baseline):
    if device.type != "cuda":
        return {
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
            "peak_delta_allocated_mb": 0.0,
            "peak_delta_reserved_mb": 0.0,
        }

    cuda_sync(device)

    peak_alloc = float(
        torch.cuda.max_memory_allocated(device) / 1024**2
    )
    peak_res = float(
        torch.cuda.max_memory_reserved(device) / 1024**2
    )

    return {
        "baseline_allocated_mb":
            float(
                baseline["baseline_allocated_mb"]
            ),

        "baseline_reserved_mb":
            float(
                baseline["baseline_reserved_mb"]
            ),

        "peak_allocated_mb":
            peak_alloc,

        "peak_reserved_mb":
            peak_res,

        "peak_delta_allocated_mb":
            max(
                0.0,
                peak_alloc
                - baseline["baseline_allocated_mb"],
            ),

        "peak_delta_reserved_mb":
            max(
                0.0,
                peak_res
                - baseline["baseline_reserved_mb"],
            ),
    }


# ---------------------------------------------------------------------
# One sample context
# ---------------------------------------------------------------------

def load_sample_context(
    bridge_root,
    sample_id,
    background_rec,
):
    p = bridge_root / f"test_{sample_id}.npz"

    if not p.exists():
        raise FileNotFoundError(p)

    z = np.load(
        p,
        allow_pickle=True,
    )

    candidate = z[
        "candidate_480"
    ].astype(np.float32)

    target = z[
        "target_480"
    ].astype(np.float32)

    dobs = z[
        "dobs_complex"
    ].astype(np.complex64)

    source_positions = z[
        "source_positions"
    ].astype(np.int64)

    src_indices = z[
        "src_indices"
    ].astype(np.int64)

    rec_indices = z[
        "rec_indices"
    ].astype(np.int64)

    frequency = float(
        scalar(z["frequency"])
    )

    cbs_iters = int(
        scalar(z["cbs_iters"])
    )

    boundary_width = int(
        scalar(z["boundary_width"])
    )

    boundary_strength = float(
        scalar(z["boundary_strength"])
    )

    boundary_type = str(
        scalar(z["boundary_type"])
    )

    if not np.array_equal(
        rec_indices,
        background_rec,
    ):
        raise RuntimeError(
            f"test_{sample_id}: "
            "receiver/background geometry mismatch"
        )

    expected_src = rec_indices[
        source_positions
    ]

    if not np.array_equal(
        src_indices,
        expected_src,
    ):
        raise RuntimeError(
            f"test_{sample_id}: "
            "src_indices != rec_indices[source_positions]"
        )

    if dobs.shape != (8, 64):
        raise RuntimeError(
            f"test_{sample_id}: "
            f"expected dobs=(8,64), got {dobs.shape}"
        )

    rr_idx = rec_indices[:, 0]
    cc_idx = rec_indices[:, 1]

    return {
        "sample": sample_id,
        "candidate": candidate,
        "target": target,
        "dobs": dobs,
        "source_positions": source_positions,
        "src_indices": src_indices,
        "rec_indices": rec_indices,
        "frequency": frequency,
        "cbs_iters": cbs_iters,
        "boundary_width": boundary_width,
        "boundary_strength": boundary_strength,
        "boundary_type": boundary_type,
        "rr_idx": rr_idx,
        "cc_idx": cc_idx,
        "candidate_sha256": sha256_array(candidate),
        "mse0": mse(candidate, target),
        "mae0": mae(candidate, target),
    }


# ---------------------------------------------------------------------
# Solver wrappers
# ---------------------------------------------------------------------

def true_forward(
    speed,
    source_indices,
    ctx,
    device_str,
    chunk_size,
):
    out = solve_cbs_chunked(
        speed=speed,
        source_indices=source_indices,
        frequency=ctx["frequency"],
        cbs_iters=ctx["cbs_iters"],
        boundary_width=ctx["boundary_width"],
        boundary_strength=ctx["boundary_strength"],
        boundary_type=ctx["boundary_type"],
        device=device_str,
        chunk_size=chunk_size,
    )

    return np.asarray(
        out,
        dtype=np.complex64,
    )


def neural_forward64(
    model,
    speed,
    background64,
    speed_mean,
    speed_std,
    wave_scale,
    device_str,
):
    out = neural_predict(
        model=model,
        speed=speed,
        backgrounds=background64,
        speed_mean=speed_mean,
        speed_std=speed_std,
        wave_scale=wave_scale,
        device=device_str,
    )

    return np.asarray(
        out,
        dtype=np.complex64,
    )


def true8_measurement_from_fields(
    true8,
    ctx,
):
    return true8[
        :,
        ctx["rr_idx"],
        ctx["cc_idx"],
    ]


def neural8_measurement_from_pred64(
    pred64,
    ctx,
):
    neural_tx = pred64[
        ctx["source_positions"]
    ]

    neural_measurement = neural_tx[
        :,
        ctx["rr_idx"],
        ctx["cc_idx"],
    ]

    return neural_tx, neural_measurement


def eval_trial_objective(
    trial,
    ctx,
    args,
    device,
):
    def _call():
        return measurement_loss(
            trial,
            ctx["dobs"],
            ctx["src_indices"],
            ctx["rec_indices"],
            ctx["frequency"],
            ctx["cbs_iters"],
            ctx["boundary_width"],
            ctx["boundary_strength"],
            ctx["boundary_type"],
            args.device,
        )

    (J1, rr1), seconds = timed_call(
        _call,
        device,
    )

    return (
        float(J1),
        float(rr1),
        float(seconds),
    )


# ---------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------

def run_exact_once(
    ctx,
    args,
    device,
):
    """
    Exact-CBS core:
        64 exact source solves -> exact gradient -> normalized update.

    Final objective check is NOT included here.
    """
    baseline = reset_peak_memory(device)
    candidate = ctx["candidate"]

    t_core0 = time.perf_counter()
    cuda_sync(device)

    true64, exact64_s = timed_call(
        lambda: true_forward(
            candidate,
            ctx["rec_indices"],
            ctx,
            args.device,
            args.chunk_size,
        ),
        device,
    )

    true8 = true64[
        ctx["source_positions"]
    ]

    true_measurement = (
        true8_measurement_from_fields(
            true8,
            ctx,
        )
    )

    J0, rr0, true_residual = (
        objective_from_measurement(
            true_measurement,
            ctx["dobs"],
        )
    )

    def _grad_update():
        g = ano_gradient(
            candidate=candidate,
            tx_waves=true8,
            basis64=true64,
            residual=true_residual,
            frequency=ctx["frequency"],
        )

        direction = normalize_direction(g)

        trial = (
            candidate
            - args.initial_step
            * direction
        ).astype(np.float32)

        return trial

    trial, grad_s = timed_call(
        _grad_update,
        device,
    )

    cuda_sync(device)
    core_s = float(
        time.perf_counter()
        - t_core0
    )

    mem = read_peak_memory(
        device,
        baseline,
    )

    return {
        "trial": trial,
        "J0": float(J0),
        "rr0": float(rr0),
        "accepted": True,
        "accepted_step_mps":
            float(args.initial_step),
        "num_trial_evals": 0,
        "true_source_equiv_core": 64,
        "exact64_true_cbs_seconds":
            float(exact64_s),
        "anchor_true8_seconds": 0.0,
        "guard_true8_seconds": 0.0,
        "neural64_seconds": 0.0,
        "gradient_update_seconds":
            float(grad_s),
        "method_core_seconds":
            float(core_s),
        **mem,
    }


def run_full_once(
    ctx,
    args,
    device,
    model,
    background64,
):
    """
    Full-ARSS core:
        MgNO64 -> neural residual -> neural adjoint gradient -> fixed update.

    No true CBS is method-core cost.
    """
    baseline = reset_peak_memory(device)
    candidate = ctx["candidate"]

    t_core0 = time.perf_counter()
    cuda_sync(device)

    pred64, neural_s = timed_call(
        lambda: neural_forward64(
            model=model,
            speed=candidate,
            background64=background64,
            speed_mean=args.speed_mean,
            speed_std=args.speed_std,
            wave_scale=args.wave_scale,
            device_str=args.device,
        ),
        device,
    )

    neural_tx, neural_measurement = (
        neural8_measurement_from_pred64(
            pred64,
            ctx,
        )
    )

    neural_residual = (
        neural_measurement
        - ctx["dobs"]
    )

    def _grad_update():
        g = ano_gradient(
            candidate=candidate,
            tx_waves=neural_tx,
            basis64=pred64,
            residual=neural_residual,
            frequency=ctx["frequency"],
        )

        direction = normalize_direction(g)

        trial = (
            candidate
            - args.initial_step
            * direction
        ).astype(np.float32)

        return trial

    trial, grad_s = timed_call(
        _grad_update,
        device,
    )

    cuda_sync(device)
    core_s = float(
        time.perf_counter()
        - t_core0
    )

    mem = read_peak_memory(
        device,
        baseline,
    )

    return {
        "trial": trial,
        "J0": None,
        "rr0": None,
        "accepted": True,
        "accepted_step_mps":
            float(args.initial_step),
        "num_trial_evals": 0,
        "true_source_equiv_core": 0,
        "exact64_true_cbs_seconds": 0.0,
        "anchor_true8_seconds": 0.0,
        "guard_true8_seconds": 0.0,
        "neural64_seconds":
            float(neural_s),
        "gradient_update_seconds":
            float(grad_s),
        "method_core_seconds":
            float(core_s),
        **mem,
    }


def run_ra_once(
    ctx,
    args,
    device,
    model,
    background64,
):
    """
    RA-ARSS core:
        exact 8-source anchor/residual
        + MgNO64 basis/wavefields
        + fixed 1.5 m/s normalized update.
    """
    baseline = reset_peak_memory(device)
    candidate = ctx["candidate"]

    t_core0 = time.perf_counter()
    cuda_sync(device)

    true8, anchor_s = timed_call(
        lambda: true_forward(
            candidate,
            ctx["src_indices"],
            ctx,
            args.device,
            args.chunk_size,
        ),
        device,
    )

    true_measurement = (
        true8_measurement_from_fields(
            true8,
            ctx,
        )
    )

    J0, rr0, true_residual = (
        objective_from_measurement(
            true_measurement,
            ctx["dobs"],
        )
    )

    pred64, neural_s = timed_call(
        lambda: neural_forward64(
            model=model,
            speed=candidate,
            background64=background64,
            speed_mean=args.speed_mean,
            speed_std=args.speed_std,
            wave_scale=args.wave_scale,
            device_str=args.device,
        ),
        device,
    )

    neural_tx, _ = (
        neural8_measurement_from_pred64(
            pred64,
            ctx,
        )
    )

    def _grad_update():
        g = ano_gradient(
            candidate=candidate,
            tx_waves=neural_tx,
            basis64=pred64,
            residual=true_residual,
            frequency=ctx["frequency"],
        )

        direction = normalize_direction(g)

        trial = (
            candidate
            - args.initial_step
            * direction
        ).astype(np.float32)

        return trial

    trial, grad_s = timed_call(
        _grad_update,
        device,
    )

    cuda_sync(device)
    core_s = float(
        time.perf_counter()
        - t_core0
    )

    mem = read_peak_memory(
        device,
        baseline,
    )

    return {
        "trial": trial,
        "J0": float(J0),
        "rr0": float(rr0),
        "accepted": True,
        "accepted_step_mps":
            float(args.initial_step),
        "num_trial_evals": 0,
        "true_source_equiv_core": 8,
        "exact64_true_cbs_seconds": 0.0,
        "anchor_true8_seconds":
            float(anchor_s),
        "guard_true8_seconds": 0.0,
        "neural64_seconds":
            float(neural_s),
        "gradient_update_seconds":
            float(grad_s),
        "method_core_seconds":
            float(core_s),
        **mem,
    }


def run_gra_once(
    ctx,
    args,
    device,
    model,
    background64,
    attempt_sink=None,
    repeat_index=None,
):
    """
    Frozen GRA-ARSS core:
        exact 8-source anchor/residual
        + MgNO64
        + RA direction
        + monotone 8-source true-CBS backtracking.

    The backtracking evaluations ARE counted in method cost.
    """
    baseline = reset_peak_memory(device)
    candidate = ctx["candidate"]

    t_core0 = time.perf_counter()
    cuda_sync(device)

    true8, anchor_s = timed_call(
        lambda: true_forward(
            candidate,
            ctx["src_indices"],
            ctx,
            args.device,
            args.chunk_size,
        ),
        device,
    )

    true_measurement = (
        true8_measurement_from_fields(
            true8,
            ctx,
        )
    )

    J0, rr0, true_residual = (
        objective_from_measurement(
            true_measurement,
            ctx["dobs"],
        )
    )

    pred64, neural_s = timed_call(
        lambda: neural_forward64(
            model=model,
            speed=candidate,
            background64=background64,
            speed_mean=args.speed_mean,
            speed_std=args.speed_std,
            wave_scale=args.wave_scale,
            device_str=args.device,
        ),
        device,
    )

    neural_tx, _ = (
        neural8_measurement_from_pred64(
            pred64,
            ctx,
        )
    )

    def _grad_direction():
        g = ano_gradient(
            candidate=candidate,
            tx_waves=neural_tx,
            basis64=pred64,
            residual=true_residual,
            frequency=ctx["frequency"],
        )
        return normalize_direction(g)

    direction, grad_s = timed_call(
        _grad_direction,
        device,
    )

    accepted = False
    accepted_step = 0.0
    accepted_trial = candidate.copy()
    accepted_J = float(J0)
    accepted_rr = float(rr0)

    num_trial_evals = 0
    guard_s = 0.0

    for k in range(
        args.max_backtracks + 1
    ):
        step = (
            args.initial_step
            * args.backtrack_factor ** k
        )

        trial = (
            candidate
            - step
            * direction
        ).astype(np.float32)

        J_trial, rr_trial, trial_s = (
            eval_trial_objective(
                trial,
                ctx,
                args,
                device,
            )
        )

        num_trial_evals += 1
        guard_s += trial_s

        drop = pct_drop(
            J0,
            J_trial,
        )

        ok = bool(
            J_trial < J0
        )

        if attempt_sink is not None:
            attempt_sink.append({
                "sample":
                    int(ctx["sample"]),
                "repeat":
                    (
                        -1
                        if repeat_index is None
                        else int(repeat_index)
                    ),
                "attempt":
                    int(k),
                "step_mps":
                    float(step),
                "J0":
                    float(J0),
                "J1":
                    float(J_trial),
                "relative_drop":
                    float(drop),
                "rr0":
                    float(rr0),
                "rr1":
                    float(rr_trial),
                "accepted":
                    bool(ok),
                "trial_seconds":
                    float(trial_s),
            })

        if ok:
            accepted = True
            accepted_step = float(step)
            accepted_trial = trial
            accepted_J = float(J_trial)
            accepted_rr = float(rr_trial)
            break

    cuda_sync(device)
    core_s = float(
        time.perf_counter()
        - t_core0
    )

    mem = read_peak_memory(
        device,
        baseline,
    )

    true_source_equiv = (
        8
        * (
            1
            + num_trial_evals
        )
    )

    return {
        "trial": accepted_trial,
        "J0": float(J0),
        "rr0": float(rr0),
        "J_final_online":
            float(accepted_J),
        "rr_final_online":
            float(accepted_rr),
        "accepted":
            bool(accepted),
        "accepted_step_mps":
            float(accepted_step),
        "num_trial_evals":
            int(num_trial_evals),
        "true_source_equiv_core":
            int(true_source_equiv),
        "exact64_true_cbs_seconds":
            0.0,
        "anchor_true8_seconds":
            float(anchor_s),
        "guard_true8_seconds":
            float(guard_s),
        "neural64_seconds":
            float(neural_s),
        "gradient_update_seconds":
            float(grad_s),
        "method_core_seconds":
            float(core_s),
        **mem,
    }


# ---------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------

METHODS = [
    "Exact-CBS",
    "Full-ARSS",
    "RA-ARSS",
    "GRA-ARSS",
]


def run_method_once(
    method,
    ctx,
    args,
    device,
    model,
    background64,
    attempt_sink=None,
    repeat_index=None,
):
    if method == "Exact-CBS":
        return run_exact_once(
            ctx,
            args,
            device,
        )

    if method == "Full-ARSS":
        return run_full_once(
            ctx,
            args,
            device,
            model,
            background64,
        )

    if method == "RA-ARSS":
        return run_ra_once(
            ctx,
            args,
            device,
            model,
            background64,
        )

    if method == "GRA-ARSS":
        return run_gra_once(
            ctx,
            args,
            device,
            model,
            background64,
            attempt_sink=attempt_sink,
            repeat_index=repeat_index,
        )

    raise ValueError(method)


def benchmark_one_method(
    method,
    ctx,
    args,
    device,
    model,
    background64,
    reference_J0,
    reference_rr0,
    run_rows,
    attempt_rows,
):
    print()
    print("-" * 120)
    print(
        f"{method} | "
        f"warmup={args.warmup} | "
        f"repeats={args.repeats}"
    )
    print("-" * 120)

    # ---------------------------
    # Full-method warmup.
    # ---------------------------
    for w in range(args.warmup):
        _ = run_method_once(
            method,
            ctx,
            args,
            device,
            model,
            background64,
            attempt_sink=None,
            repeat_index=None,
        )

        print(
            f"  warmup {w+1}/{args.warmup} done"
        )

    # ---------------------------
    # Recorded runs.
    # ---------------------------
    method_rows = []

    for rep in range(args.repeats):
        result = run_method_once(
            method,
            ctx,
            args,
            device,
            model,
            background64,
            attempt_sink=attempt_rows,
            repeat_index=rep,
        )

        trial = result["trial"]

        # ----------------------------------------------------------
        # Evaluation-only objective.
        #
        # For GRA, the accepted/final objective has already been
        # measured online by the monotone guard, so no duplicate
        # CBS solve is performed here.
        # ----------------------------------------------------------
        if method == "GRA-ARSS":
            J1 = float(
                result["J_final_online"]
            )
            rr1 = float(
                result["rr_final_online"]
            )
            validation_s = 0.0

        else:
            J1, rr1, validation_s = (
                eval_trial_objective(
                    trial,
                    ctx,
                    args,
                    device,
                )
            )

        if result["J0"] is None:
            J0 = float(reference_J0)
            rr0 = float(reference_rr0)
        else:
            J0 = float(result["J0"])
            rr0 = float(result["rr0"])

        drop = pct_drop(
            J0,
            J1,
        )

        mse0 = ctx["mse0"]
        mae0 = ctx["mae0"]

        mse1 = mse(
            trial,
            ctx["target"],
        )

        mae1 = mae(
            trial,
            ctx["target"],
        )

        mse_gain = (
            (mse0 - mse1)
            /
            max(mse0, 1e-30)
        )

        true_cbs_core_s = (
            result[
                "exact64_true_cbs_seconds"
            ]
            +
            result[
                "anchor_true8_seconds"
            ]
            +
            result[
                "guard_true8_seconds"
            ]
        )

        e2e_validation_s = (
            result["method_core_seconds"]
            + validation_s
        )

        row = {
            "sample":
                int(ctx["sample"]),

            "method":
                method,

            "repeat":
                int(rep),

            "candidate_sha256":
                ctx["candidate_sha256"],

            "method_core_seconds":
                float(
                    result[
                        "method_core_seconds"
                    ]
                ),

            "validation_only_seconds":
                float(validation_s),

            "end_to_end_with_validation_seconds":
                float(e2e_validation_s),

            "true_cbs_core_seconds":
                float(true_cbs_core_s),

            "exact64_true_cbs_seconds":
                float(
                    result[
                        "exact64_true_cbs_seconds"
                    ]
                ),

            "anchor_true8_seconds":
                float(
                    result[
                        "anchor_true8_seconds"
                    ]
                ),

            "guard_true8_seconds":
                float(
                    result[
                        "guard_true8_seconds"
                    ]
                ),

            "neural64_seconds":
                float(
                    result[
                        "neural64_seconds"
                    ]
                ),

            "gradient_update_seconds":
                float(
                    result[
                        "gradient_update_seconds"
                    ]
                ),

            "true_source_equiv_core":
                int(
                    result[
                        "true_source_equiv_core"
                    ]
                ),

            "accepted":
                bool(
                    result["accepted"]
                ),

            "accepted_step_mps":
                float(
                    result[
                        "accepted_step_mps"
                    ]
                ),

            "num_trial_evals":
                int(
                    result[
                        "num_trial_evals"
                    ]
                ),

            "J0":
                float(J0),

            "J1":
                float(J1),

            "relative_drop":
                float(drop),

            "descent":
                bool(
                    J1 < J0
                ),

            "rr0":
                float(rr0),

            "rr1":
                float(rr1),

            "mse0":
                float(mse0),

            "mse1":
                float(mse1),

            "mse_gain":
                float(mse_gain),

            "mae0":
                float(mae0),

            "mae1":
                float(mae1),

            "baseline_allocated_mb":
                float(
                    result[
                        "baseline_allocated_mb"
                    ]
                ),

            "baseline_reserved_mb":
                float(
                    result[
                        "baseline_reserved_mb"
                    ]
                ),

            "peak_allocated_mb":
                float(
                    result[
                        "peak_allocated_mb"
                    ]
                ),

            "peak_reserved_mb":
                float(
                    result[
                        "peak_reserved_mb"
                    ]
                ),

            "peak_delta_allocated_mb":
                float(
                    result[
                        "peak_delta_allocated_mb"
                    ]
                ),

            "peak_delta_reserved_mb":
                float(
                    result[
                        "peak_delta_reserved_mb"
                    ]
                ),
        }

        run_rows.append(row)
        method_rows.append(row)

        print(
            f"  rep={rep+1}/{args.repeats} | "
            f"core={row['method_core_seconds']:.4f}s | "
            f"val={row['validation_only_seconds']:.4f}s | "
            f"drop={100*drop:+.3f}% | "
            f"src-eq={row['true_source_equiv_core']} | "
            f"peakΔ={row['peak_delta_allocated_mb']:.1f}MB"
        )

    return method_rows


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def aggregate_per_sample(run_rows):
    rows = []

    sample_ids = sorted(
        set(r["sample"] for r in run_rows)
    )

    for sample_id in sample_ids:
        for method in METHODS:
            sub = [
                r
                for r in run_rows
                if (
                    r["sample"] == sample_id
                    and
                    r["method"] == method
                )
            ]

            if not sub:
                continue

            first = sub[0]

            core = [
                r["method_core_seconds"]
                for r in sub
            ]

            e2e = [
                r[
                    "end_to_end_with_validation_seconds"
                ]
                for r in sub
            ]

            true_cbs = [
                r["true_cbs_core_seconds"]
                for r in sub
            ]

            neural = [
                r["neural64_seconds"]
                for r in sub
            ]

            grad = [
                r["gradient_update_seconds"]
                for r in sub
            ]

            peak_delta = [
                r["peak_delta_allocated_mb"]
                for r in sub
            ]

            source_eq = [
                r["true_source_equiv_core"]
                for r in sub
            ]

            rows.append({
                "sample":
                    int(sample_id),

                "method":
                    method,

                "repeats":
                    len(sub),

                "core_seconds_mean":
                    float(np.mean(core)),

                "core_seconds_std":
                    float(np.std(core)),

                "core_seconds_median":
                    float(np.median(core)),

                "e2e_with_validation_mean":
                    float(np.mean(e2e)),

                "e2e_with_validation_median":
                    float(np.median(e2e)),

                "true_cbs_core_seconds_mean":
                    float(np.mean(true_cbs)),

                "neural64_seconds_mean":
                    float(np.mean(neural)),

                "gradient_update_seconds_mean":
                    float(np.mean(grad)),

                "peak_delta_allocated_mb_mean":
                    float(np.mean(peak_delta)),

                "peak_allocated_mb_max":
                    float(
                        max(
                            r["peak_allocated_mb"]
                            for r in sub
                        )
                    ),

                "true_source_equiv_mean":
                    float(np.mean(source_eq)),

                "accepted":
                    bool(first["accepted"]),

                "accepted_step_mps":
                    float(
                        first["accepted_step_mps"]
                    ),

                "num_trial_evals":
                    int(
                        first["num_trial_evals"]
                    ),

                "J0":
                    float(first["J0"]),

                "J1":
                    float(first["J1"]),

                "relative_drop":
                    float(
                        first["relative_drop"]
                    ),

                "descent":
                    bool(first["descent"]),

                "rr0":
                    float(first["rr0"]),

                "rr1":
                    float(first["rr1"]),

                "mse0":
                    float(first["mse0"]),

                "mse1":
                    float(first["mse1"]),

                "mse_gain":
                    float(first["mse_gain"]),

                "mae0":
                    float(first["mae0"]),

                "mae1":
                    float(first["mae1"]),

                "candidate_sha256":
                    first["candidate_sha256"],
            })

    return rows


def aggregate_methods(per_sample_rows):
    summary = {}
    csv_rows = []

    for method in METHODS:
        sub = [
            r
            for r in per_sample_rows
            if r["method"] == method
        ]

        if not sub:
            continue

        core = [
            r["core_seconds_median"]
            for r in sub
        ]

        e2e = [
            r["e2e_with_validation_median"]
            for r in sub
        ]

        drops = [
            r["relative_drop"]
            for r in sub
        ]

        mse_gains = [
            r["mse_gain"]
            for r in sub
        ]

        source_eq = [
            r["true_source_equiv_mean"]
            for r in sub
        ]

        peak_delta = [
            r["peak_delta_allocated_mb_mean"]
            for r in sub
        ]

        descent_count = int(
            sum(
                bool(r["descent"])
                for r in sub
            )
        )

        accepted_count = int(
            sum(
                bool(r["accepted"])
                for r in sub
            )
        )

        s = {
            "num_samples":
                len(sub),

            "descent_count":
                descent_count,

            "accepted_count":
                accepted_count,

            "method_core_seconds":
                stat(core),

            "end_to_end_with_validation_seconds":
                stat(e2e),

            "relative_drop":
                stat(drops),

            "mse_gain":
                stat(mse_gains),

            "true_source_equiv_core":
                stat(source_eq),

            "peak_delta_allocated_mb":
                stat(peak_delta),

            "true_cbs_core_seconds":
                stat([
                    r["true_cbs_core_seconds_mean"]
                    for r in sub
                ]),

            "neural64_seconds":
                stat([
                    r["neural64_seconds_mean"]
                    for r in sub
                ]),

            "gradient_update_seconds":
                stat([
                    r["gradient_update_seconds_mean"]
                    for r in sub
                ]),
        }

        summary[method] = s

        csv_rows.append({
            "method":
                method,

            "num_samples":
                len(sub),

            "descent_count":
                descent_count,

            "accepted_count":
                accepted_count,

            "core_seconds_mean_of_sample_medians":
                s[
                    "method_core_seconds"
                ]["mean"],

            "core_seconds_median_of_sample_medians":
                s[
                    "method_core_seconds"
                ]["median"],

            "e2e_seconds_mean_of_sample_medians":
                s[
                    "end_to_end_with_validation_seconds"
                ]["mean"],

            "mean_objective_drop_pct":
                100.0
                * s[
                    "relative_drop"
                ]["mean"],

            "worst_objective_drop_pct":
                100.0
                * s[
                    "relative_drop"
                ]["min"],

            "mean_mse_gain_pct":
                100.0
                * s[
                    "mse_gain"
                ]["mean"],

            "mean_true_source_equiv_core":
                s[
                    "true_source_equiv_core"
                ]["mean"],

            "mean_true_cbs_core_seconds":
                s[
                    "true_cbs_core_seconds"
                ]["mean"],

            "mean_neural64_seconds":
                s[
                    "neural64_seconds"
                ]["mean"],

            "mean_gradient_update_seconds":
                s[
                    "gradient_update_seconds"
                ]["mean"],

            "mean_peak_delta_allocated_mb":
                s[
                    "peak_delta_allocated_mb"
                ]["mean"],
        })

    # Core-time speedups relative to Exact-CBS.
    if "Exact-CBS" in summary:
        exact_core = summary[
            "Exact-CBS"
        ][
            "method_core_seconds"
        ]["mean"]

        for method, s in summary.items():
            this_core = s[
                "method_core_seconds"
            ]["mean"]

            if this_core is not None and this_core > 0:
                s[
                    "speedup_vs_exact_core"
                ] = float(
                    exact_core / this_core
                )
            else:
                s[
                    "speedup_vs_exact_core"
                ] = None

        for row in csv_rows:
            row[
                "speedup_vs_exact_core"
            ] = summary[
                row["method"]
            ].get(
                "speedup_vs_exact_core"
            )

    return summary, csv_rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "C6.4C real efficiency audit of "
            "Exact-CBS / Full-ARSS / RA-ARSS / GRA-ARSS."
        )
    )

    ap.add_argument(
        "--bridge_dir",
        required=True,
    )

    ap.add_argument(
        "--background64_npz",
        required=True,
    )

    ap.add_argument(
        "--mgno_ckpt",
        required=True,
    )

    ap.add_argument(
        "--output_dir",
        required=True,
    )

    ap.add_argument(
        "--start",
        type=int,
        default=11,
    )

    ap.add_argument(
        "--end",
        type=int,
        default=20,
    )

    # Frozen C6.4B/GRA settings.
    ap.add_argument(
        "--initial_step",
        type=float,
        default=1.5,
    )

    ap.add_argument(
        "--backtrack_factor",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--max_backtracks",
        type=int,
        default=5,
    )

    # Frozen MgNO normalization.
    ap.add_argument(
        "--speed_mean",
        type=float,
        default=1488.39,
    )

    ap.add_argument(
        "--speed_std",
        type=float,
        default=27.53,
    )

    ap.add_argument(
        "--wave_scale",
        type=float,
        default=3.72290883e-02,
    )

    ap.add_argument(
        "--chunk_size",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--device",
        default="cuda:0",
    )

    ap.add_argument(
        "--split_label",
        default="heldout_test11_20",
    )

    # C6.4C timing protocol.
    ap.add_argument(
        "--warmup",
        type=int,
        default=1,
        help=(
            "Unrecorded full-method warmup runs "
            "per sample and per method."
        ),
    )

    ap.add_argument(
        "--repeats",
        type=int,
        default=3,
        help=(
            "Recorded full-method timing repeats "
            "per sample and per method."
        ),
    )

    args = ap.parse_args()

    if args.start > args.end:
        raise ValueError("--start must be <= --end")

    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    bridge_root = Path(
        args.bridge_dir
    )

    out = Path(
        args.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = resolve_device(
        args.device
    )

    # --------------------------------------------------------------
    # Background basis.
    # --------------------------------------------------------------
    bz = np.load(
        args.background64_npz,
        allow_pickle=True,
    )

    background64 = bz[
        "background64"
    ].astype(np.complex64)

    background_rec = bz[
        "rec_indices"
    ].astype(np.int64)

    if background64.shape[0] != 64:
        raise RuntimeError(
            "background64 must contain "
            "64 source fields; "
            f"got {background64.shape}"
        )

    # --------------------------------------------------------------
    # Load frozen Strat4+4 MgNO once.
    # Model load is reported separately and excluded from per-sample
    # method-core latency.
    # --------------------------------------------------------------
    cuda_sync(device)
    t_model0 = time.perf_counter()

    (
        model,
        channels,
        vcycles,
    ) = load_mgno(
        args.mgno_ckpt,
        args.device,
    )

    cuda_sync(device)
    model_load_seconds = float(
        time.perf_counter()
        - t_model0
    )

    print("=" * 132)
    print(
        "C6.4C REAL EFFICIENCY AUDIT "
        "(Exact-CBS / Full-ARSS / RA-ARSS / GRA-ARSS)"
    )
    print("=" * 132)

    print("bridge             =", bridge_root)
    print("split              =", args.split_label)
    print("samples            =", f"{args.start}..{args.end}")
    print("device             =", args.device)
    print("MgNO ckpt          =", args.mgno_ckpt)
    print("MgNO config        =", f"C{channels}/V{vcycles}")
    print("model load         =", f"{model_load_seconds:.4f}s")
    print("initial RMS step   =", args.initial_step, "m/s")
    print("backtrack factor   =", args.backtrack_factor)
    print("max backtracks     =", args.max_backtracks)
    print("warmup/method      =", args.warmup)
    print("repeats/method     =", args.repeats)
    print("chunk size         =", args.chunk_size)

    run_rows = []
    attempt_rows = []
    reference_rows = []

    # ==============================================================
    # Sample loop
    # ==============================================================
    for sample_id in range(
        args.start,
        args.end + 1,
    ):
        print()
        print("#" * 132)
        print(f"TEST {sample_id}")
        print("#" * 132)

        ctx = load_sample_context(
            bridge_root,
            sample_id,
            background_rec,
        )

        # ----------------------------------------------------------
        # Evaluation-only reference J0.
        #
        # This is deliberately outside every method-core timer.
        # It also warms the CBS kernels before benchmark warmups.
        # RA/GRA still recompute their own exact 8-source anchor
        # inside their timed method cores, because that anchor is
        # part of those algorithms.
        # ----------------------------------------------------------
        ref_true8, ref_s = timed_call(
            lambda: true_forward(
                ctx["candidate"],
                ctx["src_indices"],
                ctx,
                args.device,
                args.chunk_size,
            ),
            device,
        )

        ref_meas = (
            true8_measurement_from_fields(
                ref_true8,
                ctx,
            )
        )

        (
            reference_J0,
            reference_rr0,
            _,
        ) = objective_from_measurement(
            ref_meas,
            ctx["dobs"],
        )

        reference_rows.append({
            "sample":
                int(sample_id),

            "J0":
                float(reference_J0),

            "rr0":
                float(reference_rr0),

            "reference_true8_seconds":
                float(ref_s),

            "mse0":
                float(ctx["mse0"]),

            "mae0":
                float(ctx["mae0"]),

            "candidate_sha256":
                ctx["candidate_sha256"],
        })

        print(
            f"reference J0={reference_J0:.10e} | "
            f"RR0={reference_rr0:.8f} | "
            f"MSE0={ctx['mse0']:.4f} | "
            f"reference-only true8={ref_s:.4f}s"
        )

        for method in METHODS:
            benchmark_one_method(
                method=method,
                ctx=ctx,
                args=args,
                device=device,
                model=model,
                background64=background64,
                reference_J0=reference_J0,
                reference_rr0=reference_rr0,
                run_rows=run_rows,
                attempt_rows=attempt_rows,
            )

    # ==============================================================
    # Save raw results
    # ==============================================================
    per_run_csv = (
        out / "per_run_timing.csv"
    )

    reference_csv = (
        out / "reference_objectives.csv"
    )

    attempts_csv = (
        out / "gra_backtracking_attempts.csv"
    )

    write_csv(
        per_run_csv,
        run_rows,
    )

    write_csv(
        reference_csv,
        reference_rows,
    )

    write_csv(
        attempts_csv,
        attempt_rows,
    )

    # ==============================================================
    # Aggregate per sample and per method
    # ==============================================================
    per_sample_rows = (
        aggregate_per_sample(
            run_rows
        )
    )

    per_sample_csv = (
        out / "per_sample_timing.csv"
    )

    write_csv(
        per_sample_csv,
        per_sample_rows,
    )

    (
        method_summary,
        method_summary_rows,
    ) = aggregate_methods(
        per_sample_rows
    )

    method_summary_csv = (
        out / "method_summary.csv"
    )

    write_csv(
        method_summary_csv,
        method_summary_rows,
    )

    # --------------------------------------------------------------
    # C6.4C gate.
    #
    # This is intentionally an efficiency characterization gate,
    # not a new hyperparameter-selection gate.
    #
    # GRA must:
    #   1) retain strict descent on >= 8/10 if 10 samples are used;
    #   2) have positive mean objective reduction;
    #   3) use fewer than 64 true-source equivalents on average.
    #
    # We DO NOT require a particular wall-clock speedup beforehand;
    # that is exactly what C6.4C is measuring.
    # --------------------------------------------------------------
    gra_s = method_summary.get(
        "GRA-ARSS"
    )

    c64c_gate = "FAIL"

    if gra_s is not None:
        n = gra_s["num_samples"]

        strong_descent_threshold = (
            8
            if n == 10
            else int(
                math.ceil(
                    0.8 * n
                )
            )
        )

        cond_descent = (
            gra_s["descent_count"]
            >=
            strong_descent_threshold
        )

        cond_drop = (
            gra_s[
                "relative_drop"
            ]["mean"] > 0
        )

        cond_source = (
            gra_s[
                "true_source_equiv_core"
            ]["mean"] < 64
        )

        if (
            cond_descent
            and
            cond_drop
            and
            cond_source
        ):
            c64c_gate = (
                "EFFICIENCY_CHARACTERIZED_PASS"
            )

        elif (
            cond_descent
            and
            cond_drop
        ):
            c64c_gate = (
                "DESCENT_PASS_COST_NOT_REDUCED"
            )

    summary = {
        "experiment":
            "C6.4C Real Efficiency Audit",

        "derived_from":
            "C6.4B Held-out GRA-ARSS",

        "evaluation_split":
            args.split_label,

        "sample_ids": [
            args.start,
            args.end,
        ],

        "num_samples":
            int(
                args.end
                - args.start
                + 1
            ),

        "device":
            args.device,

        "mgno_config": {
            "checkpoint":
                args.mgno_ckpt,
            "channels":
                int(channels),
            "vcycles":
                int(vcycles),
            "speed_mean":
                float(args.speed_mean),
            "speed_std":
                float(args.speed_std),
            "wave_scale":
                float(args.wave_scale),
        },

        "frozen_gra_protocol": {
            "initial_step_mps":
                float(args.initial_step),
            "backtrack_factor":
                float(args.backtrack_factor),
            "max_backtracks":
                int(args.max_backtracks),
            "acceptance_rule":
                "J_trial < J_current",
            "gt_used_for_acceptance":
                False,
            "cosine_used_for_acceptance":
                False,
        },

        "timing_protocol": {
            "cuda_synchronize_before_after":
                bool(
                    device.type == "cuda"
                ),
            "warmup_runs_per_sample_method":
                int(args.warmup),
            "recorded_repeats_per_sample_method":
                int(args.repeats),
            "model_load_seconds":
                float(
                    model_load_seconds
                ),
            "model_load_excluded_from_per_sample_latency":
                True,
            "reference_J0_true8_excluded_from_method_core":
                True,
            "fixed_method_final_objective_validation_excluded_from_core":
                True,
            "gra_backtracking_true8_included_in_core":
                True,
            "primary_latency_metric":
                (
                    "mean across samples of each sample's "
                    "median method_core_seconds"
                ),
        },

        "cost_convention": {
            "Exact-CBS":
                (
                    "64 exact source solves + exact gradient/update; "
                    "final objective evaluation is validation-only."
                ),
            "Full-ARSS":
                (
                    "MgNO64 + neural residual + gradient/update; "
                    "true-CBS objective checks are validation-only."
                ),
            "RA-ARSS":
                (
                    "8-source exact anchor/residual + MgNO64 "
                    "+ gradient/update; final objective check "
                    "is validation-only."
                ),
            "GRA-ARSS":
                (
                    "8-source exact anchor/residual + MgNO64 "
                    "+ gradient/update + all 8-source guarded "
                    "backtracking evaluations."
                ),
        },

        "methods":
            method_summary,

        "c6_4c_gate":
            c64c_gate,

        "files": {
            "per_run_timing_csv":
                str(per_run_csv),
            "per_sample_timing_csv":
                str(per_sample_csv),
            "method_summary_csv":
                str(method_summary_csv),
            "gra_backtracking_attempts_csv":
                str(attempts_csv),
            "reference_objectives_csv":
                str(reference_csv),
        },
    }

    summary_path = (
        out / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ==============================================================
    # Console summary
    # ==============================================================
    print()
    print("=" * 132)
    print("C6.4C FINAL EFFICIENCY SUMMARY")
    print("=" * 132)

    for method in METHODS:
        if method not in method_summary:
            continue

        s = method_summary[
            method
        ]

        print()
        print(method)

        print(
            "  descent        =",
            f"{s['descent_count']}"
            f"/{s['num_samples']}",
        )

        print(
            "  mean drop      =",
            f"{100*s['relative_drop']['mean']:+.3f}%",
        )

        print(
            "  core time mean =",
            f"{s['method_core_seconds']['mean']:.4f}s",
        )

        print(
            "  core time med  =",
            f"{s['method_core_seconds']['median']:.4f}s",
        )

        print(
            "  speedup/exact  =",
            (
                "N/A"
                if s.get(
                    "speedup_vs_exact_core"
                ) is None
                else
                f"{s['speedup_vs_exact_core']:.3f}x"
            ),
        )

        print(
            "  source-equiv   =",
            f"{s['true_source_equiv_core']['mean']:.2f}",
        )

        print(
            "  peak delta mem =",
            f"{s['peak_delta_allocated_mb']['mean']:.1f} MB",
        )

    print()
    print(
        "C6.4C GATE =",
        c64c_gate,
    )

    print(
        "saved summary =",
        summary_path,
    )


if __name__ == "__main__":
    main()
