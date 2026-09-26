#!/usr/bin/env python3
"""
C6.5A — Two-step Conditional CM + Physics Trajectory Integration

Goal
----
Integrate the frozen C6.4 physics update INSIDE the frozen C4 conditional
consistency-model trajectory, rather than applying physics after CM sampling.

Frozen C4 sampler:
    checkpoint 5000
    multistep
    steps = 40
    sigma_min = 0.002
    sigma_max = 80
    rho = 7
    ts = [0, 17, 39]
    seed = 20260908 (default, configurable only for reproduction)

For ts=[0,17,39], there are exactly two consistency mappings:
    1) x_T -> clean x0^(1)
    2) re-noise to sigma(ts=17) -> clean x0^(2)

C6.5 inserts one physics update between them:

    x_T
      -> CC(x_T, T) = x0^(1)
      -> physics correction in 480x480 domain
      -> downsample corrected clean estimate to 256x256
      -> SAME forward-SDE noise as the C4-only paired arm
      -> CC(x_tau1, tau1)
      -> final x0

Compared arms
-------------
1. Hint
2. C4-only
3. C4 + Exact-CBS
4. C4 + GRA-ARSS

Important
---------
- No GT / MSE / SSIM is used in physics acceptance.
- Exact-CBS uses frozen RMS step = 1.5 m/s.
- GRA-ARSS uses frozen C6.4B protocol:
      true8 residual anchor
      + frozen MgNO64 wavefield/basis
      + RMS-normalized RA direction
      + monotone true8 line search
      alpha0=1.5, factor=0.5, max_backtracks=5
      accept iff J_trial < J_current
- The random generator is reset per arm to the same deterministic individual
  stream. Therefore initial x_T and forward-SDE noise are paired across arms.
- Existing bridge candidate_480 is used ONLY for geometry/data and C4
  reproduction/interpolation audit. It is not used as the trajectory state.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F

from c5_audit_mgno_unseen64_sources import (
    solve_cbs_chunked,
    load_mgno,
    neural_predict,
)
from c5_paper_ano_val5 import ano_gradient
from c5_directional_fd_audit import (
    measurement_loss,
    normalize_direction,
    scalar,
)


CENTER = 1502.5
SCALE = 102.5
SPEED_MIN = 1400.0
SPEED_MAX = 1605.0
DATA_RANGE = SPEED_MAX - SPEED_MIN

C4_TS = (0, 17, 39)
C4_STEPS = 40
C4_SIGMA_MIN = 0.002
C4_SIGMA_MAX = 80.0
C4_RHO = 7.0


# =====================================================================
# Utilities
# =====================================================================

def add_cosign_to_path(cosign_root):
    root = Path(cosign_root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def sha256_array(x):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    return hashlib.sha256(x.tobytes()).hexdigest()


def norm_to_speed(x):
    return np.asarray(x, dtype=np.float32) * SCALE + CENTER


def speed_to_norm(x):
    return (np.asarray(x, dtype=np.float32) - CENTER) / SCALE


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


def metric_dict(pred_speed, gt_speed):
    from skimage.metrics import structural_similarity

    pred_speed = np.asarray(pred_speed, dtype=np.float64)
    gt_speed = np.asarray(gt_speed, dtype=np.float64)

    m = mse(pred_speed, gt_speed)
    a = mae(pred_speed, gt_speed)

    p = float(
        10.0 * math.log10(
            DATA_RANGE**2 / max(m, 1e-30)
        )
    )

    s = float(
        structural_similarity(
            pred_speed,
            gt_speed,
            data_range=DATA_RANGE,
        )
    )

    return {
        "mse": m,
        "mae": a,
        "psnr": p,
        "ssim": s,
    }


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


def relative_drop(J0, J1):
    return float(
        (J0 - J1)
        / max(abs(J0), 1e-30)
    )


def stat(values):
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return

    # Rows from different arms intentionally carry different
    # diagnostics (e.g. Exact vs GRA timing fields).  Build a
    # stable union of all keys instead of assuming rows[0] defines
    # the complete schema.
    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        w.writeheader()
        w.writerows(rows)


def cuda_sync(device):
    if device.type == "cuda":
        th.cuda.synchronize(device)


def timed_call(fn, device):
    cuda_sync(device)
    t0 = time.perf_counter()
    out = fn()
    cuda_sync(device)
    return out, float(time.perf_counter() - t0)


# =====================================================================
# Bilinear image/physics-domain bridge
# =====================================================================

def resize_2d_np(
    x,
    out_hw,
    align_corners,
):
    """
    Bilinear resize of one 2-D real field using PyTorch.
    """
    a = th.from_numpy(
        np.asarray(x, dtype=np.float32)
    )[None, None]

    y = F.interpolate(
        a,
        size=tuple(out_hw),
        mode="bilinear",
        align_corners=align_corners,
    )

    return (
        y[0, 0]
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def choose_bridge_alignment(
    final_c4_norm_256,
    bridge_candidate_480,
):
    """
    Determine which PyTorch bilinear convention reproduces the existing
    C6.2 bridge candidate most closely.
    """
    # C4 / condition-cache use image coordinates, while the
    # CBS bridge uses physical coordinates.  C6.2 froze the mapping
    # as a 2-D transpose before interpolation.
    speed256 = norm_to_speed(
        final_c4_norm_256
    ).T.copy()

    candidates = {}

    for ac in [False, True]:
        x480 = resize_2d_np(
            speed256,
            (480, 480),
            align_corners=ac,
        )
        candidates[ac] = {
            "x480": x480,
            "rrmse": rrmse(
                x480,
                bridge_candidate_480,
            ),
            "mae": mae(
                x480,
                bridge_candidate_480,
            ),
        }

    best_ac = min(
        candidates,
        key=lambda k:
            candidates[k]["rrmse"],
    )

    return (
        best_ac,
        candidates,
    )


def norm256_to_speed480(
    x_norm_256,
    align_corners,
):
    # Frozen C6.2 coordinate convention:
    # image-domain (C4) -> transpose -> physics-domain (CBS).
    speed256_physics = (
        norm_to_speed(x_norm_256)
        .T
        .copy()
    )

    return resize_2d_np(
        speed256_physics,
        (480, 480),
        align_corners=align_corners,
    )


def speed480_to_norm256(
    x_speed_480,
    align_corners,
):
    speed256_physics = resize_2d_np(
        x_speed_480,
        (256, 256),
        align_corners=align_corners,
    )

    # Inverse of the frozen C6.2 mapping:
    # physics-domain -> transpose -> image-domain.
    speed256_image = (
        speed256_physics
        .T
        .copy()
    )

    return speed_to_norm(
        speed256_image
    )


# =====================================================================
# Load C4 models
# =====================================================================

def create_c4_models(
    backbone,
    control,
    device,
):
    from cc.script_util import (
        create_model_and_diffusion,
    )

    control_net, model, diffusion = (
        create_model_and_diffusion(
            image_size=256,
            class_cond=False,
            learn_sigma=False,
            num_channels=256,
            num_res_blocks=2,
            channel_mult="",
            num_heads=4,
            num_head_channels=64,
            num_heads_upsample=-1,
            attention_resolutions="32,16,8",
            dropout=0.0,
            use_checkpoint=False,
            use_scale_shift_norm=False,
            resblock_updown=True,
            use_fp16=True,
            use_new_attention_order=False,
            weight_schedule="uniform",
            sigma_min=C4_SIGMA_MIN,
            sigma_max=C4_SIGMA_MAX,
            loss_norm="l2",
            loss_type="recon",
            distillation=True,
            control=True,
            in_channels=1,
        )
    )

    model.load_state_dict(
        th.load(
            backbone,
            map_location="cpu",
        ),
        strict=True,
    )

    control_net.load_state_dict(
        th.load(
            control,
            map_location="cpu",
        ),
        strict=True,
    )

    model.to(device)
    control_net.to(device)

    model.convert_to_fp16()
    control_net.convert_to_fp16()

    model.eval()
    control_net.eval()

    return (
        control_net,
        model,
        diffusion,
    )


def sigma_from_ts_index(
    ts_index,
    steps=C4_STEPS,
    sigma_min=C4_SIGMA_MIN,
    sigma_max=C4_SIGMA_MAX,
    rho=C4_RHO,
):
    t_max_rho = sigma_max ** (1.0 / rho)
    t_min_rho = sigma_min ** (1.0 / rho)

    t = (
        t_max_rho
        + ts_index / (steps - 1)
        * (t_min_rho - t_max_rho)
    ) ** rho

    return float(
        np.clip(
            t,
            sigma_min,
            sigma_max,
        )
    )


@th.no_grad()
def cc_recon(
    x_t,
    sigma,
    hint,
    control_net,
    model,
    diffusion,
):
    s = th.full(
        (x_t.shape[0],),
        float(sigma),
        dtype=x_t.dtype,
        device=x_t.device,
    )

    _, denoised = diffusion.recon(
        model,
        control_net,
        x_t,
        hint,
        s,
    )

    return denoised.clamp(-1, 1)


# =====================================================================
# Physics context
# =====================================================================

def load_bridge_sample(
    bridge_root,
    sample_id,
    background_rec,
):
    p = (
        Path(bridge_root)
        / f"test_{sample_id}.npz"
    )

    if not p.exists():
        raise FileNotFoundError(p)

    z = np.load(
        p,
        allow_pickle=True,
    )

    candidate_480 = z[
        "candidate_480"
    ].astype(np.float32)

    target_480 = z[
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

    return {
        "sample": sample_id,
        "candidate_480": candidate_480,
        "target_480": target_480,
        "dobs": dobs,
        "source_positions": source_positions,
        "src_indices": src_indices,
        "rec_indices": rec_indices,
        "frequency": frequency,
        "cbs_iters": cbs_iters,
        "boundary_width": boundary_width,
        "boundary_strength": boundary_strength,
        "boundary_type": boundary_type,
        "rr_idx": rec_indices[:, 0],
        "cc_idx": rec_indices[:, 1],
    }


def true_forward(
    speed,
    source_indices,
    ctx,
    args,
):
    y = solve_cbs_chunked(
        speed=speed,
        source_indices=source_indices,
        frequency=ctx["frequency"],
        cbs_iters=ctx["cbs_iters"],
        boundary_width=ctx[
            "boundary_width"
        ],
        boundary_strength=ctx[
            "boundary_strength"
        ],
        boundary_type=ctx[
            "boundary_type"
        ],
        device=args.device,
        chunk_size=args.chunk_size,
    )

    return np.asarray(
        y,
        dtype=np.complex64,
    )


def neural_forward64(
    model,
    speed,
    background64,
    args,
):
    y = neural_predict(
        model=model,
        speed=speed,
        backgrounds=background64,
        speed_mean=args.speed_mean,
        speed_std=args.speed_std,
        wave_scale=args.wave_scale,
        device=args.device,
    )

    return np.asarray(
        y,
        dtype=np.complex64,
    )


def measurement_from_true8(
    true8,
    ctx,
):
    return true8[
        :,
        ctx["rr_idx"],
        ctx["cc_idx"],
    ]


def measurement_objective_speed480(
    speed480,
    ctx,
    args,
):
    return measurement_loss(
        speed480,
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


# =====================================================================
# Frozen physics operators
# =====================================================================

def exact_physics_update(
    x_speed_480,
    ctx,
    args,
    device,
):
    """
    Frozen Exact-CBS direction with fixed RMS step.
    """
    true64, forward_s = timed_call(
        lambda: true_forward(
            x_speed_480,
            ctx["rec_indices"],
            ctx,
            args,
        ),
        device,
    )

    true8 = true64[
        ctx["source_positions"]
    ]

    meas = measurement_from_true8(
        true8,
        ctx,
    )

    J0, rr0, residual = (
        objective_from_measurement(
            meas,
            ctx["dobs"],
        )
    )

    def _grad():
        g = ano_gradient(
            candidate=x_speed_480,
            tx_waves=true8,
            basis64=true64,
            residual=residual,
            frequency=ctx["frequency"],
        )

        d = normalize_direction(g)

        trial = (
            x_speed_480
            - args.initial_step
            * d
        ).astype(np.float32)

        return trial

    trial, grad_s = timed_call(
        _grad,
        device,
    )

    # Online J1 is not required by Exact method itself,
    # but measured here for the trajectory diagnostic.
    (
        J1,
        rr1,
    ), eval_s = timed_call(
        lambda: measurement_objective_speed480(
            trial,
            ctx,
            args,
        ),
        device,
    )

    return trial, {
        "J_before": float(J0),
        "J_after": float(J1),
        "rr_before": float(rr0),
        "rr_after": float(rr1),
        "relative_drop":
            relative_drop(J0, J1),
        "accepted": bool(J1 < J0),
        "accepted_step_mps":
            float(args.initial_step),
        "num_trial_evals": 1,
        "true_source_equiv_core": 64,
        "true_source_equiv_with_diagnostic":
            72,
        "true64_seconds":
            float(forward_s),
        "neural64_seconds": 0.0,
        "guard_seconds": 0.0,
        "gradient_seconds":
            float(grad_s),
        "diagnostic_eval_seconds":
            float(eval_s),
    }


def gra_physics_update(
    x_speed_480,
    ctx,
    args,
    device,
    mgno_model,
    background64,
    attempt_rows,
):
    """
    Frozen C6.4B GRA-ARSS.
    """
    true8, anchor_s = timed_call(
        lambda: true_forward(
            x_speed_480,
            ctx["src_indices"],
            ctx,
            args,
        ),
        device,
    )

    true_meas = (
        measurement_from_true8(
            true8,
            ctx,
        )
    )

    J0, rr0, true_residual = (
        objective_from_measurement(
            true_meas,
            ctx["dobs"],
        )
    )

    pred64, neural_s = timed_call(
        lambda: neural_forward64(
            mgno_model,
            x_speed_480,
            background64,
            args,
        ),
        device,
    )

    neural_tx = pred64[
        ctx["source_positions"]
    ]

    def _direction():
        g_ra = ano_gradient(
            candidate=x_speed_480,
            tx_waves=neural_tx,
            basis64=pred64,
            residual=true_residual,
            frequency=ctx["frequency"],
        )
        return normalize_direction(g_ra)

    direction, grad_s = timed_call(
        _direction,
        device,
    )

    accepted = False
    accepted_step = 0.0
    accepted_trial = (
        x_speed_480.copy()
    )
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
            x_speed_480
            - step * direction
        ).astype(np.float32)

        (
            J_trial,
            rr_trial,
        ), trial_s = timed_call(
            lambda trial=trial:
                measurement_objective_speed480(
                    trial,
                    ctx,
                    args,
                ),
            device,
        )

        num_trial_evals += 1
        guard_s += trial_s

        drop = relative_drop(
            J0,
            J_trial,
        )

        ok = bool(
            J_trial < J0
        )

        attempt_rows.append({
            "sample":
                int(ctx["sample"]),
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

    return accepted_trial, {
        "J_before":
            float(J0),
        "J_after":
            float(accepted_J),
        "rr_before":
            float(rr0),
        "rr_after":
            float(accepted_rr),
        "relative_drop":
            relative_drop(
                J0,
                accepted_J,
            ),
        "accepted":
            bool(accepted),
        "accepted_step_mps":
            float(accepted_step),
        "num_trial_evals":
            int(num_trial_evals),
        "true_source_equiv_core":
            int(
                8 * (
                    1
                    + num_trial_evals
                )
            ),
        "true_source_equiv_with_diagnostic":
            int(
                8 * (
                    1
                    + num_trial_evals
                )
            ),
        "true64_seconds": 0.0,
        "anchor_true8_seconds":
            float(anchor_s),
        "neural64_seconds":
            float(neural_s),
        "guard_seconds":
            float(guard_s),
        "gradient_seconds":
            float(grad_s),
        "diagnostic_eval_seconds":
            0.0,
    }


# =====================================================================
# Custom paired 2-step C4 trajectory
# =====================================================================

def make_generator(
    sample_index,
    num_samples,
    seed,
):
    from cc.random_util import (
        get_generator,
    )

    g = get_generator(
        "determ-indiv",
        num_samples=num_samples,
        seed=seed,
    )

    g.set_done_samples(
        sample_index
    )

    return g


@th.no_grad()
def run_two_step_arm(
    arm,
    hint_norm_256,
    sample_index,
    num_samples,
    args,
    device,
    control_net,
    c4_model,
    diffusion,
    ctx,
    align_corners,
    mgno_model,
    background64,
    attempt_rows,
):
    """
    arm:
        C4-only
        C4+Exact
        C4+GRA
    """
    generator = make_generator(
        sample_index=sample_index,
        num_samples=num_samples,
        seed=args.seed,
    )

    hint = th.from_numpy(
        hint_norm_256[
            None, None
        ].astype(np.float32)
    ).to(device)

    # Same x_T for every arm.
    x = (
        generator.randn(
            1,
            1,
            256,
            256,
            device=device,
        )
        * C4_SIGMA_MAX
    )

    sigma_T = sigma_from_ts_index(
        C4_TS[0]
    )

    t0 = time.perf_counter()
    cuda_sync(device)

    # -------------------------------------------------------------
    # Consistency mapping #1.
    # -------------------------------------------------------------
    x0_first = cc_recon(
        x,
        sigma_T,
        hint,
        control_net,
        c4_model,
        diffusion,
    )

    cuda_sync(device)
    cc1_s = float(
        time.perf_counter() - t0
    )

    x0_first_norm = (
        x0_first.float()
        .cpu()
        .numpy()[0, 0]
        .astype(np.float32)
    )

    physics_info = {
        "J_before": None,
        "J_after": None,
        "rr_before": None,
        "rr_after": None,
        "relative_drop": None,
        "accepted": True,
        "accepted_step_mps": 0.0,
        "num_trial_evals": 0,
        "true_source_equiv_core": 0,
        "true_source_equiv_with_diagnostic": 0,
        "true64_seconds": 0.0,
        "anchor_true8_seconds": 0.0,
        "neural64_seconds": 0.0,
        "guard_seconds": 0.0,
        "gradient_seconds": 0.0,
        "diagnostic_eval_seconds": 0.0,
    }

    physics_s = 0.0

    # State immediately after the optional physics correction,
    # represented back in the C4 image-domain normalization.
    physics_norm_256 = x0_first_norm.copy()

    # -------------------------------------------------------------
    # Physics correction on clean manifold.
    # -------------------------------------------------------------
    if arm != "C4-only":
        speed480 = norm256_to_speed480(
            x0_first_norm,
            align_corners=align_corners,
        )

        cuda_sync(device)
        tp0 = time.perf_counter()

        if arm == "C4+Exact":
            corrected480, physics_info = (
                exact_physics_update(
                    speed480,
                    ctx,
                    args,
                    device,
                )
            )

        elif arm == "C4+GRA":
            corrected480, physics_info = (
                gra_physics_update(
                    speed480,
                    ctx,
                    args,
                    device,
                    mgno_model,
                    background64,
                    attempt_rows,
                )
            )

        else:
            raise ValueError(arm)

        cuda_sync(device)
        physics_s = float(
            time.perf_counter() - tp0
        )

        corrected_norm_256 = (
            speed480_to_norm256(
                corrected480,
                align_corners=align_corners,
            )
        )

        physics_norm_256 = (
            corrected_norm_256
            .astype(np.float32)
            .copy()
        )

        x0_for_sde = th.from_numpy(
            corrected_norm_256[
                None, None
            ].astype(np.float32)
        ).to(device)

    else:
        x0_for_sde = x0_first

    # -------------------------------------------------------------
    # Forward SDE / re-noise to ts=17.
    #
    # Important: because each arm uses a fresh determ-indiv generator with
    # identical sample_index/seed, this is the SAME noise realization.
    # -------------------------------------------------------------
    sigma_mid = sigma_from_ts_index(
        C4_TS[1]
    )

    noise_scale = float(
        np.sqrt(
            max(
                sigma_mid**2
                - C4_SIGMA_MIN**2,
                0.0,
            )
        )
    )

    x_mid = (
        x0_for_sde
        + generator.randn_like(
            x0_for_sde
        )
        * noise_scale
    )

    # -------------------------------------------------------------
    # Consistency mapping #2.
    # -------------------------------------------------------------
    t1 = time.perf_counter()
    cuda_sync(device)

    x0_second = cc_recon(
        x_mid,
        sigma_mid,
        hint,
        control_net,
        c4_model,
        diffusion,
    )

    cuda_sync(device)
    cc2_s = float(
        time.perf_counter() - t1
    )

    # ts=39 maps to sigma_min, so the final forward-SDE noise magnitude
    # sqrt(sigma_min^2 - sigma_min^2) is exactly zero.
    final_norm = (
        x0_second.float()
        .cpu()
        .numpy()[0, 0]
        .astype(np.float32)
    )

    return {
        "first_clean_norm":
            x0_first_norm,
        "physics_norm":
            physics_norm_256,
        "final_norm":
            final_norm,
        "physics":
            physics_info,
        "cc1_seconds":
            float(cc1_s),
        "physics_seconds":
            float(physics_s),
        "cc2_seconds":
            float(cc2_s),
        "core_seconds":
            float(
                cc1_s
                + physics_s
                + cc2_s
            ),
        "sigma_T":
            float(sigma_T),
        "sigma_mid":
            float(sigma_mid),
    }


# =====================================================================
# Official C4 audit
# =====================================================================

@th.no_grad()
def official_c4_sample(
    hint_norm_256,
    sample_index,
    num_samples,
    args,
    device,
    control_net,
    c4_model,
    diffusion,
):
    from cc.karras_diffusion import (
        control_sample,
    )

    hint = th.from_numpy(
        hint_norm_256[
            None, None
        ].astype(np.float32)
    ).to(device)

    generator = make_generator(
        sample_index=sample_index,
        num_samples=num_samples,
        seed=args.seed,
    )

    condition_args = {
        "hint": hint,
        "y_n": th.zeros_like(hint),
        "measurement_cond_fn": None,
    }

    sample, _ = control_sample(
        diffusion=diffusion,
        control_net=control_net,
        controlled_unet=c4_model,
        shape=(1, 1, 256, 256),
        steps=C4_STEPS,
        clip_denoised=True,
        progress=False,
        callback=None,
        model_kwargs={},
        device=device,
        sigma_min=C4_SIGMA_MIN,
        sigma_max=C4_SIGMA_MAX,
        rho=C4_RHO,
        sampler="multistep",
        generator=generator,
        ts=C4_TS,
        condition_args=condition_args,
    )

    return (
        sample.float()
        .cpu()
        .numpy()[0, 0]
        .astype(np.float32)
    )


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description=(
            "C6.5A: frozen C4 2-step conditional CM "
            "+ Exact/GRA physics trajectory integration."
        )
    )

    # C4 / CoSIGN.
    ap.add_argument(
        "--cosign_root",
        required=True,
        help=(
            "Directory containing the cc/ package, "
            "usually cosign_official."
        ),
    )

    ap.add_argument(
        "--backbone",
        required=True,
    )

    ap.add_argument(
        "--control",
        required=True,
    )

    ap.add_argument(
        "--hint_norm",
        required=True,
        help="C4 hint array, shape [N,256,256].",
    )

    ap.add_argument(
        "--gt_norm",
        required=True,
        help="GT normalized array, shape [N,256,256].",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260908,
    )

    # Physics.
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

    # Frozen GRA.
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

    # Split.
    ap.add_argument(
        "--start",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--end",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--id_offset",
        type=int,
        default=1,
        help=(
            "Array index = sample_id - id_offset; image-to-physics uses the frozen C6.2 transpose. "
            "For test_1 -> array[0], use 1."
        ),
    )

    ap.add_argument(
        "--split_label",
        default="dev_test1_10",
    )

    ap.add_argument(
        "--device",
        default="cuda:0",
    )

    ap.add_argument(
        "--output_dir",
        required=True,
    )

    ap.add_argument(
        "--c4_repro_tol",
        type=float,
        default=1e-6,
        help=(
            "RRMSE tolerance between custom C4-only "
            "trajectory and official control_sample."
        ),
    )

    ap.add_argument(
        "--bridge_interp_tol",
        type=float,
        default=5e-4,
        help=(
            "RRMSE tolerance between regenerated C4-only "
            "480 field and existing bridge candidate_480."
        ),
    )

    args = ap.parse_args()

    # cc imports must be available before create_c4_models().
    add_cosign_to_path(
        args.cosign_root
    )

    device = th.device(
        args.device
    )

    if device.type == "cuda":
        if not th.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )
        th.empty(
            1,
            device=device,
        )
        th.cuda.synchronize(device)

    out = Path(
        args.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Arrays.
    # ---------------------------------------------------------------
    hint_norm_all = np.load(
        args.hint_norm
    ).astype(np.float32)

    gt_norm_all = np.load(
        args.gt_norm
    ).astype(np.float32)

    if hint_norm_all.shape != gt_norm_all.shape:
        raise RuntimeError(
            "hint/gt shape mismatch: "
            f"{hint_norm_all.shape} vs "
            f"{gt_norm_all.shape}"
        )

    if (
        hint_norm_all.ndim != 3
        or
        hint_norm_all.shape[1:] != (256, 256)
    ):
        raise RuntimeError(
            "Expected hint/gt shape [N,256,256], got "
            f"{hint_norm_all.shape}"
        )

    n_total = len(
        hint_norm_all
    )

    # ---------------------------------------------------------------
    # C4.
    # ---------------------------------------------------------------
    (
        control_net,
        c4_model,
        diffusion,
    ) = create_c4_models(
        args.backbone,
        args.control,
        device,
    )

    # ---------------------------------------------------------------
    # MgNO/background.
    # ---------------------------------------------------------------
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
            "background64 must have 64 sources."
        )

    (
        mgno_model,
        mgno_channels,
        mgno_vcycles,
    ) = load_mgno(
        args.mgno_ckpt,
        args.device,
    )

    print("=" * 136)
    print(
        "C6.5A CONDITIONAL-CM + PHYSICS "
        "TRAJECTORY INTEGRATION"
    )
    print("=" * 136)

    print("split              =", args.split_label)
    print("samples            =", f"{args.start}..{args.end}")
    print("C4 seed            =", args.seed)
    print("C4 schedule        =", list(C4_TS))
    print("C4 NFE             =", len(C4_TS) - 1)
    print("MgNO config        =", f"C{mgno_channels}/V{mgno_vcycles}")
    print("GRA step           =", args.initial_step)
    print("GRA backtrack      =", args.backtrack_factor)
    print("GRA max backtracks =", args.max_backtracks)

    sample_rows = []
    physics_rows = []
    attempt_rows = []
    audit_rows = []
    retention_rows = []

    arms = [
        "Hint",
        "C4-only",
        "C4+Exact",
        "C4+GRA",
    ]

    # Global interpolation convention is frozen after the first sample.
    chosen_align_corners = None

    for sample_id in range(
        args.start,
        args.end + 1,
    ):
        array_idx = (
            sample_id
            - args.id_offset
        )

        if not (
            0 <= array_idx < n_total
        ):
            raise IndexError(
                f"sample {sample_id}: array_idx={array_idx} "
                f"outside [0,{n_total-1}]"
            )

        print()
        print("#" * 136)
        print(
            f"TEST {sample_id} "
            f"(array_idx={array_idx})"
        )
        print("#" * 136)

        hint_norm = (
            hint_norm_all[
                array_idx
            ]
        )

        gt_norm = (
            gt_norm_all[
                array_idx
            ]
        )

        gt_speed = norm_to_speed(
            gt_norm
        )

        hint_speed = norm_to_speed(
            hint_norm
        )

        ctx = load_bridge_sample(
            args.bridge_dir,
            sample_id,
            background_rec,
        )

        # -----------------------------------------------------------
        # Identity audit:
        # val20 array index <-> physical bridge sample.
        #
        # This protects against a silent off-by-k mapping error between
        # C4 arrays and CBS files.  These fields are evaluation/provenance
        # only and are NOT used by the physics acceptance rule.
        # -----------------------------------------------------------
        bridge_gt_256 = resize_2d_np(
            ctx["target_480"],
            (256, 256),
            align_corners=False,
        )

        # Prefer the exact 256 target stored by the bridge when present.
        bridge_path = (
            Path(args.bridge_dir)
            / f"test_{sample_id}.npz"
        )
        with np.load(
            bridge_path,
            allow_pickle=True,
        ) as _z_identity:
            if "target_256" in _z_identity.files:
                bridge_gt_256 = (
                    _z_identity["target_256"]
                    .astype(np.float32)
                )

            if "hint_256" in _z_identity.files:
                bridge_hint_256 = (
                    _z_identity["hint_256"]
                    .astype(np.float32)
                )
            else:
                bridge_hint_256 = None

            source_file_identity = (
                str(
                    np.asarray(
                        _z_identity["source_file"]
                    ).reshape(-1)[0]
                )
                if "source_file" in _z_identity.files
                else ""
            )

        # val20 arrays are in image coordinates; bridge target/hint
        # are in physical coordinates.  Apply the same frozen C6.2
        # transpose before doing identity checks.
        gt_speed_physics = (
            gt_speed
            .T
            .copy()
        )

        hint_speed_physics = (
            hint_speed
            .T
            .copy()
        )

        gt_identity_rr = rrmse(
            gt_speed_physics,
            bridge_gt_256,
        )

        gt_identity_max = float(
            np.max(
                np.abs(
                    gt_speed_physics
                    - bridge_gt_256
                )
            )
        )

        if bridge_hint_256 is not None:
            hint_identity_rr = rrmse(
                hint_speed_physics,
                bridge_hint_256,
            )

            hint_identity_max = float(
                np.max(
                    np.abs(
                        hint_speed_physics
                        - bridge_hint_256
                    )
                )
            )
        else:
            hint_identity_rr = float("nan")
            hint_identity_max = float("nan")

        identity_tol = 1e-6

        if gt_identity_rr > identity_tol:
            raise RuntimeError(
                f"test_{sample_id}: GT/bridge identity failed: "
                f"RRMSE={gt_identity_rr:.6e}, "
                f"maxabs={gt_identity_max:.6e}"
            )

        if (
            bridge_hint_256 is not None
            and hint_identity_rr > identity_tol
        ):
            raise RuntimeError(
                f"test_{sample_id}: HINT/bridge identity failed: "
                f"RRMSE={hint_identity_rr:.6e}, "
                f"maxabs={hint_identity_max:.6e}"
            )

        print(
            "identity audit     | "
            f"GT rr={gt_identity_rr:.3e} | "
            f"HINT rr={hint_identity_rr:.3e} | "
            f"source={Path(source_file_identity).name}"
        )

        # -----------------------------------------------------------
        # Audit custom C4-only trajectory against official sampler.
        # -----------------------------------------------------------
        official = official_c4_sample(
            hint_norm_256=hint_norm,
            sample_index=array_idx,
            num_samples=n_total,
            args=args,
            device=device,
            control_net=control_net,
            c4_model=c4_model,
            diffusion=diffusion,
        )

        # Temporary align=False only; C4-only 256 path does not depend
        # on interpolation, so this is irrelevant here.
        custom_c4 = run_two_step_arm(
            arm="C4-only",
            hint_norm_256=hint_norm,
            sample_index=array_idx,
            num_samples=n_total,
            args=args,
            device=device,
            control_net=control_net,
            c4_model=c4_model,
            diffusion=diffusion,
            ctx=ctx,
            align_corners=False,
            mgno_model=mgno_model,
            background64=background64,
            attempt_rows=[],
        )

        custom_final = (
            custom_c4["final_norm"]
        )

        c4_repro_rr = rrmse(
            custom_final,
            official,
        )

        c4_repro_max = float(
            np.max(
                np.abs(
                    custom_final
                    - official
                )
            )
        )

        if c4_repro_rr > args.c4_repro_tol:
            raise RuntimeError(
                f"test_{sample_id}: custom C4 trajectory "
                f"does not reproduce official C4: "
                f"RRMSE={c4_repro_rr:.6e}, "
                f"tol={args.c4_repro_tol:.6e}"
            )

        # -----------------------------------------------------------
        # Audit / freeze interpolation convention against C6.2 bridge.
        # -----------------------------------------------------------
        (
            local_ac,
            ac_results,
        ) = choose_bridge_alignment(
            official,
            ctx["candidate_480"],
        )

        if chosen_align_corners is None:
            chosen_align_corners = (
                local_ac
            )

        if local_ac != chosen_align_corners:
            raise RuntimeError(
                f"test_{sample_id}: interpolation convention "
                "changes across samples; bridge generation is "
                "not reproduced consistently."
            )

        bridge_rr = (
            ac_results[
                chosen_align_corners
            ]["rrmse"]
        )

        bridge_mae = (
            ac_results[
                chosen_align_corners
            ]["mae"]
        )

        print(
            "C4 audit           | "
            f"rrmse={c4_repro_rr:.3e} | "
            f"max={c4_repro_max:.3e}"
        )

        print(
            "bridge audit       | "
            f"align_corners={chosen_align_corners} | "
            f"rrmse={bridge_rr:.3e} | "
            f"mae={bridge_mae:.6f} m/s"
        )

        if bridge_rr > args.bridge_interp_tol:
            raise RuntimeError(
                f"test_{sample_id}: bridge interpolation "
                f"audit failed: RRMSE={bridge_rr:.6e} > "
                f"{args.bridge_interp_tol:.6e}"
            )

        audit_rows.append({
            "sample":
                sample_id,
            "array_idx":
                array_idx,
            "source_file":
                source_file_identity,
            "gt_identity_rrmse":
                gt_identity_rr,
            "gt_identity_maxabs":
                gt_identity_max,
            "hint_identity_rrmse":
                hint_identity_rr,
            "hint_identity_maxabs":
                hint_identity_max,
            "c4_custom_vs_official_rrmse":
                c4_repro_rr,
            "c4_custom_vs_official_maxabs":
                c4_repro_max,
            "align_corners":
                bool(
                    chosen_align_corners
                ),
            "bridge_rrmse":
                bridge_rr,
            "bridge_mae_mps":
                bridge_mae,
            "bridge_candidate_sha256":
                sha256_array(
                    ctx["candidate_480"]
                ),
        })

        # -----------------------------------------------------------
        # Baseline hint.
        # -----------------------------------------------------------
        hint_m = metric_dict(
            hint_speed,
            gt_speed,
        )

        # Measurement evaluation uses same audited interpolation.
        hint_480 = norm256_to_speed480(
            hint_norm,
            chosen_align_corners,
        )

        (
            hint_J,
            hint_rr,
        ), hint_eval_s = timed_call(
            lambda:
                measurement_objective_speed480(
                    hint_480,
                    ctx,
                    args,
                ),
            device,
        )

        sample_rows.append({
            "sample":
                sample_id,
            "array_idx":
                array_idx,
            "method":
                "Hint",
            **hint_m,
            "measurement_J":
                float(hint_J),
            "measurement_rr":
                float(hint_rr),
            "physics_drop_at_insert":
                0.0,
            "physics_accepted":
                True,
            "physics_step_mps":
                0.0,
            "physics_trial_evals":
                0,
            "true_source_equiv_core":
                0,
            "cc_nfe":
                0,
            "cc1_seconds":
                0.0,
            "physics_seconds":
                0.0,
            "cc2_seconds":
                0.0,
            "method_core_seconds":
                0.0,
            "final_eval_seconds":
                float(
                    hint_eval_s
                ),
        })

        # -----------------------------------------------------------
        # Run paired trajectory arms.
        # -----------------------------------------------------------
        trajectory_outputs = {}

        for arm in [
            "C4-only",
            "C4+Exact",
            "C4+GRA",
        ]:
            local_attempt_rows = []

            result = run_two_step_arm(
                arm=arm,
                hint_norm_256=hint_norm,
                sample_index=array_idx,
                num_samples=n_total,
                args=args,
                device=device,
                control_net=control_net,
                c4_model=c4_model,
                diffusion=diffusion,
                ctx=ctx,
                align_corners=chosen_align_corners,
                mgno_model=mgno_model,
                background64=background64,
                attempt_rows=local_attempt_rows,
            )

            # Attach sample id to attempts.
            for r in local_attempt_rows:
                attempt_rows.append(r)

            final_norm = (
                result["final_norm"]
            )

            final_speed = (
                norm_to_speed(
                    final_norm
                )
            )

            m = metric_dict(
                final_speed,
                gt_speed,
            )

            final_480 = (
                norm256_to_speed480(
                    final_norm,
                    chosen_align_corners,
                )
            )

            (
                final_J,
                final_rr,
            ), final_eval_s = timed_call(
                lambda:
                    measurement_objective_speed480(
                        final_480,
                        ctx,
                        args,
                    ),
                device,
            )

            pinfo = result[
                "physics"
            ]

            row = {
                "sample":
                    sample_id,
                "array_idx":
                    array_idx,
                "method":
                    arm,
                **m,
                "measurement_J":
                    float(final_J),
                "measurement_rr":
                    float(final_rr),
                "physics_drop_at_insert":
                    (
                        0.0
                        if pinfo[
                            "relative_drop"
                        ] is None
                        else
                        float(
                            pinfo[
                                "relative_drop"
                            ]
                        )
                    ),
                "physics_accepted":
                    bool(
                        pinfo[
                            "accepted"
                        ]
                    ),
                "physics_step_mps":
                    float(
                        pinfo[
                            "accepted_step_mps"
                        ]
                    ),
                "physics_trial_evals":
                    int(
                        pinfo[
                            "num_trial_evals"
                        ]
                    ),
                "true_source_equiv_core":
                    int(
                        pinfo[
                            "true_source_equiv_core"
                        ]
                    ),
                "cc_nfe":
                    2,
                "cc1_seconds":
                    float(
                        result[
                            "cc1_seconds"
                        ]
                    ),
                "physics_seconds":
                    float(
                        result[
                            "physics_seconds"
                        ]
                    ),
                "cc2_seconds":
                    float(
                        result[
                            "cc2_seconds"
                        ]
                    ),
                "method_core_seconds":
                    float(
                        result[
                            "core_seconds"
                        ]
                    ),
                "final_eval_seconds":
                    float(
                        final_eval_s
                    ),
            }

            sample_rows.append(
                row
            )

            trajectory_outputs[
                arm
            ] = {
                "first_clean_norm":
                    result[
                        "first_clean_norm"
                    ],
                "physics_norm":
                    result[
                        "physics_norm"
                    ],
                "final_norm":
                    final_norm,
            }

            # -------------------------------------------------------
            # C6.5B retention diagnostic:
            # first clean -> physics state -> CM2 final.
            #
            # No parameter selection and no acceptance-rule changes.
            # -------------------------------------------------------
            if arm != "C4-only":
                first_speed = norm_to_speed(
                    result["first_clean_norm"]
                )
                physics_speed = norm_to_speed(
                    result["physics_norm"]
                )

                first_m = metric_dict(
                    first_speed,
                    gt_speed,
                )
                physics_m = metric_dict(
                    physics_speed,
                    gt_speed,
                )

                J_before = float(
                    pinfo["J_before"]
                )
                J_after = float(
                    pinfo["J_after"]
                )

                physics_gain = (
                    J_before - J_after
                )
                final_gain_vs_pre = (
                    J_before - float(final_J)
                )

                if physics_gain > 1e-30:
                    retention_ratio = (
                        final_gain_vs_pre
                        / physics_gain
                    )
                    washout_fraction = (
                        float(final_J) - J_after
                    ) / physics_gain
                else:
                    retention_ratio = float("nan")
                    washout_fraction = float("nan")

                delta_phys = (
                    physics_speed
                    - first_speed
                ).astype(np.float64)

                delta_cm2 = (
                    final_speed
                    - physics_speed
                ).astype(np.float64)

                dp = delta_phys.ravel()
                dc = delta_cm2.ravel()

                dp2 = float(
                    np.dot(dp, dp)
                )
                dc2 = float(
                    np.dot(dc, dc)
                )

                if dp2 > 1e-30 and dc2 > 1e-30:
                    cm2_vs_phys_cos = float(
                        np.dot(dp, dc)
                        /
                        np.sqrt(dp2 * dc2)
                    )
                    cm2_projection_on_phys = float(
                        np.dot(dp, dc)
                        / dp2
                    )
                    directional_retention = float(
                        1.0
                        + cm2_projection_on_phys
                    )
                    update_rms_ratio = float(
                        np.sqrt(dc2 / dp2)
                    )
                else:
                    cm2_vs_phys_cos = float("nan")
                    cm2_projection_on_phys = float("nan")
                    directional_retention = float("nan")
                    update_rms_ratio = float("nan")

                retention_rows.append({
                    "sample":
                        int(sample_id),
                    "method":
                        arm,
                    "accepted":
                        bool(
                            pinfo["accepted"]
                        ),
                    "accepted_step_mps":
                        float(
                            pinfo[
                                "accepted_step_mps"
                            ]
                        ),
                    "J_before_physics":
                        J_before,
                    "J_after_physics":
                        J_after,
                    "J_after_cm2":
                        float(final_J),
                    "physics_J_gain":
                        float(physics_gain),
                    "final_J_gain_vs_pre":
                        float(final_gain_vs_pre),
                    "retention_ratio":
                        float(retention_ratio),
                    "washout_fraction":
                        float(washout_fraction),
                    "cm2_vs_physics_cos":
                        float(cm2_vs_phys_cos),
                    "cm2_projection_on_physics":
                        float(
                            cm2_projection_on_phys
                        ),
                    "directional_retention":
                        float(
                            directional_retention
                        ),
                    "cm2_to_physics_update_rms_ratio":
                        float(update_rms_ratio),
                    "first_clean_mse":
                        float(first_m["mse"]),
                    "physics_state_mse":
                        float(physics_m["mse"]),
                    "final_mse":
                        float(m["mse"]),
                    "physics_mse_gain":
                        float(
                            first_m["mse"]
                            - physics_m["mse"]
                        ),
                    "final_mse_gain_vs_first":
                        float(
                            first_m["mse"]
                            - m["mse"]
                        ),
                })

            if arm != "C4-only":
                physics_rows.append({
                    "sample":
                        sample_id,
                    "method":
                        arm,
                    **pinfo,
                })

            print(
                f"{arm:10s} | "
                f"MSE={m['mse']:.3f} | "
                f"PSNR={m['psnr']:.3f} | "
                f"SSIM={m['ssim']:.5f} | "
                f"J={final_J:.6e} | "
                f"phys_drop="
                f"{100*row['physics_drop_at_insert']:+.2f}% | "
                f"step={row['physics_step_mps']:.4f}"
            )

        # -----------------------------------------------------------
        # Save per-sample trajectory cache.
        # -----------------------------------------------------------
        np.savez_compressed(
            out /
            f"test_{sample_id}_trajectory.npz",

            hint_norm=
                hint_norm,

            gt_norm=
                gt_norm,

            official_c4_final_norm=
                official,

            c4_first_clean_norm=
                trajectory_outputs[
                    "C4-only"
                ][
                    "first_clean_norm"
                ],

            c4_final_norm=
                trajectory_outputs[
                    "C4-only"
                ][
                    "final_norm"
                ],

            exact_physics_norm=
                trajectory_outputs[
                    "C4+Exact"
                ][
                    "physics_norm"
                ],

            gra_physics_norm=
                trajectory_outputs[
                    "C4+GRA"
                ][
                    "physics_norm"
                ],

            exact_final_norm=
                trajectory_outputs[
                    "C4+Exact"
                ][
                    "final_norm"
                ],

            gra_final_norm=
                trajectory_outputs[
                    "C4+GRA"
                ][
                    "final_norm"
                ],

            bridge_candidate_480=
                ctx["candidate_480"],

            align_corners=
                np.asarray(
                    [
                        int(
                            chosen_align_corners
                        )
                    ],
                    dtype=np.int64,
                ),

            c4_ts=
                np.asarray(
                    C4_TS,
                    dtype=np.int64,
                ),
        )

    # =================================================================
    # Aggregate.
    # =================================================================
    summary_rows = []
    summary = {}

    for method in arms:
        sub = [
            r
            for r in sample_rows
            if r["method"] == method
        ]

        ms = {
            "num_samples":
                len(sub),
        }

        for key in [
            "mse",
            "mae",
            "psnr",
            "ssim",
            "measurement_J",
            "measurement_rr",
            "method_core_seconds",
            "true_source_equiv_core",
        ]:
            ms[key] = stat([
                r[key]
                for r in sub
            ])

        if method == "Hint":
            ms[
                "mse_wins_vs_c4"
            ] = None
            ms[
                "ssim_wins_vs_c4"
            ] = None
        else:
            c4_map = {
                r["sample"]: r
                for r in sample_rows
                if r["method"]
                == "C4-only"
            }

            ms[
                "mse_wins_vs_c4"
            ] = int(
                sum(
                    r["mse"]
                    <
                    c4_map[
                        r["sample"]
                    ]["mse"]
                    for r in sub
                )
            )

            ms[
                "ssim_wins_vs_c4"
            ] = int(
                sum(
                    r["ssim"]
                    >
                    c4_map[
                        r["sample"]
                    ]["ssim"]
                    for r in sub
                )
            )

        summary[
            method
        ] = ms

        summary_rows.append({
            "method":
                method,
            "num_samples":
                len(sub),
            "mse_mean":
                ms["mse"]["mean"],
            "mae_mean":
                ms["mae"]["mean"],
            "psnr_mean":
                ms["psnr"]["mean"],
            "ssim_mean":
                ms["ssim"]["mean"],
            "measurement_J_mean":
                ms[
                    "measurement_J"
                ]["mean"],
            "measurement_rr_mean":
                ms[
                    "measurement_rr"
                ]["mean"],
            "method_core_seconds_mean":
                ms[
                    "method_core_seconds"
                ]["mean"],
            "true_source_equiv_mean":
                ms[
                    "true_source_equiv_core"
                ]["mean"],
            "mse_wins_vs_c4":
                ms[
                    "mse_wins_vs_c4"
                ],
            "ssim_wins_vs_c4":
                ms[
                    "ssim_wins_vs_c4"
                ],
        })

    # Physics-only aggregate.
    physics_summary = {}

    for method in [
        "C4+Exact",
        "C4+GRA",
    ]:
        sub = [
            r
            for r in physics_rows
            if r["method"] == method
        ]

        physics_summary[
            method
        ] = {
            "accepted_count":
                int(
                    sum(
                        bool(
                            r[
                                "accepted"
                            ]
                        )
                        for r in sub
                    )
                ),
            "insert_relative_drop":
                stat([
                    r[
                        "relative_drop"
                    ]
                    for r in sub
                ]),
            "accepted_step_mps":
                stat([
                    r[
                        "accepted_step_mps"
                    ]
                    for r in sub
                ]),
            "true_source_equiv_core":
                stat([
                    r[
                        "true_source_equiv_core"
                    ]
                    for r in sub
                ]),
        }

    # C6.5A diagnostic gate:
    # This is intentionally not an optimization/tuning gate.
    #
    # Strong integration pass:
    # - custom C4 reproduction passed by construction;
    # - GRA physics accepts/descends on >=80% samples;
    # - final C4+GRA mean measurement J < C4-only;
    # - final C4+GRA mean MSE <= C4-only OR has >=50% MSE wins.
    n = (
        args.end
        - args.start
        + 1
    )

    c4s = summary[
        "C4-only"
    ]

    gras = summary[
        "C4+GRA"
    ]

    gra_phys = physics_summary[
        "C4+GRA"
    ]

    cond_phys = (
        gra_phys[
            "accepted_count"
        ]
        >=
        math.ceil(
            0.8 * n
        )
    )

    cond_measurement = (
        gras[
            "measurement_J"
        ]["mean"]
        <
        c4s[
            "measurement_J"
        ]["mean"]
    )

    cond_image = (
        gras["mse"]["mean"]
        <=
        c4s["mse"]["mean"]
        or
        gras[
            "mse_wins_vs_c4"
        ]
        >=
        math.ceil(
            0.5 * n
        )
    )

    if (
        cond_phys
        and
        cond_measurement
        and
        cond_image
    ):
        gate = (
            "STRONG_INTEGRATION_PASS"
        )
    elif (
        cond_phys
        and
        cond_measurement
    ):
        gate = (
            "PHYSICS_PASS_IMAGE_TRADEOFF"
        )
    else:
        gate = (
            "INTEGRATION_FAIL"
        )

    per_sample_csv = (
        out /
        "per_sample_metrics.csv"
    )

    summary_csv = (
        out /
        "method_summary.csv"
    )

    physics_csv = (
        out /
        "physics_insert_metrics.csv"
    )

    attempts_csv = (
        out /
        "gra_backtracking_attempts.csv"
    )

    audit_csv = (
        out /
        "trajectory_audit.csv"
    )

    retention_csv = (
        out /
        "physics_retention_metrics.csv"
    )

    write_csv(
        per_sample_csv,
        sample_rows,
    )

    write_csv(
        summary_csv,
        summary_rows,
    )

    write_csv(
        physics_csv,
        physics_rows,
    )

    write_csv(
        attempts_csv,
        attempt_rows,
    )

    write_csv(
        audit_csv,
        audit_rows,
    )

    write_csv(
        retention_csv,
        retention_rows,
    )

    # Retention-only aggregate on accepted physics updates.
    retention_summary = {}

    for method in [
        "C4+Exact",
        "C4+GRA",
    ]:
        sub_all = [
            r
            for r in retention_rows
            if r["method"] == method
        ]

        sub = [
            r
            for r in sub_all
            if (
                r["accepted"]
                and
                np.isfinite(
                    r["retention_ratio"]
                )
            )
        ]

        def _finite_stat(key):
            vals = [
                r[key]
                for r in sub
                if np.isfinite(r[key])
            ]
            if not vals:
                return None
            return stat(vals)

        retention_summary[method] = {
            "num_samples":
                len(sub_all),
            "accepted_count":
                int(
                    sum(
                        bool(r["accepted"])
                        for r in sub_all
                    )
                ),
            "final_measurement_win_count":
                int(
                    sum(
                        r["J_after_cm2"]
                        <
                        r["J_before_physics"]
                        for r in sub_all
                    )
                ),
            "retention_ratio":
                _finite_stat(
                    "retention_ratio"
                ),
            "washout_fraction":
                _finite_stat(
                    "washout_fraction"
                ),
            "cm2_vs_physics_cos":
                _finite_stat(
                    "cm2_vs_physics_cos"
                ),
            "directional_retention":
                _finite_stat(
                    "directional_retention"
                ),
            "cm2_to_physics_update_rms_ratio":
                _finite_stat(
                    "cm2_to_physics_update_rms_ratio"
                ),
        }

    summary_json = {
        "experiment":
            "C6.5B Physics-Retention Audit on Two-step Conditional CM",

        "split":
            args.split_label,

        "sample_ids": [
            args.start,
            args.end,
        ],

        "frozen_c4": {
            "seed":
                args.seed,
            "sampler":
                "multistep",
            "steps":
                C4_STEPS,
            "ts":
                list(C4_TS),
            "sigma_min":
                C4_SIGMA_MIN,
            "sigma_max":
                C4_SIGMA_MAX,
            "rho":
                C4_RHO,
            "nfe":
                len(C4_TS) - 1,
        },

        "frozen_gra": {
            "initial_step_mps":
                args.initial_step,
            "backtrack_factor":
                args.backtrack_factor,
            "max_backtracks":
                args.max_backtracks,
            "acceptance_rule":
                "J_trial < J_current",
            "gt_used_for_acceptance":
                False,
            "cosine_used_for_acceptance":
                False,
        },

        "bridge_alignment": {
            "align_corners":
                bool(
                    chosen_align_corners
                ),
            "selection_rule":
                (
                    "choose the PyTorch bilinear convention "
                    "that reproduces the pre-existing frozen "
                    "C6.2 bridge candidate_480"
                ),
        },

        "methods":
            summary,

        "physics_insert":
            physics_summary,

        "physics_retention":
            retention_summary,

        "gate":
            gate,

        "files": {
            "per_sample_metrics":
                str(per_sample_csv),
            "method_summary":
                str(summary_csv),
            "physics_insert_metrics":
                str(physics_csv),
            "gra_backtracking_attempts":
                str(attempts_csv),
            "trajectory_audit":
                str(audit_csv),
            "physics_retention_metrics":
                str(retention_csv),
        },
    }

    summary_path = (
        out /
        "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary_json,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 136)
    print(
        "C6.5A FINAL SUMMARY"
    )
    print("=" * 136)

    print(
        f"{'method':>12s} "
        f"{'MSE':>11s} "
        f"{'MAE':>9s} "
        f"{'PSNR':>9s} "
        f"{'SSIM':>9s} "
        f"{'meas_J':>12s} "
        f"{'MSEwin':>8s} "
        f"{'src-eq':>8s} "
        f"{'time':>9s}"
    )

    for r in summary_rows:
        mw = r[
            "mse_wins_vs_c4"
        ]

        mw_text = (
            "-"
            if mw is None
            else
            f"{mw}/{n}"
        )

        print(
            f"{r['method']:>12s} "
            f"{r['mse_mean']:11.4f} "
            f"{r['mae_mean']:9.4f} "
            f"{r['psnr_mean']:9.4f} "
            f"{r['ssim_mean']:9.5f} "
            f"{r['measurement_J_mean']:12.5e} "
            f"{mw_text:>8s} "
            f"{r['true_source_equiv_mean']:8.2f} "
            f"{r['method_core_seconds_mean']:9.3f}"
        )

    print()
    print("-" * 136)
    print("C6.5B PHYSICS-RETENTION SUMMARY")
    print("-" * 136)

    for _method in [
        "C4+Exact",
        "C4+GRA",
    ]:
        _s = retention_summary[_method]
        _rr = _s["retention_ratio"]
        _cos = _s["cm2_vs_physics_cos"]

        print(
            f"{_method:10s} | "
            f"accepted={_s['accepted_count']}/{_s['num_samples']} | "
            f"final-J-win={_s['final_measurement_win_count']}/{_s['num_samples']} | "
            f"retention_mean="
            + (
                "nan"
                if _rr is None
                else f"{_rr['mean']:+.4f}"
            )
            + " | cm2/phys cos="
            + (
                "nan"
                if _cos is None
                else f"{_cos['mean']:+.4f}"
            )
        )

    print()
    print(
        "Exact physics accepted =",
        f"{physics_summary['C4+Exact']['accepted_count']}/{n}",
    )

    print(
        "GRA physics accepted   =",
        f"{physics_summary['C4+GRA']['accepted_count']}/{n}",
    )

    print(
        "GRA insert mean drop   =",
        f"{100*physics_summary['C4+GRA']['insert_relative_drop']['mean']:+.3f}%",
    )

    print(
        "C6.5A GATE             =",
        gate,
    )

    print(
        "summary                =",
        summary_path,
    )


if __name__ == "__main__":
    main()
