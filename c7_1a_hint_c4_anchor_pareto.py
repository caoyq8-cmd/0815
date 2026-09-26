
#!/usr/bin/env python3
"""
C6.6 Terminal Guarded Physics Correction

Frozen end-to-end protocol:
    hint
      -> frozen 2-step conditional CM (C4)
      -> final clean reconstruction
      -> terminal Exact-CBS or GRA-ARSS correction
      -> output

Key difference from C6.5:
    There is NO consistency mapping after the physics correction.

The GRA rule is unchanged from the sealed C6.4B protocol:
    initial RMS step = 1.5 m/s
    backtrack factor = 0.5
    max backtracks = 5
    accept iff true 8-source measurement J_trial < J_current

No GT/MSE/SSIM information enters the physics acceptance rule.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch as th

import c6_5b_physics_retention_audit as B

from c5_audit_mgno_unseen64_sources import load_mgno


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="raise",
        )
        w.writeheader()
        w.writerows(rows)


def safe_mean(vals):
    x = np.asarray(vals, dtype=np.float64)
    return float(x.mean())


def main():
    ap = argparse.ArgumentParser(
        description=(
            "C6.6 frozen terminal Exact/GRA physics correction "
            "after the final 2-step conditional-CM reconstruction."
        )
    )

    # C4.
    ap.add_argument("--cosign_root", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--hint_norm", required=True)
    ap.add_argument("--gt_norm", required=True)
    ap.add_argument("--seed", type=int, default=20260908)

    # Physics.
    ap.add_argument("--bridge_dir", required=True)
    ap.add_argument("--background64_npz", required=True)
    ap.add_argument("--mgno_ckpt", required=True)

    ap.add_argument("--speed_mean", type=float, default=1488.39)
    ap.add_argument("--speed_std", type=float, default=27.53)
    ap.add_argument("--wave_scale", type=float, default=3.72290883e-02)
    ap.add_argument("--chunk_size", type=int, default=8)

    # Frozen GRA.
    ap.add_argument("--initial_step", type=float, default=1.5)
    ap.add_argument("--backtrack_factor", type=float, default=0.5)
    ap.add_argument("--max_backtracks", type=int, default=5)

    # Split.
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=10)
    ap.add_argument(
        "--id_offset",
        type=int,
        default=1,
        help="array_idx = sample_id - id_offset",
    )
    ap.add_argument(
        "--split_label",
        default="dev_test1_10",
    )

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output_dir", required=True)

    # Audits.
    ap.add_argument("--c4_repro_tol", type=float, default=1e-6)
    ap.add_argument("--bridge_interp_tol", type=float, default=5e-4)

    args = ap.parse_args()

    B.add_cosign_to_path(args.cosign_root)

    device = th.device(args.device)
    if device.type == "cuda":
        if not th.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        th.empty(1, device=device)
        th.cuda.synchronize(device)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Data.
    # -----------------------------------------------------------------
    hint_all = np.load(args.hint_norm).astype(np.float32)
    gt_all = np.load(args.gt_norm).astype(np.float32)

    if hint_all.shape != gt_all.shape:
        raise RuntimeError(
            f"hint/gt shape mismatch: {hint_all.shape} vs {gt_all.shape}"
        )

    if (
        hint_all.ndim != 3
        or hint_all.shape[1:] != (256, 256)
    ):
        raise RuntimeError(
            f"Expected [N,256,256], got {hint_all.shape}"
        )

    n_total = len(hint_all)

    # -----------------------------------------------------------------
    # Frozen C4.
    # -----------------------------------------------------------------
    (
        control_net,
        c4_model,
        diffusion,
    ) = B.create_c4_models(
        args.backbone,
        args.control,
        device,
    )

    # -----------------------------------------------------------------
    # Frozen MgNO / homogeneous 64-source basis.
    # -----------------------------------------------------------------
    bz = np.load(
        args.background64_npz,
        allow_pickle=True,
    )
    background64 = bz["background64"].astype(np.complex64)
    background_rec = bz["rec_indices"].astype(np.int64)

    if background64.shape[0] != 64:
        raise RuntimeError(
            f"Expected 64 backgrounds, got {background64.shape}"
        )

    (
        mgno_model,
        mgno_channels,
        mgno_vcycles,
    ) = load_mgno(
        args.mgno_ckpt,
        args.device,
    )

    print("=" * 142)
    print("C7.1A HINT-C4 ANCHOR PARETO SCAN")
    print("=" * 142)
    print("split              =", args.split_label)
    print("samples            =", f"{args.start}..{args.end}")
    print("C4 seed            =", args.seed)
    print("C4 schedule        =", list(B.C4_TS))
    print("C4 NFE             =", len(B.C4_TS) - 1)
    print("MgNO config        =", f"C{mgno_channels}/V{mgno_vcycles}")
    print("GRA step           =", args.initial_step)
    print("GRA backtrack      =", args.backtrack_factor)
    print("GRA max backtracks =", args.max_backtracks)
    print("post-physics CM    = NONE")
    print()

    rows = []
    anchor_rows = []
    audit_rows = []
    physics_rows = []
    attempt_rows = []

    chosen_align_corners = None

    for sample_id in range(args.start, args.end + 1):
        array_idx = sample_id - args.id_offset

        if not (0 <= array_idx < n_total):
            raise RuntimeError(
                f"sample {sample_id}: array_idx={array_idx} out of range"
            )

        print("#" * 142)
        print(f"TEST {sample_id} (array_idx={array_idx})")
        print("#" * 142)

        hint_norm = hint_all[array_idx]
        gt_norm = gt_all[array_idx]

        hint_speed = B.norm_to_speed(hint_norm)
        gt_speed = B.norm_to_speed(gt_norm)

        ctx = B.load_bridge_sample(
            args.bridge_dir,
            sample_id,
            background_rec,
        )

        # -------------------------------------------------------------
        # Identity audit.
        # bridge arrays are in physical coordinates; C4 arrays are image
        # coordinates, hence the frozen C6.2 transpose.
        # -------------------------------------------------------------
        bridge_path = (
            Path(args.bridge_dir)
            / f"test_{sample_id}.npz"
        )

        with np.load(
            bridge_path,
            allow_pickle=True,
        ) as z:
            bridge_gt = z["target_256"].astype(np.float32)
            bridge_hint = z["hint_256"].astype(np.float32)

            source_file = (
                str(
                    np.asarray(
                        z["source_file"]
                    ).reshape(-1)[0]
                )
                if "source_file" in z.files
                else ""
            )

        gt_rr = B.rrmse(
            gt_speed.T.copy(),
            bridge_gt,
        )
        hint_rr = B.rrmse(
            hint_speed.T.copy(),
            bridge_hint,
        )

        if gt_rr > 1e-6:
            raise RuntimeError(
                f"test_{sample_id}: GT identity failed {gt_rr:.6e}"
            )

        if hint_rr > 1e-6:
            raise RuntimeError(
                f"test_{sample_id}: HINT identity failed {hint_rr:.6e}"
            )

        # -------------------------------------------------------------
        # Official vs decomposed C4 audit.
        # -------------------------------------------------------------
        official = B.official_c4_sample(
            hint_norm_256=hint_norm,
            sample_index=array_idx,
            num_samples=n_total,
            args=args,
            device=device,
            control_net=control_net,
            c4_model=c4_model,
            diffusion=diffusion,
        )

        c4 = B.run_two_step_arm(
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

        c4_final_norm = c4["final_norm"]

        c4_repro_rr = B.rrmse(
            c4_final_norm,
            official,
        )
        c4_repro_max = float(
            np.max(
                np.abs(
                    c4_final_norm
                    - official
                )
            )
        )

        if c4_repro_rr > args.c4_repro_tol:
            raise RuntimeError(
                f"test_{sample_id}: C4 reproduction failed "
                f"{c4_repro_rr:.6e}"
            )

        (
            local_ac,
            ac_results,
        ) = B.choose_bridge_alignment(
            official,
            ctx["candidate_480"],
        )

        if chosen_align_corners is None:
            chosen_align_corners = local_ac

        if local_ac != chosen_align_corners:
            raise RuntimeError(
                f"test_{sample_id}: interpolation convention changed"
            )

        bridge_rr = ac_results[
            chosen_align_corners
        ]["rrmse"]
        bridge_mae = ac_results[
            chosen_align_corners
        ]["mae"]

        if bridge_rr > args.bridge_interp_tol:
            raise RuntimeError(
                f"test_{sample_id}: bridge audit failed "
                f"{bridge_rr:.6e}"
            )

        print(
            "audit              | "
            f"GT={gt_rr:.2e} HINT={hint_rr:.2e} | "
            f"C4={c4_repro_rr:.2e} | "
            f"bridge={bridge_rr:.2e} | "
            f"align_corners={chosen_align_corners}"
        )

        audit_rows.append({
            "sample": sample_id,
            "array_idx": array_idx,
            "source_file": source_file,
            "gt_identity_rrmse": gt_rr,
            "hint_identity_rrmse": hint_rr,
            "c4_custom_vs_official_rrmse": c4_repro_rr,
            "c4_custom_vs_official_maxabs": c4_repro_max,
            "align_corners": bool(chosen_align_corners),
            "bridge_rrmse": bridge_rr,
            "bridge_mae_mps": bridge_mae,
        })

        # -------------------------------------------------------------
        # Hint reference.
        # -------------------------------------------------------------
        hint_m = B.metric_dict(
            hint_speed,
            gt_speed,
        )

        hint_480 = B.norm256_to_speed480(
            hint_norm,
            chosen_align_corners,
        )

        (
            hint_J,
            hint_meas_rr,
        ), hint_eval_s = B.timed_call(
            lambda:
                B.measurement_objective_speed480(
                    hint_480,
                    ctx,
                    args,
                ),
            device,
        )

        rows.append({
            "sample": sample_id,
            "method": "Hint",
            **hint_m,
            "measurement_J": float(hint_J),
            "measurement_rr": float(hint_meas_rr),
            "physics_accepted": True,
            "physics_drop": 0.0,
            "physics_step_mps": 0.0,
            "physics_trial_evals": 0,
            "true_source_equiv_core": 0,
            "cc_nfe": 0,
            "cc_seconds": 0.0,
            "physics_seconds": 0.0,
            "method_core_seconds": 0.0,
            "evaluation_seconds": float(hint_eval_s),
        })

        # -------------------------------------------------------------
        # Frozen final C4 state.
        # -------------------------------------------------------------
        c4_speed = B.norm_to_speed(
            c4_final_norm
        )

        c4_m = B.metric_dict(
            c4_speed,
            gt_speed,
        )

        c4_480 = B.norm256_to_speed480(
            c4_final_norm,
            chosen_align_corners,
        )

        (
            c4_J,
            c4_meas_rr,
        ), c4_eval_s = B.timed_call(
            lambda:
                B.measurement_objective_speed480(
                    c4_480,
                    ctx,
                    args,
                ),
            device,
        )

        c4_seconds = float(
            c4["cc1_seconds"]
            + c4["cc2_seconds"]
        )

        rows.append({
            "sample": sample_id,
            "method": "C4-only",
            **c4_m,
            "measurement_J": float(c4_J),
            "measurement_rr": float(c4_meas_rr),
            "physics_accepted": True,
            "physics_drop": 0.0,
            "physics_step_mps": 0.0,
            "physics_trial_evals": 0,
            "true_source_equiv_core": 0,
            "cc_nfe": 2,
            "cc_seconds": c4_seconds,
            "physics_seconds": 0.0,
            "method_core_seconds": c4_seconds,
            "evaluation_seconds": float(c4_eval_s),
        })

        # -------------------------------------------------------------
        # C7.1A Hint-C4 anchor Pareto scan.
        #
        # DEV-ONLY analysis:
        #   x_lambda = (1-lambda) * x_C4 + lambda * x_Hint
        #
        # GT metrics are recorded only for development analysis.
        # They are NOT used in any physics acceptance rule.
        # -------------------------------------------------------------
        print()
        print("C7.1A HINT-C4 ANCHOR PARETO")

        anchor_lambdas = [
            0.0,
            0.25,
            0.50,
            0.75,
            1.0,
        ]

        for anchor_lambda in anchor_lambdas:
            lam = float(anchor_lambda)

            anchor_norm = (
                (1.0 - lam) * c4_final_norm
                + lam * hint_norm
            ).astype(np.float32)

            anchor_speed = B.norm_to_speed(
                anchor_norm
            )

            # DEV-only GT metrics.
            anchor_gt_m = B.metric_dict(
                anchor_speed,
                gt_speed,
            )

            # Non-GT structural references.
            anchor_hint_m = B.metric_dict(
                anchor_speed,
                hint_speed,
            )

            anchor_c4_m = B.metric_dict(
                anchor_speed,
                c4_speed,
            )

            anchor_480 = B.norm256_to_speed480(
                anchor_norm,
                chosen_align_corners,
            )

            (
                anchor_J,
                anchor_rr,
            ), anchor_eval_s = B.timed_call(
                lambda anchor_480=anchor_480:
                    B.measurement_objective_speed480(
                        anchor_480,
                        ctx,
                        args,
                    ),
                device,
            )

            row_anchor = {
                "sample": int(sample_id),
                "array_idx": int(array_idx),
                "lambda": lam,

                # DEV-only GT evaluation.
                "mse": float(
                    anchor_gt_m["mse"]
                ),
                "mae": float(
                    anchor_gt_m["mae"]
                ),
                "psnr": float(
                    anchor_gt_m["psnr"]
                ),
                "ssim": float(
                    anchor_gt_m["ssim"]
                ),

                # True physics.
                "measurement_J": float(
                    anchor_J
                ),
                "measurement_rr": float(
                    anchor_rr
                ),

                # Relative physics changes.
                "J_drop_vs_C4": float(
                    B.relative_drop(
                        c4_J,
                        anchor_J,
                    )
                ),
                "J_drop_vs_Hint": float(
                    B.relative_drop(
                        hint_J,
                        anchor_J,
                    )
                ),

                # Non-GT structural proxies.
                "ssim_to_hint": float(
                    anchor_hint_m["ssim"]
                ),
                "mae_to_hint": float(
                    anchor_hint_m["mae"]
                ),
                "ssim_to_c4": float(
                    anchor_c4_m["ssim"]
                ),
                "mae_to_c4": float(
                    anchor_c4_m["mae"]
                ),

                "evaluation_seconds": float(
                    anchor_eval_s
                ),
            }

            anchor_rows.append(
                row_anchor
            )

            print(
                f"lambda={lam:4.2f} | "
                f"J={anchor_J:.6e} | "
                f"MSE={anchor_gt_m['mse']:.3f} | "
                f"MAE={anchor_gt_m['mae']:.4f} | "
                f"SSIM={anchor_gt_m['ssim']:.5f} | "
                f"SSIM->Hint={anchor_hint_m['ssim']:.5f}"
            )

        print()

        # -------------------------------------------------------------
        # Terminal Exact and terminal GRA.
        # Both start from EXACTLY the same final C4 state.
        # -------------------------------------------------------------
        terminal_outputs = {}

        for method in [
            "C4+TExact",
            "C4+TGRA",
        ]:
            local_attempts = []

            B.cuda_sync(device)
            tp0 = time.perf_counter()

            if method == "C4+TExact":
                corrected480, pinfo = (
                    B.exact_physics_update(
                        c4_480,
                        ctx,
                        args,
                        device,
                    )
                )
            else:
                corrected480, pinfo = (
                    B.gra_physics_update(
                        c4_480,
                        ctx,
                        args,
                        device,
                        mgno_model,
                        background64,
                        local_attempts,
                    )
                )

            B.cuda_sync(device)
            physics_s = float(
                time.perf_counter() - tp0
            )

            for ar in local_attempts:
                ar["method"] = method
                attempt_rows.append(ar)

            corrected_norm = (
                B.speed480_to_norm256(
                    corrected480,
                    chosen_align_corners,
                )
            )

            corrected_speed = B.norm_to_speed(
                corrected_norm
            )

            mm = B.metric_dict(
                corrected_speed,
                gt_speed,
            )

            # J_after is the terminal measurement objective.
            final_J = float(
                pinfo["J_after"]
            )
            final_rr = float(
                pinfo["rr_after"]
            )

            # Independent audit of the final terminal state.
            (
                J_check,
                rr_check,
            ), final_eval_s = B.timed_call(
                lambda corrected480=corrected480:
                    B.measurement_objective_speed480(
                        corrected480,
                        ctx,
                        args,
                    ),
                device,
            )

            J_check_rel = abs(
                float(J_check) - final_J
            ) / max(
                abs(final_J),
                1e-30,
            )

            if J_check_rel > 1e-6:
                raise RuntimeError(
                    f"test_{sample_id} {method}: "
                    f"terminal J audit mismatch "
                    f"{J_check_rel:.6e}"
                )

            drop_vs_c4 = B.relative_drop(
                c4_J,
                final_J,
            )

            row = {
                "sample": sample_id,
                "method": method,
                **mm,
                "measurement_J": final_J,
                "measurement_rr": final_rr,
                "physics_accepted": bool(
                    pinfo["accepted"]
                ),
                "physics_drop": float(
                    drop_vs_c4
                ),
                "physics_step_mps": float(
                    pinfo["accepted_step_mps"]
                ),
                "physics_trial_evals": int(
                    pinfo["num_trial_evals"]
                ),
                "true_source_equiv_core": int(
                    pinfo["true_source_equiv_core"]
                ),
                "cc_nfe": 2,
                "cc_seconds": c4_seconds,
                "physics_seconds": physics_s,
                "method_core_seconds": float(
                    c4_seconds + physics_s
                ),
                "evaluation_seconds": float(
                    final_eval_s
                ),
                "terminal_J_audit_relerr": float(
                    J_check_rel
                ),
                "MSE_gain_vs_C4": float(
                    (
                        c4_m["mse"]
                        - mm["mse"]
                    )
                    / max(
                        c4_m["mse"],
                        1e-30,
                    )
                ),
                "MAE_gain_vs_C4": float(
                    (
                        c4_m["mae"]
                        - mm["mae"]
                    )
                    / max(
                        c4_m["mae"],
                        1e-30,
                    )
                ),
            }

            rows.append(row)

            physics_rows.append({
                "sample": sample_id,
                "method": method,
                "J_C4": float(c4_J),
                "J_terminal": final_J,
                "relative_drop_vs_C4": float(
                    drop_vs_c4
                ),
                "accepted": bool(
                    pinfo["accepted"]
                ),
                "accepted_step_mps": float(
                    pinfo["accepted_step_mps"]
                ),
                "num_trial_evals": int(
                    pinfo["num_trial_evals"]
                ),
                "true_source_equiv_core": int(
                    pinfo["true_source_equiv_core"]
                ),
                "MSE_C4": float(
                    c4_m["mse"]
                ),
                "MSE_terminal": float(
                    mm["mse"]
                ),
                "MSE_gain_vs_C4": float(
                    row["MSE_gain_vs_C4"]
                ),
                "SSIM_C4": float(
                    c4_m["ssim"]
                ),
                "SSIM_terminal": float(
                    mm["ssim"]
                ),
            })

            terminal_outputs[method] = {
                "corrected_norm": corrected_norm,
                "corrected_480": corrected480,
            }

            print(
                f"{method:10s} | "
                f"J={c4_J:.6e}->{final_J:.6e} | "
                f"drop={100*drop_vs_c4:+7.3f}% | "
                f"MSE={c4_m['mse']:.3f}->{mm['mse']:.3f} | "
                f"SSIM={c4_m['ssim']:.5f}->{mm['ssim']:.5f} | "
                f"accepted={bool(pinfo['accepted'])} | "
                f"step={float(pinfo['accepted_step_mps']):.4f}"
            )

        # Save light trajectory state.
        np.savez_compressed(
            out / f"test_{sample_id}_terminal_trajectory.npz",
            hint_norm=hint_norm,
            gt_norm=gt_norm,
            c4_final_norm=c4_final_norm,
            terminal_exact_norm=terminal_outputs[
                "C4+TExact"
            ]["corrected_norm"],
            terminal_gra_norm=terminal_outputs[
                "C4+TGRA"
            ]["corrected_norm"],
        )

        print()

    write_csv(
        out / "anchor_pareto_scan.csv",
        anchor_rows,
    )

    # -----------------------------------------------------------------
    # Aggregate.
    # -----------------------------------------------------------------
    methods = [
        "Hint",
        "C4-only",
        "C4+TExact",
        "C4+TGRA",
    ]

    method_summary = {}

    c4_by_sample = {
        int(r["sample"]): r
        for r in rows
        if r["method"] == "C4-only"
    }

    for method in methods:
        sub = [
            r for r in rows
            if r["method"] == method
        ]

        s = {
            "num_samples": len(sub),
            "mse": B.stat([
                r["mse"] for r in sub
            ]),
            "mae": B.stat([
                r["mae"] for r in sub
            ]),
            "psnr": B.stat([
                r["psnr"] for r in sub
            ]),
            "ssim": B.stat([
                r["ssim"] for r in sub
            ]),
            "measurement_J": B.stat([
                r["measurement_J"]
                for r in sub
            ]),
            "method_core_seconds": B.stat([
                r["method_core_seconds"]
                for r in sub
            ]),
            "true_source_equiv_core": B.stat([
                r["true_source_equiv_core"]
                for r in sub
            ]),
        }

        if method in {
            "C4+TExact",
            "C4+TGRA",
        }:
            s["accepted_count"] = int(
                sum(
                    bool(r["physics_accepted"])
                    for r in sub
                )
            )

            s["strict_J_win_vs_C4_count"] = int(
                sum(
                    float(r["measurement_J"])
                    <
                    float(
                        c4_by_sample[
                            int(r["sample"])
                        ]["measurement_J"]
                    )
                    for r in sub
                )
            )

            s["nonincrease_J_vs_C4_count"] = int(
                sum(
                    float(r["measurement_J"])
                    <=
                    float(
                        c4_by_sample[
                            int(r["sample"])
                        ]["measurement_J"]
                    )
                    + 1e-12
                    for r in sub
                )
            )

            s["MSE_win_vs_C4_count"] = int(
                sum(
                    float(r["mse"])
                    <
                    float(
                        c4_by_sample[
                            int(r["sample"])
                        ]["mse"]
                    )
                    for r in sub
                )
            )

            s["SSIM_win_vs_C4_count"] = int(
                sum(
                    float(r["ssim"])
                    >
                    float(
                        c4_by_sample[
                            int(r["sample"])
                        ]["ssim"]
                    )
                    for r in sub
                )
            )

            s["relative_J_drop_vs_C4"] = B.stat([
                r["physics_drop"]
                for r in sub
            ])

            s["MSE_gain_vs_C4"] = B.stat([
                r["MSE_gain_vs_C4"]
                for r in sub
            ])

        method_summary[method] = s

    c4s = method_summary["C4-only"]
    gras = method_summary["C4+TGRA"]

    # Terminal GRA is physics-valid when it never increases true J.
    physics_pass = (
        gras["nonincrease_J_vs_C4_count"]
        == gras["num_samples"]
        and
        gras["measurement_J"]["mean"]
        <= c4s["measurement_J"]["mean"]
        + 1e-12
    )

    image_pass = (
        gras["mse"]["mean"]
        <= c4s["mse"]["mean"]
        or
        gras["MSE_win_vs_C4_count"]
        >= int(
            np.ceil(
                0.5
                * gras["num_samples"]
            )
        )
    )

    if physics_pass and image_pass:
        gate = "STRONG_TERMINAL_PASS"
    elif physics_pass:
        gate = "PHYSICS_PASS_IMAGE_TRADEOFF"
    else:
        gate = "TERMINAL_FAIL"

    # -----------------------------------------------------------------
    # Save.
    # -----------------------------------------------------------------
    per_csv = out / "per_sample_metrics.csv"
    phys_csv = out / "terminal_physics_metrics.csv"
    attempts_csv = out / "gra_backtracking_attempts.csv"
    audit_csv = out / "trajectory_audit.csv"
    method_csv = out / "method_summary.csv"

    write_csv(per_csv, rows)
    write_csv(phys_csv, physics_rows)
    write_csv(attempts_csv, attempt_rows)
    write_csv(audit_csv, audit_rows)

    method_rows = []
    for method in methods:
        s = method_summary[method]
        method_rows.append({
            "method": method,
            "num_samples": s["num_samples"],
            "MSE_mean": s["mse"]["mean"],
            "MAE_mean": s["mae"]["mean"],
            "PSNR_mean": s["psnr"]["mean"],
            "SSIM_mean": s["ssim"]["mean"],
            "measurement_J_mean": s["measurement_J"]["mean"],
            "core_seconds_mean": s["method_core_seconds"]["mean"],
            "source_equiv_mean": s["true_source_equiv_core"]["mean"],
            "accepted_count": s.get(
                "accepted_count",
                "",
            ),
            "J_win_vs_C4": s.get(
                "strict_J_win_vs_C4_count",
                "",
            ),
            "J_nonincrease_vs_C4": s.get(
                "nonincrease_J_vs_C4_count",
                "",
            ),
            "MSE_win_vs_C4": s.get(
                "MSE_win_vs_C4_count",
                "",
            ),
            "SSIM_win_vs_C4": s.get(
                "SSIM_win_vs_C4_count",
                "",
            ),
            "mean_J_drop_vs_C4": (
                s["relative_J_drop_vs_C4"]["mean"]
                if "relative_J_drop_vs_C4" in s
                else ""
            ),
            "mean_MSE_gain_vs_C4": (
                s["MSE_gain_vs_C4"]["mean"]
                if "MSE_gain_vs_C4" in s
                else ""
            ),
        })

    write_csv(method_csv, method_rows)

    summary = {
        "experiment": (
            "C6.6 Terminal Guarded Physics Correction"
        ),
        "split": args.split_label,
        "samples": [
            args.start,
            args.end,
        ],
        "protocol": {
            "c4_seed": args.seed,
            "c4_ts": list(B.C4_TS),
            "c4_nfe": len(B.C4_TS) - 1,
            "physics_location": (
                "after final C4 clean reconstruction"
            ),
            "post_physics_consistency_mapping": False,
            "gra_initial_step_mps": args.initial_step,
            "gra_backtrack_factor": args.backtrack_factor,
            "gra_max_backtracks": args.max_backtracks,
            "acceptance_rule": (
                "accept iff true 8-source measurement J_trial < J_current; "
                "no GT/MSE/SSIM/cosine guard"
            ),
            "mgno_ckpt": str(args.mgno_ckpt),
            "background64_npz": str(
                args.background64_npz
            ),
        },
        "methods": method_summary,
        "terminal_gate": gate,
        "interpretation": (
            "Terminal GRA is evaluated as a final guarded physics projector. "
            "No generative mapping is allowed after the accepted physics update."
        ),
        "files": {
            "per_sample_metrics": str(per_csv),
            "terminal_physics_metrics": str(phys_csv),
            "gra_backtracking_attempts": str(
                attempts_csv
            ),
            "trajectory_audit": str(audit_csv),
            "method_summary": str(method_csv),
        },
    }

    summary_path = out / "summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Console summary.
    # -----------------------------------------------------------------
    print("=" * 142)
    print("C6.6 FINAL SUMMARY")
    print("=" * 142)
    print(
        f"{'method':>12s} "
        f"{'MSE':>11s} "
        f"{'MAE':>9s} "
        f"{'PSNR':>9s} "
        f"{'SSIM':>9s} "
        f"{'meas_J':>12s} "
        f"{'Jwin':>8s} "
        f"{'MSEwin':>8s} "
        f"{'src-eq':>9s} "
        f"{'time':>9s}"
    )

    for method in methods:
        s = method_summary[method]

        jwin = (
            "-"
            if method in {"Hint", "C4-only"}
            else (
                f"{s['strict_J_win_vs_C4_count']}"
                f"/{s['num_samples']}"
            )
        )

        msewin = (
            "-"
            if method in {"Hint", "C4-only"}
            else (
                f"{s['MSE_win_vs_C4_count']}"
                f"/{s['num_samples']}"
            )
        )

        print(
            f"{method:>12s} "
            f"{s['mse']['mean']:11.4f} "
            f"{s['mae']['mean']:9.4f} "
            f"{s['psnr']['mean']:9.4f} "
            f"{s['ssim']['mean']:9.5f} "
            f"{s['measurement_J']['mean']:12.5e} "
            f"{jwin:>8s} "
            f"{msewin:>8s} "
            f"{s['true_source_equiv_core']['mean']:9.2f} "
            f"{s['method_core_seconds']['mean']:9.3f}"
        )

    print()
    print(
        "Terminal Exact accepted =",
        f"{method_summary['C4+TExact']['accepted_count']}"
        f"/{method_summary['C4+TExact']['num_samples']}"
    )
    print(
        "Terminal GRA accepted   =",
        f"{method_summary['C4+TGRA']['accepted_count']}"
        f"/{method_summary['C4+TGRA']['num_samples']}"
    )
    print(
        "Terminal GRA J noninc   =",
        f"{method_summary['C4+TGRA']['nonincrease_J_vs_C4_count']}"
        f"/{method_summary['C4+TGRA']['num_samples']}"
    )
    print(
        "Terminal GRA mean Jdrop =",
        f"{100*method_summary['C4+TGRA']['relative_J_drop_vs_C4']['mean']:+.3f}%"
    )
    print(
        "Terminal GRA mean MSEgain =",
        f"{100*method_summary['C4+TGRA']['MSE_gain_vs_C4']['mean']:+.3f}%"
    )
    print(
        "C6.6 GATE               =",
        gate,
    )
    print(
        "summary                  =",
        summary_path,
    )


if __name__ == "__main__":
    main()
