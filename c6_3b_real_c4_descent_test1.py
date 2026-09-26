import argparse
import json
import hashlib
from pathlib import Path

import numpy as np

from c5_directional_fd_audit import (
    measurement_loss,
    normalize_direction,
    scalar,
)

from c5_paper_ano_val5 import cosine


def array_hash(x):
    return hashlib.sha256(
        np.ascontiguousarray(
            np.asarray(x, dtype=np.float32)
        ).tobytes()
    ).hexdigest()


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--sample",
        required=True,
    )

    ap.add_argument(
        "--gradient",
        required=True,
    )

    ap.add_argument(
        "--step",
        type=float,
        default=1.5,
    )

    ap.add_argument(
        "--device",
        default="cuda:0",
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    z = np.load(
        args.sample,
        allow_pickle=True,
    )

    d = np.load(
        args.gradient,
        allow_pickle=True,
    )

    candidate = z[
        "candidate_480"
    ].astype(np.float32)

    cached_candidate = d[
        "candidate_speed"
    ].astype(np.float32)

    # ------------------------------------------------------------
    # State identity guard
    # ------------------------------------------------------------

    if not np.array_equal(
        candidate,
        cached_candidate,
    ):
        raise RuntimeError(
            "Candidate / gradient-state mismatch."
        )

    current_hash = array_hash(candidate)

    if "candidate_sha256" in d.files:

        cached_hash = str(
            np.asarray(
                d["candidate_sha256"]
            ).reshape(-1)[0]
        )

        if current_hash != cached_hash:
            raise RuntimeError(
                "Candidate SHA256 mismatch."
            )

    g_exact = d[
        "cbs_gradient"
    ].astype(np.float64)

    g_arss = d[
        "full_ano_gradient"
    ].astype(np.float64)

    dobs = z[
        "dobs_complex"
    ].astype(np.complex64)

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

    print("=" * 110)
    print("C6.3B REAL-C4 FROZEN 1.5 m/s DESCENT — TEST1")
    print("=" * 110)

    print("candidate hash =", current_hash)
    print("step           =", args.step)

    # ------------------------------------------------------------
    # Baseline physical objective
    # ------------------------------------------------------------

    J0, rr0 = measurement_loss(
        candidate,
        dobs,
        src_indices,
        rec_indices,
        frequency,
        cbs_iters,
        boundary_width,
        boundary_strength,
        boundary_type,
        args.device,
    )

    print()
    print(f"J0  = {J0:.10e}")
    print(f"RR0 = {rr0:.8f}")

    print()
    print(
        "cos(ARSS,Exact) =",
        f"{cosine(g_arss, g_exact):+.8f}",
    )

    results = {}

    methods = {
        "Exact-CBS": g_exact,
        "ARSS-MgNO": g_arss,
    }

    for name, grad in methods.items():

        direction = normalize_direction(
            grad
        )

        direction_rms = float(
            np.sqrt(
                np.mean(
                    direction ** 2
                )
            )
        )

        # EXACT C5.7 update convention:
        # no new clipping, no retuning.
        trial = (
            candidate
            - args.step * direction
        ).astype(np.float32)

        J1, rr1 = measurement_loss(
            trial,
            dobs,
            src_indices,
            rec_indices,
            frequency,
            cbs_iters,
            boundary_width,
            boundary_strength,
            boundary_type,
            args.device,
        )

        relative_drop = (
            (J0 - J1)
            /
            max(abs(J0), 1e-30)
        )

        descent = bool(
            J1 < J0
        )

        results[name] = {
            "J0": float(J0),
            "J1": float(J1),
            "relative_drop": float(
                relative_drop
            ),
            "rr0": float(rr0),
            "rr1": float(rr1),
            "descent": descent,
            "direction_rms": direction_rms,
            "cosine_to_exact": float(
                cosine(
                    grad,
                    g_exact,
                )
            ),
            "trial_min": float(
                trial.min()
            ),
            "trial_max": float(
                trial.max()
            ),
        }

        print()
        print("-" * 110)
        print(name)
        print("-" * 110)

        print(
            "direction RMS =",
            f"{direction_rms:.8f}",
        )

        print(
            "cosine exact  =",
            f"{results[name]['cosine_to_exact']:+.8f}",
        )

        print(
            "J1            =",
            f"{J1:.10e}",
        )

        print(
            "RR1           =",
            f"{rr1:.8f}",
        )

        print(
            "objective drop =",
            f"{100*relative_drop:+.4f}%",
        )

        print(
            "descent        =",
            descent,
        )

        print(
            "trial range    =",
            float(trial.min()),
            float(trial.max()),
        )

    out = Path(args.output)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "sample": 1,
        "step": args.step,
        "candidate_sha256": current_hash,
        "gradient_cosine":
            float(
                cosine(
                    g_arss,
                    g_exact,
                )
            ),
        "results": results,
    }

    out.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print("C6.3B TEST1 DECISION")
    print("=" * 110)

    exact_ok = results[
        "Exact-CBS"
    ]["descent"]

    arss_ok = results[
        "ARSS-MgNO"
    ]["descent"]

    if exact_ok and arss_ok:

        print(
            "[PASS] Exact and ARSS both descend."
        )

    elif exact_ok and not arss_ok:

        print(
            "[SURROGATE TRANSFER FAIL] "
            "Exact descends but ARSS does not."
        )

    elif not exact_ok:

        print(
            "[REFERENCE/STEP WARNING] "
            "Exact direction itself does not "
            "descend at frozen 1.5 m/s."
        )

    print()
    print("saved =", out)


if __name__ == "__main__":
    main()
