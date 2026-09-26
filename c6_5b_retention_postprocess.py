
#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def b(x):
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def stats(vals):
    x = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if x.size == 0:
        return None
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Robust post-processing for C6.5B physics-retention audit."
    )
    ap.add_argument("--run_dir", required=True)
    ap.add_argument(
        "--tiny_rel_drop",
        type=float,
        default=1e-3,
        help="Diagnostic threshold for tiny accepted physics drops (default 0.1%%).",
    )
    args = ap.parse_args()

    root = Path(args.run_dir)
    ret_path = root / "physics_retention_metrics.csv"
    per_path = root / "per_sample_metrics.csv"

    if not ret_path.exists():
        raise FileNotFoundError(ret_path)
    if not per_path.exists():
        raise FileNotFoundError(per_path)

    ret = read_csv(ret_path)
    per = read_csv(per_path)

    # Final C4-only reference by sample.
    c4 = {}
    for r in per:
        if r["method"] == "C4-only":
            sid = int(r["sample"])
            c4[sid] = {
                "J": f(r["measurement_J"]),
                "mse": f(r["mse"]),
            }

    out = {
        "run_dir": str(root),
        "tiny_rel_drop_threshold": args.tiny_rel_drop,
        "methods": {},
    }

    print("=" * 150)
    print("C6.5B ROBUST PHYSICS-RETENTION POSTPROCESS")
    print("=" * 150)

    for method in ["C4+Exact", "C4+GRA"]:
        rows = [r for r in ret if r["method"] == method]
        accepted = [r for r in rows if b(r["accepted"])]

        physics_gains = np.asarray(
            [f(r["physics_J_gain"]) for r in accepted],
            dtype=np.float64,
        )
        final_gains_pre = np.asarray(
            [f(r["final_J_gain_vs_pre"]) for r in accepted],
            dtype=np.float64,
        )

        denom = float(np.sum(physics_gains))
        numer = float(np.sum(final_gains_pre))

        if abs(denom) > 1e-30:
            aggregate_retention = numer / denom
            aggregate_washout = 1.0 - aggregate_retention
        else:
            aggregate_retention = float("nan")
            aggregate_washout = float("nan")

        individual_ret = [
            f(r["retention_ratio"])
            for r in accepted
        ]

        rel_physics_drop = []
        tiny = []
        for r in accepted:
            j0 = f(r["J_before_physics"])
            gain = f(r["physics_J_gain"])
            rel = gain / max(abs(j0), 1e-30)
            rel_physics_drop.append(rel)
            if rel < args.tiny_rel_drop:
                tiny.append({
                    "sample": int(r["sample"]),
                    "relative_physics_drop": rel,
                    "retention_ratio": f(r["retention_ratio"]),
                })

        # "Survival" means the final state is still better than the
        # pre-physics clean state used as the insertion anchor.
        survival_count = sum(
            f(r["J_after_cm2"]) < f(r["J_before_physics"])
            for r in rows
        )

        # This is the comparison that the previous console label did NOT give:
        # final physics arm versus final C4-only output.
        final_c4_J_wins = 0
        final_c4_mse_wins = 0
        final_J_deltas = []
        final_mse_deltas = []

        for r in rows:
            sid = int(r["sample"])
            if sid not in c4:
                continue

            jf = f(r["J_after_cm2"])
            mf = f(r["final_mse"])

            jc = c4[sid]["J"]
            mc = c4[sid]["mse"]

            final_c4_J_wins += int(jf < jc)
            final_c4_mse_wins += int(mf < mc)

            final_J_deltas.append(jf - jc)
            final_mse_deltas.append(mf - mc)

        cosine_vals = [
            f(r["cm2_vs_physics_cos"])
            for r in accepted
        ]
        directional_ret = [
            f(r["directional_retention"])
            for r in accepted
        ]
        update_ratio = [
            f(r["cm2_to_physics_update_rms_ratio"])
            for r in accepted
        ]

        method_out = {
            "num_samples": len(rows),
            "accepted_count": len(accepted),
            "survival_vs_prephysics_count": int(survival_count),
            "final_vs_c4only_J_win_count": int(final_c4_J_wins),
            "final_vs_c4only_MSE_win_count": int(final_c4_mse_wins),
            "aggregate_retention_ratio": float(aggregate_retention),
            "aggregate_washout_fraction": float(aggregate_washout),
            "sum_physics_J_gain": denom,
            "sum_final_J_gain_vs_pre": numer,
            "individual_retention_ratio": stats(individual_ret),
            "relative_physics_drop": stats(rel_physics_drop),
            "tiny_denominator_cases": tiny,
            "cm2_vs_physics_cos": stats(cosine_vals),
            "directional_retention": stats(directional_ret),
            "cm2_to_physics_update_rms_ratio": stats(update_ratio),
            "final_J_delta_vs_c4only": stats(final_J_deltas),
            "final_MSE_delta_vs_c4only": stats(final_mse_deltas),
        }
        out["methods"][method] = method_out

        ir = method_out["individual_retention_ratio"]
        cs = method_out["cm2_vs_physics_cos"]

        print()
        print(method)
        print("-" * 150)
        print(
            f"accepted                         = "
            f"{len(accepted)}/{len(rows)}"
        )
        print(
            f"final better than pre-physics    = "
            f"{survival_count}/{len(rows)}"
        )
        print(
            f"final J better than C4-only      = "
            f"{final_c4_J_wins}/{len(rows)}"
        )
        print(
            f"final MSE better than C4-only    = "
            f"{final_c4_mse_wins}/{len(rows)}"
        )
        print(
            f"aggregate retention              = "
            f"{aggregate_retention:+.6f}"
        )
        print(
            f"aggregate washout                = "
            f"{aggregate_washout:+.6f}"
        )

        if ir is not None:
            print(
                f"individual retention median      = "
                f"{ir['median']:+.6f}"
            )
            print(
                f"individual retention mean        = "
                f"{ir['mean']:+.6f}"
            )

        if cs is not None:
            print(
                f"CM2-vs-physics cosine mean/med   = "
                f"{cs['mean']:+.6f} / "
                f"{cs['median']:+.6f}"
            )

        print(
            f"tiny accepted drops (<{100*args.tiny_rel_drop:.3f}%) = "
            f"{len(tiny)}"
        )

        if tiny:
            for item in tiny:
                print(
                    f"  sample {item['sample']:2d}: "
                    f"phys_drop={100*item['relative_physics_drop']:+.6f}% "
                    f"raw_ret={item['retention_ratio']:+.6f}"
                )

    out_path = root / "physics_retention_robust_summary.json"
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 150)
    print("saved =", out_path)
    print("=" * 150)


if __name__ == "__main__":
    main()
