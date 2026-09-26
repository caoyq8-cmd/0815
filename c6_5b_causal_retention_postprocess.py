
#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def F(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def B(x):
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def stat(vals):
    x = np.asarray(
        [v for v in vals if np.isfinite(v)],
        dtype=np.float64,
    )
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
        description=(
            "Paired causal retention audit for C6.5B. "
            "Separates the net physics effect from CM2's own trajectory effect."
        )
    )
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    root = Path(args.run_dir)
    ret_rows = read_csv(root / "physics_retention_metrics.csv")
    per_rows = read_csv(root / "per_sample_metrics.csv")

    # Paired no-physics final reference.
    c4 = {}
    for r in per_rows:
        if r["method"] == "C4-only":
            sid = int(r["sample"])
            c4[sid] = {
                "J_final": F(r["measurement_J"]),
                "mse_final": F(r["mse"]),
                "mae_final": F(r["mae"]),
            }

    summary = {
        "definition": {
            "causal_retention_ratio": (
                "(J_C4only_final - J_physics_arm_final) / "
                "(J_prephysics - J_after_physics)"
            ),
            "interpretation": {
                "gt_1": "physics advantage is amplified after CM2",
                "between_0_1": "part of the physics advantage survives CM2",
                "eq_0": "no final advantage versus paired C4-only",
                "lt_0": "physics arm ends worse than paired C4-only",
            },
        },
        "methods": {},
    }

    print("=" * 154)
    print("C6.5B PAIRED CAUSAL PHYSICS-RETENTION AUDIT")
    print("=" * 154)

    for method in ["C4+Exact", "C4+GRA"]:
        rows = [r for r in ret_rows if r["method"] == method]
        accepted = [r for r in rows if B(r["accepted"])]

        indiv = []
        numerators = []
        denominators = []
        final_J_delta = []
        final_MSE_delta = []
        accepted_final_J_wins = 0
        accepted_final_MSE_wins = 0

        sample_rows = []

        for r in rows:
            sid = int(r["sample"])
            if sid not in c4:
                continue

            j_c4 = c4[sid]["J_final"]
            mse_c4 = c4[sid]["mse_final"]

            j_final = F(r["J_after_cm2"])
            mse_final = F(r["final_mse"])

            dJ_final = j_c4 - j_final
            dMSE_final = mse_c4 - mse_final

            final_J_delta.append(dJ_final)
            final_MSE_delta.append(dMSE_final)

            accepted_flag = B(r["accepted"])
            denom = F(r["physics_J_gain"])

            ratio = float("nan")
            if accepted_flag and denom > 1e-30:
                ratio = dJ_final / denom
                indiv.append(ratio)
                numerators.append(dJ_final)
                denominators.append(denom)

                accepted_final_J_wins += int(dJ_final > 0)
                accepted_final_MSE_wins += int(dMSE_final > 0)

            sample_rows.append({
                "sample": sid,
                "method": method,
                "accepted": accepted_flag,
                "physics_J_gain": denom,
                "J_C4only_final": j_c4,
                "J_physics_arm_final": j_final,
                "final_J_advantage_vs_C4only": dJ_final,
                "causal_retention_ratio": ratio,
                "MSE_C4only_final": mse_c4,
                "MSE_physics_arm_final": mse_final,
                "final_MSE_advantage_vs_C4only": dMSE_final,
                "cm2_vs_physics_cos": F(r["cm2_vs_physics_cos"]),
            })

        num_sum = float(np.sum(numerators)) if numerators else float("nan")
        den_sum = float(np.sum(denominators)) if denominators else float("nan")
        agg = (
            num_sum / den_sum
            if denominators and abs(den_sum) > 1e-30
            else float("nan")
        )

        # Overall paired wins include rejected cases as well.
        overall_J_wins = sum(v > 0 for v in final_J_delta)
        overall_MSE_wins = sum(v > 0 for v in final_MSE_delta)

        method_summary = {
            "num_samples": len(rows),
            "accepted_count": len(accepted),
            "accepted_final_J_win_count": int(accepted_final_J_wins),
            "accepted_final_MSE_win_count": int(accepted_final_MSE_wins),
            "overall_final_J_win_count": int(overall_J_wins),
            "overall_final_MSE_win_count": int(overall_MSE_wins),
            "aggregate_causal_retention_ratio": float(agg),
            "sum_final_J_advantage_vs_C4only": num_sum,
            "sum_physics_J_gain": den_sum,
            "individual_causal_retention": stat(indiv),
            "final_J_advantage_vs_C4only": stat(final_J_delta),
            "final_MSE_advantage_vs_C4only": stat(final_MSE_delta),
            "sample_rows": sample_rows,
        }
        summary["methods"][method] = method_summary

        s = method_summary["individual_causal_retention"]

        print()
        print(method)
        print("-" * 154)
        print(
            f"accepted                         = "
            f"{len(accepted)}/{len(rows)}"
        )
        print(
            f"accepted & final-J win           = "
            f"{accepted_final_J_wins}/{len(accepted)}"
            if accepted else
            "accepted & final-J win           = 0/0"
        )
        print(
            f"accepted & final-MSE win         = "
            f"{accepted_final_MSE_wins}/{len(accepted)}"
            if accepted else
            "accepted & final-MSE win         = 0/0"
        )
        print(
            f"overall final-J win vs C4-only   = "
            f"{overall_J_wins}/{len(rows)}"
        )
        print(
            f"overall final-MSE win vs C4-only = "
            f"{overall_MSE_wins}/{len(rows)}"
        )
        print(
            f"aggregate causal retention       = "
            f"{agg:+.6f}"
        )
        if s is not None:
            print(
                f"individual causal ret mean/med   = "
                f"{s['mean']:+.6f} / {s['median']:+.6f}"
            )

    # Save detailed paired table.
    detailed = []
    for method in ["C4+Exact", "C4+GRA"]:
        detailed.extend(summary["methods"][method]["sample_rows"])

    csv_path = root / "physics_retention_causal_pairs.csv"
    if detailed:
        fieldnames = list(detailed[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(detailed)

    # Remove row payload from compact JSON after writing CSV.
    compact = json.loads(json.dumps(summary))
    for method in compact["methods"]:
        compact["methods"][method].pop("sample_rows", None)

    json_path = root / "physics_retention_causal_summary.json"
    json_path.write_text(
        json.dumps(compact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 154)
    print("saved CSV  =", csv_path)
    print("saved JSON =", json_path)
    print("=" * 154)


if __name__ == "__main__":
    main()
