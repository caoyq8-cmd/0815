
#!/usr/bin/env python3
"""
C6.8 Statistical Consolidation and Thesis-Table Export

Reads:
  - C6.6 DEV run
  - C6.6 secondary-transfer run
  - C6.7 structural trade-off run

Produces:
  - combined_method_summary.csv
  - paired_terminal_summary.csv
  - structural_tradeoff_summary.csv
  - c6_8_summary.json
  - c6_thesis_tables.tex

Pure post-processing only. No model inference and no tuning.
"""

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
    x = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if x.size == 0:
        return None
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)


def pct(x):
    return 100.0 * x


def latex_escape(s):
    return str(s).replace("_", r"\_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_run", required=True)
    ap.add_argument("--transfer_run", required=True)
    ap.add_argument("--c67_run", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = [
        ("DEV10", Path(args.dev_run)),
        ("Transfer10", Path(args.transfer_run)),
    ]

    all_rows = []
    per_split = {}

    for split_name, run in runs:
        rows = read_csv(run / "per_sample_metrics.csv")
        for r in rows:
            rr = dict(r)
            rr["_split"] = split_name
            rr["_sample_global"] = f"{split_name}:{r['sample']}"
            all_rows.append(rr)
        per_split[split_name] = rows

    methods = [
        "Hint",
        "C4-only",
        "C4+TExact",
        "C4+TGRA",
    ]

    combined_rows = []
    combined_summary = {}

    for method in methods:
        sub = [r for r in all_rows if r["method"] == method]

        s = {
            "num_samples": len(sub),
            "mse": stat([F(r["mse"]) for r in sub]),
            "mae": stat([F(r["mae"]) for r in sub]),
            "psnr": stat([F(r["psnr"]) for r in sub]),
            "ssim": stat([F(r["ssim"]) for r in sub]),
            "measurement_J": stat([F(r["measurement_J"]) for r in sub]),
            "core_seconds": stat([F(r["method_core_seconds"]) for r in sub]),
            "source_equiv": stat([F(r["true_source_equiv_core"]) for r in sub]),
        }

        combined_summary[method] = s

        combined_rows.append({
            "method": method,
            "n": len(sub),
            "MSE_mean": s["mse"]["mean"],
            "MAE_mean": s["mae"]["mean"],
            "PSNR_mean": s["psnr"]["mean"],
            "SSIM_mean": s["ssim"]["mean"],
            "measurement_J_mean": s["measurement_J"]["mean"],
            "core_seconds_mean": s["core_seconds"]["mean"],
            "source_equiv_mean": s["source_equiv"]["mean"],
        })

    c4_by_key = {
        r["_sample_global"]: r
        for r in all_rows
        if r["method"] == "C4-only"
    }

    paired_rows = []
    paired_summary = {}

    for method in ["C4+TExact", "C4+TGRA"]:
        sub = [r for r in all_rows if r["method"] == method]

        j_win = 0
        mse_win = 0
        ssim_win = 0
        j_drop = []
        mse_gain = []

        for r in sub:
            c4 = c4_by_key[r["_sample_global"]]

            j0 = F(c4["measurement_J"])
            j1 = F(r["measurement_J"])
            m0 = F(c4["mse"])
            m1 = F(r["mse"])
            s0 = F(c4["ssim"])
            s1 = F(r["ssim"])

            j_win += int(j1 < j0)
            mse_win += int(m1 < m0)
            ssim_win += int(s1 > s0)

            j_drop.append((j0 - j1) / max(abs(j0), 1e-30))
            mse_gain.append((m0 - m1) / max(abs(m0), 1e-30))

        paired_summary[method] = {
            "J_win": j_win,
            "MSE_win": mse_win,
            "SSIM_win": ssim_win,
            "num_samples": len(sub),
            "relative_J_drop": stat(j_drop),
            "relative_MSE_gain": stat(mse_gain),
        }

        paired_rows.append({
            "method": method,
            "n": len(sub),
            "J_win": j_win,
            "MSE_win": mse_win,
            "SSIM_win": ssim_win,
            "mean_relative_J_drop": float(np.mean(j_drop)),
            "median_relative_J_drop": float(np.median(j_drop)),
            "mean_relative_MSE_gain": float(np.mean(mse_gain)),
            "median_relative_MSE_gain": float(np.median(mse_gain)),
        })

    # Oracle retention / efficiency.
    c4_mse = combined_summary["C4-only"]["mse"]["mean"]
    ex_mse = combined_summary["C4+TExact"]["mse"]["mean"]
    gr_mse = combined_summary["C4+TGRA"]["mse"]["mean"]

    c4_j = combined_summary["C4-only"]["measurement_J"]["mean"]
    ex_j = combined_summary["C4+TExact"]["measurement_J"]["mean"]
    gr_j = combined_summary["C4+TGRA"]["measurement_J"]["mean"]

    mse_oracle_gain = c4_mse - ex_mse
    gra_mse_gain = c4_mse - gr_mse
    mse_oracle_retention = (
        gra_mse_gain / mse_oracle_gain
        if abs(mse_oracle_gain) > 1e-30
        else float("nan")
    )

    j_oracle_gain = c4_j - ex_j
    gra_j_gain = c4_j - gr_j
    j_oracle_retention = (
        gra_j_gain / j_oracle_gain
        if abs(j_oracle_gain) > 1e-30
        else float("nan")
    )

    exact_src = combined_summary["C4+TExact"]["source_equiv"]["mean"]
    gra_src = combined_summary["C4+TGRA"]["source_equiv"]["mean"]
    source_reduction = 1.0 - gra_src / exact_src

    exact_time = combined_summary["C4+TExact"]["core_seconds"]["mean"]
    gra_time = combined_summary["C4+TGRA"]["core_seconds"]["mean"]
    wallclock_ratio = gra_time / exact_time

    # C6.7.
    c67 = json.loads(
        (Path(args.c67_run) / "summary.json").read_text(encoding="utf-8")
    )
    g = c67["groups"]["combined"]
    tg = g["paired_vs_c4"]["C4+TGRA"]

    metric_map = {
        "delta_mse_vs_c4": "Delta MSE",
        "delta_mae_vs_c4": "Delta MAE",
        "delta_ssim_vs_c4": "Delta SSIM",
        "delta_smooth5_mse_vs_c4": "Delta smooth5 MSE",
        "delta_smooth5_ssim_vs_c4": "Delta smooth5 SSIM",
        "delta_smooth9_mse_vs_c4": "Delta smooth9 MSE",
        "delta_smooth9_ssim_vs_c4": "Delta smooth9 SSIM",
        "delta_inner32_mse_vs_c4": "Delta inner32 MSE",
        "delta_border32_mse_vs_c4": "Delta border32 MSE",
        "delta_p95_vs_c4": "Delta p95 |err|",
        "delta_p99_vs_c4": "Delta p99 |err|",
        "delta_max_vs_c4": "Delta max |err|",
    }

    structural_rows = []
    for key, label in metric_map.items():
        d = tg["metric_deltas"][key]
        structural_rows.append({
            "metric": label,
            "mean": d["stat"]["mean"],
            "median": d["stat"]["median"],
            "std": d["stat"]["std"],
            "bootstrap_low95": d["bootstrap_mean_ci95"]["low95"],
            "bootstrap_high95": d["bootstrap_mean_ci95"]["high95"],
            "wilcoxon_p": d["wilcoxon_p_two_sided"],
            "win_count": d["win_count"],
            "n": d["num_samples"],
        })

    redistribution = {
        "improved_pixel_fraction_mean":
            tg["improved_pixel_fraction"]["mean"],
        "worsened_pixel_fraction_mean":
            tg["worsened_pixel_fraction"]["mean"],
        "top10_c4_error_mse_gain_frac_mean":
            tg["top10_c4_error_mse_gain_frac"]["mean"],
        "bottom90_c4_error_mae_gain_frac_mean":
            tg["bottom90_c4_error_mae_gain_frac"]["mean"],
    }

    spec = c67["spectrum"]["combined"]["C4+TGRA"]
    spectrum = {
        "low_energy_frac_mean": spec["low_energy_frac"]["mean"],
        "mid_energy_frac_mean": spec["mid_energy_frac"]["mean"],
        "high_energy_frac_mean": spec["high_energy_frac"]["mean"],
        "update_rms_mean": spec["update_rms"]["mean"],
        "update_tv_mean": spec["update_tv"]["mean"],
    }

    final_summary = {
        "experiment": "C6.8 Statistical Consolidation",
        "combined_n": len({
            r["_sample_global"]
            for r in all_rows
            if r["method"] == "C4-only"
        }),
        "combined_methods": combined_summary,
        "paired_terminal": paired_summary,
        "oracle_efficiency": {
            "TGRA_MSE_gain_retention_vs_Exact":
                mse_oracle_retention,
            "TGRA_measurement_gain_retention_vs_Exact":
                j_oracle_retention,
            "TGRA_source_equivalent_reduction_vs_Exact":
                source_reduction,
            "TGRA_wallclock_ratio_vs_Exact":
                wallclock_ratio,
        },
        "structural_redistribution": redistribution,
        "terminal_update_spectrum": spectrum,
        "c67_structural_rows": structural_rows,
    }

    write_csv(out / "combined_method_summary.csv", combined_rows)
    write_csv(out / "paired_terminal_summary.csv", paired_rows)
    write_csv(out / "structural_tradeoff_summary.csv", structural_rows)

    (out / "c6_8_summary.json").write_text(
        json.dumps(final_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # LaTeX export.
    # -------------------------------------------------------------
    lines = []
    lines.append(r"% Auto-generated by C6.8 statistical consolidation")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{C6.6终端物理校正在20个样本上的合并结果}")
    lines.append(r"\label{tab:c66_combined}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"方法 & MSE$\downarrow$ & MAE$\downarrow$ & PSNR$\uparrow$ & SSIM$\uparrow$ & $J\downarrow$ & Source-eq \\")
    lines.append(r"\midrule")
    for row in combined_rows:
        lines.append(
            f"{latex_escape(row['method'])} & "
            f"{row['MSE_mean']:.4f} & "
            f"{row['MAE_mean']:.4f} & "
            f"{row['PSNR_mean']:.4f} & "
            f"{row['SSIM_mean']:.5f} & "
            f"{row['measurement_J_mean']:.5e} & "
            f"{row['source_equiv_mean']:.2f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{TGRA相对C4-only的结构性误差再分配统计}")
    lines.append(r"\label{tab:c67_tradeoff}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"指标 & 均值变化 & 95\% CI下界 & 95\% CI上界 & Wilcoxon $p$ \\")
    lines.append(r"\midrule")
    for row in structural_rows:
        p = row["wilcoxon_p"]
        ptxt = "--" if p is None else f"{p:.4g}"
        lines.append(
            f"{latex_escape(row['metric'])} & "
            f"{row['mean']:+.6f} & "
            f"{row['bootstrap_low95']:+.6f} & "
            f"{row['bootstrap_high95']:+.6f} & "
            f"{ptxt} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    lines.append(r"% Key consolidated statements")
    lines.append(
        f"% TGRA J wins: {paired_summary['C4+TGRA']['J_win']}/"
        f"{paired_summary['C4+TGRA']['num_samples']}"
    )
    lines.append(
        f"% TGRA MSE wins: {paired_summary['C4+TGRA']['MSE_win']}/"
        f"{paired_summary['C4+TGRA']['num_samples']}"
    )
    lines.append(
        f"% TGRA source-equivalent reduction vs Exact: "
        f"{pct(source_reduction):.2f}%"
    )
    lines.append(
        f"% TGRA MSE-gain retention vs Exact: "
        f"{pct(mse_oracle_retention):.2f}%"
    )
    lines.append(
        f"% TGRA measurement-gain retention vs Exact: "
        f"{pct(j_oracle_retention):.2f}%"
    )
    lines.append(
        f"% TGRA wall-clock ratio vs Exact: "
        f"{wallclock_ratio:.3f}x"
    )
    lines.append(
        f"% Pixel improved/worsened: "
        f"{pct(redistribution['improved_pixel_fraction_mean']):.2f}% / "
        f"{pct(redistribution['worsened_pixel_fraction_mean']):.2f}%"
    )
    lines.append(
        f"% Top10 error MSE gain: "
        f"{pct(redistribution['top10_c4_error_mse_gain_frac_mean']):.2f}%"
    )
    lines.append(
        f"% Bottom90 MAE gain: "
        f"{pct(redistribution['bottom90_c4_error_mae_gain_frac_mean']):.2f}%"
    )

    (out / "c6_thesis_tables.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 150)
    print("C6.8 STATISTICAL CONSOLIDATION")
    print("=" * 150)
    print(f"combined samples = {final_summary['combined_n']}")
    print()
    print(
        "TGRA J wins / MSE wins / SSIM wins = "
        f"{paired_summary['C4+TGRA']['J_win']}/"
        f"{paired_summary['C4+TGRA']['num_samples']} / "
        f"{paired_summary['C4+TGRA']['MSE_win']}/"
        f"{paired_summary['C4+TGRA']['num_samples']} / "
        f"{paired_summary['C4+TGRA']['SSIM_win']}/"
        f"{paired_summary['C4+TGRA']['num_samples']}"
    )
    print(
        "TGRA source-eq reduction vs Exact = "
        f"{pct(source_reduction):+.2f}%"
    )
    print(
        "TGRA MSE-gain retention vs Exact  = "
        f"{pct(mse_oracle_retention):+.2f}%"
    )
    print(
        "TGRA J-gain retention vs Exact    = "
        f"{pct(j_oracle_retention):+.2f}%"
    )
    print(
        "TGRA wall-clock ratio vs Exact    = "
        f"{wallclock_ratio:.3f}x"
    )
    print()
    print(
        "Pixel improved / worsened         = "
        f"{pct(redistribution['improved_pixel_fraction_mean']):.2f}% / "
        f"{pct(redistribution['worsened_pixel_fraction_mean']):.2f}%"
    )
    print(
        "Top10 C4-error MSE gain           = "
        f"{pct(redistribution['top10_c4_error_mse_gain_frac_mean']):+.2f}%"
    )
    print(
        "Bottom90 C4-error MAE gain        = "
        f"{pct(redistribution['bottom90_c4_error_mae_gain_frac_mean']):+.2f}%"
    )
    print(
        "Spectrum low/mid/high             = "
        f"{spectrum['low_energy_frac_mean']:.4f} / "
        f"{spectrum['mid_energy_frac_mean']:.4f} / "
        f"{spectrum['high_energy_frac_mean']:.4f}"
    )
    print()
    print("saved =", out)


if __name__ == "__main__":
    main()
