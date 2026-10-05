import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(
    "/home/featurize/work/USCT_repro/"
    "USCT_download/formal_phys20/results"
)

DEV = (
    ROOT
    / "fiveway_true_cbs_dev10_v3"
    / "fourway_per_sample.csv"
)

HELD = (
    ROOT
    / "fiveway_true_cbs_heldout10_v3"
    / "fourway_per_sample.csv"
)

OUT = (
    ROOT
    / "pgcr_mechanism_analysis"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def load_split(path, split_name):

    df = pd.read_csv(path)

    rows = []

    for sid in sorted(
        df["formal_val_id"].unique()
    ):

        d = df[
            df["formal_val_id"] == sid
        ]

        inv = d[
            d["method"] == "InversionNet"
        ].iloc[0]

        hp = d[
            d["method"] == "HP-C4"
        ].iloc[0]

        agr = d[
            d["method"] == "AGR-MSE"
        ].iloc[0]

        sta = d[
            d["method"] == "AGR-Stable"
        ].iloc[0]

        oracle = d[
            d["method"] == "GT256-Oracle"
        ].iloc[0]

        accept = (
            sta["measurement_rrmse"]
            <
            inv["measurement_rrmse"]
        )

        pgcr_rr = (
            sta["measurement_rrmse"]
            if accept
            else inv["measurement_rrmse"]
        )

        pgcr_mse = (
            sta["image_mse"]
            if accept
            else inv["image_mse"]
        )

        delta_mse = (
            sta["image_mse"]
            - inv["image_mse"]
        )

        delta_rr = (
            sta["measurement_rrmse"]
            - inv["measurement_rrmse"]
        )

        image_improve = (
            delta_mse < 0
        )

        physics_improve = (
            delta_rr < 0
        )

        if (
            image_improve
            and physics_improve
        ):
            quadrant = "both_improve"

        elif (
            image_improve
            and not physics_improve
        ):
            quadrant = "image_only"

        elif (
            not image_improve
            and physics_improve
        ):
            quadrant = "physics_only"

        else:
            quadrant = "both_worse"

        rows.append({
            "split":
                split_name,

            "formal_val_id":
                int(sid),

            "original_index":
                int(
                    inv["original_index"]
                ),

            "oracle_rrmse":
                float(
                    oracle[
                        "measurement_rrmse"
                    ]
                ),

            "inv_mse":
                float(
                    inv["image_mse"]
                ),

            "inv_rrmse":
                float(
                    inv[
                        "measurement_rrmse"
                    ]
                ),

            "hp_mse":
                float(
                    hp["image_mse"]
                ),

            "hp_rrmse":
                float(
                    hp[
                        "measurement_rrmse"
                    ]
                ),

            "agr_mse":
                float(
                    agr["image_mse"]
                ),

            "agr_rrmse":
                float(
                    agr[
                        "measurement_rrmse"
                    ]
                ),

            "stable_mse":
                float(
                    sta["image_mse"]
                ),

            "stable_rrmse":
                float(
                    sta[
                        "measurement_rrmse"
                    ]
                ),

            "delta_mse_stable":
                float(
                    delta_mse
                ),

            "delta_rrmse_stable":
                float(
                    delta_rr
                ),

            "stable_image_improve":
                bool(
                    image_improve
                ),

            "stable_physics_improve":
                bool(
                    physics_improve
                ),

            "quadrant":
                quadrant,

            "pgcr_accept":
                bool(
                    accept
                ),

            "pgcr_mse":
                float(
                    pgcr_mse
                ),

            "pgcr_rrmse":
                float(
                    pgcr_rr
                ),
        })

    return pd.DataFrame(rows)


dev = load_split(
    DEV,
    "phys-dev10",
)

held = load_split(
    HELD,
    "phys-heldout10",
)

all_df = pd.concat(
    [dev, held],
    ignore_index=True,
)

all_df.to_csv(
    OUT / "mechanism_per_sample.csv",
    index=False,
)


def method_summary(df, name):

    if name == "HP-C4":
        mse = df["hp_mse"]
        rr = df["hp_rrmse"]

    elif name == "AGR-MSE":
        mse = df["agr_mse"]
        rr = df["agr_rrmse"]

    elif name == "AGR-Stable":
        mse = df["stable_mse"]
        rr = df["stable_rrmse"]

    elif name == "PGCR-v1":
        mse = df["pgcr_mse"]
        rr = df["pgcr_rrmse"]

    else:
        raise ValueError(name)

    base_mse = df["inv_mse"]
    base_rr = df["inv_rrmse"]

    return {
        "mean_image_mse":
            float(mse.mean()),

        "mean_cbs_rrmse":
            float(rr.mean()),

        "image_wins":
            int(
                (mse < base_mse).sum()
            ),

        "physics_wins":
            int(
                (rr < base_rr).sum()
            ),

        "image_gain_pct":
            float(
                100
                * (
                    base_mse.mean()
                    - mse.mean()
                )
                / base_mse.mean()
            ),

        "physics_gain_pct":
            float(
                100
                * (
                    base_rr.mean()
                    - rr.mean()
                )
                / base_rr.mean()
            ),
    }


def summarize_split(df):

    quadrant_counts = (
        df["quadrant"]
        .value_counts()
        .to_dict()
    )

    pearson = float(
        df[
            "delta_mse_stable"
        ].corr(
            df[
                "delta_rrmse_stable"
            ],
            method="pearson",
        )
    )

    spearman = float(
        df[
            "delta_mse_stable"
        ].corr(
            df[
                "delta_rrmse_stable"
            ],
            method="spearman",
        )
    )

    return {
        "num_samples":
            int(len(df)),

        "baseline": {
            "mean_image_mse":
                float(
                    df[
                        "inv_mse"
                    ].mean()
                ),

            "mean_cbs_rrmse":
                float(
                    df[
                        "inv_rrmse"
                    ].mean()
                ),
        },

        "methods": {
            name:
                method_summary(
                    df,
                    name
                )

            for name in [
                "HP-C4",
                "AGR-MSE",
                "AGR-Stable",
                "PGCR-v1",
            ]
        },

        "quadrants":
            quadrant_counts,

        "delta_correlation": {
            "pearson":
                pearson,

            "spearman":
                spearman,
        },

        "pgcr": {
            "accepted":
                int(
                    df[
                        "pgcr_accept"
                    ].sum()
                ),

            "rejected":
                int(
                    (
                        ~df[
                            "pgcr_accept"
                        ]
                    ).sum()
                ),
        },
    }


summary = {
    "development":
        summarize_split(dev),

    "physics_heldout":
        summarize_split(held),

    "combined_descriptive":
        summarize_split(all_df),
}


with open(
    OUT / "mechanism_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ==========================================================
# FIGURE 1:
# AGR-Stable ΔImage MSE vs ΔCBS-RRMSE
# ==========================================================

fig, ax = plt.subplots(
    figsize=(7.2, 5.6)
)

for split_name, marker in [
    ("phys-dev10", "o"),
    ("phys-heldout10", "s"),
]:

    d = all_df[
        all_df["split"]
        == split_name
    ]

    ax.scatter(
        d["delta_mse_stable"],
        d["delta_rrmse_stable"],
        marker=marker,
        label=split_name,
        s=55,
    )

    for _, r in d.iterrows():

        ax.annotate(
            str(
                int(
                    r[
                        "formal_val_id"
                    ]
                )
            ),
            (
                r[
                    "delta_mse_stable"
                ],
                r[
                    "delta_rrmse_stable"
                ],
            ),
            xytext=(4,4),
            textcoords="offset points",
            fontsize=8,
        )


ax.axhline(
    0,
    linewidth=1,
)

ax.axvline(
    0,
    linewidth=1,
)

ax.set_xlabel(
    r"$\Delta$ Image MSE "
    r"(AGR-Stable - InversionNet)"
)

ax.set_ylabel(
    r"$\Delta$ CBS RRMSE "
    r"(AGR-Stable - InversionNet)"
)

ax.set_title(
    "Image-space improvement vs physics consistency"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUT
    / "delta_mse_vs_delta_cbs_rrmse.png",
    dpi=300,
)

fig.savefig(
    OUT
    / "delta_mse_vs_delta_cbs_rrmse.pdf",
)

plt.close(fig)


# ==========================================================
# FIGURE 2:
# Per-sample physics delta
# ==========================================================

fig, ax = plt.subplots(
    figsize=(8.0, 4.8)
)

x = (
    all_df[
        "formal_val_id"
    ].to_numpy()
)

y = (
    all_df[
        "delta_rrmse_stable"
    ].to_numpy()
)

ax.bar(
    x,
    y,
)

ax.axhline(
    0,
    linewidth=1,
)

ax.set_xlabel(
    "Formal validation ID"
)

ax.set_ylabel(
    r"$\Delta$ CBS RRMSE "
    r"(AGR-Stable - InversionNet)"
)

ax.set_title(
    "Per-sample physics effect of AGR-Stable"
)

ax.set_xticks(x)

fig.tight_layout()

fig.savefig(
    OUT
    / "stable_physics_delta_per_sample.png",
    dpi=300,
)

fig.savefig(
    OUT
    / "stable_physics_delta_per_sample.pdf",
)

plt.close(fig)


# ==========================================================
# LATEX TABLE
# ==========================================================

latex_path = (
    OUT
    / "pgcr_physics_results_table.tex"
)

with open(
    latex_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
r"""\begin{table}[htbp]
\centering
\caption{PGCR物理一致性实验结果}
\label{tab:pgcr_physics}
\begin{tabular}{llrrrr}
\toprule
数据划分 & 方法 &
Image MSE$\downarrow$ &
CBS-RRMSE$\downarrow$ &
物理改善(\%)$\uparrow$ &
Physics wins \\
\midrule
"""
    )

    for split_label, d in [
        ("phys-dev10", dev),
        ("phys-heldout10", held),
    ]:

        base_mse = (
            d["inv_mse"].mean()
        )

        base_rr = (
            d["inv_rrmse"].mean()
        )

        f.write(
            f"{split_label} & "
            f"InversionNet & "
            f"{base_mse:.4f} & "
            f"{base_rr:.5f} & "
            f"-- & -- \\\\\n"
        )

        for name in [
            "AGR-Stable",
            "PGCR-v1",
        ]:

            s = method_summary(
                d,
                name,
            )

            f.write(
                f" & {name} & "
                f"{s['mean_image_mse']:.4f} & "
                f"{s['mean_cbs_rrmse']:.5f} & "
                f"{s['physics_gain_pct']:.2f} & "
                f"{s['physics_wins']}/"
                f"{len(d)} \\\\\n"
            )

        f.write(
            r"\midrule" + "\n"
        )

    f.write(
r"""\bottomrule
\end{tabular}
\end{table}
"""
    )


print("=" * 100)
print("PGCR MECHANISM ANALYSIS")
print("=" * 100)

for split_key in [
    "development",
    "physics_heldout",
    "combined_descriptive",
]:

    s = summary[
        split_key
    ]

    print()
    print(
        split_key
    )

    print(
        "  n =",
        s["num_samples"]
    )

    print(
        "  quadrants =",
        s["quadrants"]
    )

    print(
        "  Pearson =",
        f"{s['delta_correlation']['pearson']:.6f}"
    )

    print(
        "  Spearman =",
        f"{s['delta_correlation']['spearman']:.6f}"
    )

    for name in [
        "AGR-Stable",
        "PGCR-v1",
    ]:

        m = (
            s[
                "methods"
            ][name]
        )

        print(
            f"  {name:12s} | "
            f"MSE={m['mean_image_mse']:.6f} | "
            f"R={m['mean_cbs_rrmse']:.8f} | "
            f"ImgWin={m['image_wins']}/"
            f"{s['num_samples']} | "
            f"PhysWin={m['physics_wins']}/"
            f"{s['num_samples']} | "
            f"PhysGain="
            f"{m['physics_gain_pct']:+.3f}%"
        )

print()
print("saved =", OUT)
