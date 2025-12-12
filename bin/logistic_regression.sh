"""
python ./src/plot_logreg_best_bar.py

Make a bar plot of the best logistic regression macro-F1
for each embedding type and dataset.

Assumes a TSV with columns:
  dataset, method, mode, C, embed_types, accuracy, macro_f1, pred_path, timestamp

and logistic-regression rows have:
  method = "logistic_regression"
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_PATH = "./data/logreg_results.tsv"
OUT_PATH = Path("./data/images/logreg/logreg_best_f1_by_embedding_grouped_rb.png")

# ------------------------------------------------------------------
# Group definitions
# ------------------------------------------------------------------
DIRECT_METHODS = ["ts_direct", "letsc_direct", "vis_direct"]
SUMMARY_METHODS = ["text_summary", "letsc_summary", "vis_summary"]

# New: paired feature combos (purple)
PAIRED_METHODS = [
    "ts_direct-text_summary",
    "vis_direct-vis_summary",
    "letsc_direct-letsc_summary",
]

# New: misc combos (orange)
MISC_METHODS = [
    "ts_direct-vis_direct",
    "text_summary-letsc_summary-vis_summary",
    "ts_direct-text_summary-letsc_summary-vis_summary",
]


def build_palette_and_order(unique_methods):
    """
    Build:
      - hue_order: list of embedding types in the desired plotting order
      - palette:   dict mapping embedding_type -> color

    Direct methods   → shades of blue
    Summary methods  → shades of red
    Paired combos    → shades of purple
    Misc combos      → shades of orange
    Others           → gray
    """
    # keep only the methods that actually appear
    direct_present = [m for m in DIRECT_METHODS if m in unique_methods]
    summary_present = [m for m in SUMMARY_METHODS if m in unique_methods]
    paired_present = [m for m in PAIRED_METHODS if m in unique_methods]
    misc_present = [m for m in MISC_METHODS if m in unique_methods]

    # anything not in the above groups is "other"
    others = [
        m
        for m in unique_methods
        if m not in direct_present + summary_present + paired_present + misc_present
    ]

    # order: direct → summary → paired → misc → others
    hue_order = direct_present + summary_present + paired_present + misc_present + others

    # base palettes
    blue_shades = sns.color_palette("Blues", n_colors=max(len(direct_present) + 1, 3))[1:]
    red_shades = sns.color_palette("Reds", n_colors=max(len(summary_present) + 1, 3))[1:]
    purple_shades = sns.color_palette("Purples", n_colors=max(len(paired_present) + 1, 3))[1:]
    orange_shades = sns.color_palette("Oranges", n_colors=max(len(misc_present) + 1, 3))[1:]
    grey_shades = sns.color_palette("Greys", n_colors=max(len(others) + 1, 3))[1:]

    palette = {}

    for i, m in enumerate(direct_present):
        palette[m] = blue_shades[i % len(blue_shades)]

    for i, m in enumerate(summary_present):
        palette[m] = red_shades[i % len(red_shades)]

    for i, m in enumerate(paired_present):
        palette[m] = purple_shades[i % len(purple_shades)]

    for i, m in enumerate(misc_present):
        palette[m] = orange_shades[i % len(orange_shades)]

    for i, m in enumerate(others):
        palette[m] = grey_shades[i % len(grey_shades)]

    return hue_order, palette


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH, sep="\t")
    print(df)

    # Keep only logistic regression rows
    df_lr = df[df["method"] == "logistic_regression"].copy()
    if df_lr.empty:
        raise ValueError("No logistic_regression entries found in logreg_results.tsv.")

    # Ensure fields are in the right type
    df_lr["macro_f1"] = pd.to_numeric(df_lr["macro_f1"], errors="coerce")
    df_lr["C"] = pd.to_numeric(df_lr["C"], errors="coerce")

    # For compatibility with the KNN plotting logic, call it "embedding_type"
    df_lr["embedding_type"] = df_lr["embed_types"]

    # Best F1 per dataset × embedding_type
    idx_best = (
        df_lr.groupby(["dataset", "embedding_type"])["macro_f1"]
        .idxmax()
        .dropna()
        .astype(int)
    )
    best_df = df_lr.loc[idx_best].copy()

    # Build consistent ordering + palette
    embed_methods = sorted(best_df["embedding_type"].unique())
    hue_order, palette = build_palette_and_order(embed_methods)

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    ax = sns.barplot(
        data=best_df,
        x="dataset",
        y="macro_f1",
        hue="embedding_type",
        hue_order=hue_order,
        palette=palette,
    )

    ax.set_title(
        "Best Logistic Regression Macro-F1 per Embedding Type\n"
        "(Direct, Summary, Paired, Misc)"
    )
    ax.set_ylabel("Best Macro F1")
    ax.set_xlabel("Dataset")

    # Annotate bars with the C that achieved this best F1
    for bar, (_, row) in zip(ax.patches, best_df.iterrows()):
        height = bar.get_height()
        C_val = row["C"]
        # Format C nicely (e.g., 0.1, 1, 10)
        if float(C_val).is_integer():
            C_label = f"{int(C_val)}"
        else:
            C_label = f"{C_val:.2g}"

        ax.annotate(
            f"C={C_label}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.legend(title="Embedding Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)
    plt.close()

    print(f"Saved grouped-color LogReg bar plot → {OUT_PATH}")


if __name__ == "__main__":
    main()
