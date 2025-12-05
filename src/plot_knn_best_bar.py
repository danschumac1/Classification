
"""
python ./src/plot_knn_best_bar.py
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_PATH = "./data/knn_results.tsv"
OUT_PATH = Path("./data/images/knn/knn_best_f1_by_embedding_grouped_rb.png")

# ------------------------------------------------------------------
# Group definitions
# ------------------------------------------------------------------
DIRECT_METHODS = ["ts_direct", "letsc_direct", "vis_direct"]
SUMMARY_METHODS = ["text_summary", "letsc_summary", "vis_summary"]


def build_palette_and_order(unique_methods):
    """
    Build:
      - hue_order: list of embedding types in the desired plotting order
      - palette:   dict mapping embedding_type -> color

    Direct methods   → shades of blue
    Summary methods  → shades of red
    Others           → gray
    """
    # keep only the methods that actually appear
    direct_present = [m for m in DIRECT_METHODS if m in unique_methods]
    summary_present = [m for m in SUMMARY_METHODS if m in unique_methods]
    others = [m for m in unique_methods if m not in direct_present + summary_present]

    hue_order = direct_present + summary_present + others

    # colors
    blue_shades = sns.color_palette("Blues", n_colors=max(len(direct_present) + 1, 3))[1:]
    red_shades = sns.color_palette("Reds", n_colors=max(len(summary_present) + 1, 3))[1:]
    grey_shades = sns.color_palette("Greys", n_colors=max(len(others) + 1, 3))[1:]

    palette = {}

    for i, m in enumerate(direct_present):
        palette[m] = blue_shades[i % len(blue_shades)]

    for i, m in enumerate(summary_present):
        palette[m] = red_shades[i % len(red_shades)]

    for i, m in enumerate(others):
        palette[m] = grey_shades[i % len(grey_shades)]

    return hue_order, palette


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH, sep="\t")
    df_knn = df[df["method"] == "knn"].copy()
    if df_knn.empty:
        raise ValueError("No KNN entries found in knn_results.tsv.")

    # Parse "mode" like "ts_direct-k5"
    parsed = df_knn["mode"].str.extract(r"^(.*)-k(\d+)$")
    df_knn["embedding_type"] = parsed[0]
    df_knn["k"] = parsed[1].astype(int)
    df_knn["macro_f1"] = pd.to_numeric(df_knn["macro_f1"], errors="coerce")

    # Best F1 per dataset × embedding_type
    idx_best = (
        df_knn.groupby(["dataset", "embedding_type"])["macro_f1"]
        .idxmax()
        .dropna()
        .astype(int)
    )
    best_df = df_knn.loc[idx_best].copy()

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

    ax.set_title("Best KNN Macro-F1 per Embedding Type (Grouped by Direct vs Summary)")
    ax.set_ylabel("Best Macro F1")
    ax.set_xlabel("Dataset")

    # Annotate bars with the k that achieved this best F1
    for bar, (_, row) in zip(ax.patches, best_df.iterrows()):
        height = bar.get_height()
        k_val = int(row["k"])
        ax.annotate(
            f"k={k_val}",
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

    print(f"Saved grouped-color bar plot → {OUT_PATH}")


if __name__ == "__main__":
    main()
