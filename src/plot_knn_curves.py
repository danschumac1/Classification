"""
python ./src/plot_knn_curves.py
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_PATH = "./data/knn_results.tsv"
OUT_ROOT = Path("./data/images/knn")

DIRECT_METHODS = ["ts_direct", "letsc_direct", "vis_direct"]
SUMMARY_METHODS = ["text_summary", "letsc_summary", "vis_summary"]


def build_palette_and_order(unique_methods):
    """
    Same grouping as the bar chart:
      - direct: blue family
      - summary: red family
      - others: gray
    """
    direct_present = [m for m in DIRECT_METHODS if m in unique_methods]
    summary_present = [m for m in SUMMARY_METHODS if m in unique_methods]
    others = [m for m in unique_methods if m not in direct_present + summary_present]

    hue_order = direct_present + summary_present + others

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
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH, sep="\t")
    df_knn = df[df["method"] == "knn"].copy()
    if df_knn.empty:
        raise ValueError("No KNN rows (method == 'knn') found in knn_results.tsv")

    parsed = df_knn["mode"].str.extract(r"^(.*)-k(\d+)$")
    df_knn["embedding_type"] = parsed[0]
    df_knn["k"] = parsed[1].astype(int)
    df_knn["macro_f1"] = pd.to_numeric(df_knn["macro_f1"], errors="coerce")

    all_methods = sorted(df_knn["embedding_type"].unique())
    hue_order, palette = build_palette_and_order(all_methods)

    sns.set(style="whitegrid")

    for dataset in sorted(df_knn["dataset"].unique()):
        sub = df_knn[df_knn["dataset"] == dataset].copy()
        if sub.empty:
            continue

        plt.figure(figsize=(7, 4.5))

        ax = sns.lineplot(
            data=sub,
            x="k",
            y="macro_f1",
            hue="embedding_type",
            hue_order=hue_order,
            palette=palette,
            marker="o",
        )

        ax.set_title(f"KNN Macro-F1 vs k — {dataset.upper()}")
        ax.set_xlabel("k (number of neighbors)")
        ax.set_ylabel("Macro F1")

        desired_ks = [1, 3, 5, 7, 10]
        ticks = [k for k in desired_ks if k in sub["k"].unique()]
        ax.set_xticks(ticks)

        plt.legend(title="Embedding Type", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        out_path = OUT_ROOT / f"{dataset}_knn_f1_vs_k_rb.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"Saved plot for {dataset} → {out_path}")


if __name__ == "__main__":
    main()
