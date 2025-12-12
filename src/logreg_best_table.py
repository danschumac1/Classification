#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


TO USE:
    python src/logreg_best_table.py \
        --metric accuracy \
        --method knn


Make a pivot table of average macro_f1 per dataset × embed_types.

Reads:
    ./data/logreg_results.tsv   (from the earlier processing step)

Writes:
    ./data/logreg_pivot_macro_f1.tsv

Output format (TSV):

dataset   <embed_type_1>  <embed_type_2>  ...  <embed_type_k>
ctu       ...
emg       ...
har       ...
tee       ...
Grand Total  ...

Each cell is the mean macro_f1 for that dataset/embedding_type combination.
"""

import argparse
from pathlib import Path
import pandas as pd



def parse_args():
    parser = argparse.ArgumentParser(
        description="Make a pivot table of average macro_f1 per dataset × embed_types."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="macro_f1",
        choices=["macro_f1", "accuracy"],
        help="The metric column to pivot on (default: macro_f1).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "knn"],
        help="The classification method to filter on (default: logistic_regression).",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if args.method == "logistic_regression":
        IN_PATH = Path("./data/logreg_results.tsv")
    elif args.method == "knn":
        IN_PATH = Path("./data/knn_results.tsv")

    OUT_PATH = Path(f"./data/{args.method}_pivot_{args.metric}.tsv")
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    df = pd.read_csv(IN_PATH, sep="\t")

    # If method column exists, keep only the specified method
    if "method" in df.columns:
        df = df[df["method"] == args.method].copy()
        if df.empty:
            raise ValueError(f"No {args.method} rows found in input file.")

    # Ensure metric is numeric
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")

    # Pivot: rows = dataset, columns = embed_types, values = mean(metric)
    pivot = pd.pivot_table(
        df,
        index="dataset",
        columns="embed_types",
        values=args.metric,
        aggfunc="mean",
    )

    # Add Grand Total row: overall mean per embed_types
    grand = (
        df.groupby("embed_types")[args.metric]
        .mean()
        .reindex(pivot.columns)  # align column order
    )
    pivot.loc["Grand Total"] = grand

    # --------------------------------------
    # Compute average rank per embedding type
    # --------------------------------------
    # Use only real datasets for ranking (exclude Grand Total)
    # --------------------------------------
    # Compute average rank per embedding type
    # --------------------------------------
    # Use only real datasets for ranking (exclude Grand Total)
    metric_only = pivot.loc[pivot.index != "Grand Total"]

    # Rank per dataset (row); higher metric = better (rank 1 = best)
    ranks = metric_only.rank(axis=1, ascending=False, method="average")

    # Average rank per embedding type (column)
    avg_rank = ranks.mean(axis=0)  # index: embed_types

    # --------------------------------------
    # Build final table: methods as rows
    # --------------------------------------
    # After transpose: rows = embed_types, cols = datasets
    out = metric_only.T.copy()  # index: embed_types

    # Add Avg column using the Grand Total row (mean per embed type)
    out["Avg"] = grand[out.index].values

    # Add avgRank column by *name*, not position
    out["avgRank"] = out.index.to_series().map(avg_rank)

    # Name the index so we can convert it to a proper "method" column
    out.index.name = "method"

    # Now reset index so first column is literally "method"
    out = out.reset_index()  # columns: method, ctu, emg, har, tee, Avg, avgRank

    # Optional: sort by avgRank (best = smallest)
    out = out.sort_values(by="avgRank", ascending=True)

    # ------------------------------------------------------------
    # Inject VL-Time baseline row for accuracy metric (optional)
    # ------------------------------------------------------------
    if args.metric == "accuracy":
        # Make sure we match the existing column names exactly
        vl_time_row = {
            "method": "vl_time_baseline",
            "ctu": 0.667,
            "emg": 0.917,
            "har": 0.636,
            "tee": 0.643,
            "Avg": 0.716,
            "avgRank": 5.50,
        }

        vl_df = pd.DataFrame([vl_time_row])

        # Only keep the columns that already exist in 'out',
        # and align order to avoid creating extra columns.
        vl_df = vl_df[out.columns]

        # Append, then re-sort by avgRank
        out = pd.concat([out, vl_df], ignore_index=True)

        # ------------------------------------------------------------
    # Canonical method order (desired final table order)
    # ------------------------------------------------------------
    desired_order = [
        "t_d",
        "l_d",
        "v_d",
        "t_s",
        "l_s",
        "v_s",
        "t_d + t_s",
        "l_d + l_s",
        "v_d + v_s",
        "t_d + v_d",
        "t_s + l_s + v_s",
        "t_d + t_s + l_s + v_s",
        "vl_time_baseline"
    ]

    # ------------------------------------------------------------
    # Mapping from your raw embedding names → canonical names
    # ------------------------------------------------------------
    rename_map = {
        "ts_direct": "t_d",
        "letsc_direct": "l_d",
        "vis_direct": "v_d",

        "text_summary": "t_s",
        "letsc_summary": "l_s",
        "vis_summary": "v_s",

        # combos
        "ts_direct-text_summary": "t_d + t_s",
        "letsc_direct-letsc_summary": "l_d + l_s",
        "vis_direct-vis_summary": "v_d + v_s",

        "ts_direct-vis_direct": "t_d + v_d",
        "text_summary-letsc_summary-vis_summary": "t_s + l_s + v_s",

        "ts_direct-text_summary-letsc_summary-vis_summary":
            "t_d + t_s + l_s + v_s",

        # baseline
        "vl_time_baseline": "vl_time_baseline"
    }

    # ------------------------------------------------------------
    # Apply renaming to the "method" column
    # ------------------------------------------------------------
    out["method"] = out["method"].map(rename_map)

    # Drop any rows that failed to map (methods not in rename_map)
    out = out.dropna(subset=["method"])

    # ------------------------------------------------------------
    # Reorder according to desired order (keeping only present methods)
    # ------------------------------------------------------------
    present = [m for m in desired_order if m in out["method"].values]

    out = (
        out.set_index("method")
           .loc[present]
           .reset_index()
    )


    # --------------------------------------
    # Write TSV
    # --------------------------------------
    out.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"Wrote pivot table → {OUT_PATH}")
if __name__ == "__main__":
    main()
