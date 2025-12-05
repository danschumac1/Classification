# ./src/eval.py
"""
2025-10-30
Author: Dan Schumacher

Evaluate a JSONL of predictions (one line per example):
  {"pred": 1, "gt": 0}       # idx optional: {"idx": 140, "pred": 1, "gt": 0}

Run:
  python ./src/eval.py \
    --pred_path ./data/generations/har/random_prior.jsonl \
    
Behavior:
- Computes Accuracy and Macro-F1.
- Appends a row to results.tsv (writes header if file doesn't exist or is empty).
- Tries to infer dataset/mode/seed from pred_path. You can override with flags.
"""

import argparse
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

from utils.file_io import append_row, ensure_header, load_jsonl

def parse_args():
    p = argparse.ArgumentParser(description="Append evaluation metrics to a TSV leaderboard.")
    p.add_argument("--pred_path", type=str, required=True,
                   help="Path to JSONL predictions with fields {pred, gt} per line.")
    p.add_argument("--results_path", type=str, default="./data/results.tsv",
                   help="TSV to append results into (header auto-written).")
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.pred_path)
    method = path.parent.name               # e.g., "visual_prompting"
    dataset = path.parent.parent.name       # e.g., "tee"
    mode = path.name.removesuffix(".jsonl") # e.g., "0-shot"
    # Read predictions

    results_list = load_jsonl(args.pred_path)
    gts = [line["gt"] for line in results_list]
    preds = [line["pred"] for line in results_list]

    # Metrics
    acc = float(accuracy_score(gts, preds))
    f1m = float(f1_score(gts, preds, average="macro"))

    # Infer fields; allow CLI overrides

    # Results row
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [
        "dataset",
        "method",
        "mode",
        "accuracy",
        "macro_f1",
        "pred_path",
        "timestamp"
    ]
    ensure_header(args.results_path, header)
    row = [
        dataset,
        method,
        mode,
        f"{acc:.6f}",
        f"{f1m:.6f}",
        args.pred_path,
        timestamp,
    ]
    append_row(args.results_path, row)


    # Console echo
    print(f"File                : {args.pred_path}")
    print(f"Dataset             : {dataset}")
    print(f"Method              : {method}")
    print(f"Mode                : {mode}")
    print(f"Acc                 : {acc:.4f} ({acc:.2%})")
    print(f"Macro F1            : {f1m:.4f} ({f1m:.2%})")
    print(f"Results Appended to : {args.results_path}")


if __name__ == "__main__":
    main()
