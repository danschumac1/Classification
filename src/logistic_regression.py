'''
2025-12-04
Author: Dan Schumacher
How to run:
   python ./src/logistic_regression.py \
    --dataset har \
    --embedding_types ts_direct
'''


import argparse
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
import itertools

from utils.file_io import append_jsonl
from utils.loaders import load_train_test
from eval import accuracy_score, f1_score


def extract_scalar(x):
    return x.item() if hasattr(x, "item") else x

EMBEDDING_TYPES = [
    "ts_direct",
    "letsc_direct",
    "vis_direct",
    # "ts_special", # TODO
    "text_summary",
    "letsc_summary",
    "vis_summary",
]

EMBEDDING_COMBOS = []
for r in range(2, len(EMBEDDING_TYPES) + 1):
    EMBEDDING_COMBOS.extend(itertools.combinations(EMBEDDING_TYPES, r))
EMBEDDING_COMBOS = ["-".join(combo) for combo in EMBEDDING_COMBOS]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Logistic regression baseline on embeddings.")
    # REQUIRED arguments
    p.add_argument(
        "--dataset", type=str, required=True,
        help="dataset to run logistic regression on",
    )
    p.add_argument(
        "--embedding_types", type=str, required=True,
        choices=EMBEDDING_TYPES + EMBEDDING_COMBOS,
        help="type of embedding to use",
    )
    # OPTIONAL arguments
    p.add_argument(
        "--normalize", type=int, default=1, choices=[0, 1],
        help="whether to normalize data in load_train_test",
    )
    p.add_argument(
        "--C", type=float, default=1.0,
        help="Inverse of regularization strength for LogisticRegression",
    )
    p.add_argument(
        "--max_iter", type=int, default=10000,
        help="Maximum number of iterations for LogisticRegression",
    )
    return p.parse_args()


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------
    args = parse_args()

    # LOAD ORIGINAL DATA
    train, test = load_train_test(
        f"./data/samples/{args.dataset}",
        0,  # shots NA here
        mmap=False,
        attach_artifacts=True,
        normalize=bool(args.normalize),
    )

    # LOAD EMBEDDINGS
    embed_types = args.embedding_types.split("-")
    all_train_embeddings = []
    all_test_embeddings = []
    for et in embed_types:
        assert et in EMBEDDING_TYPES, f"unknown embedding type: {et}"
        trn_path = f"./data/features/{args.dataset}/embeddings/{et}/train_embeddings.npy"
        test_path = trn_path.replace("train_embeddings.npy", "test_embeddings.npy")
        et_trn_embeds = np.load(trn_path)
        et_test_embeds = np.load(test_path)
        all_train_embeddings.append(et_trn_embeds)
        all_test_embeddings.append(et_test_embeds)

    train_embeddings = np.concatenate(all_train_embeddings, axis=1)
    test_embeddings = np.concatenate(all_test_embeddings, axis=1)

    # make sure same number of embeddings as samples
    assert train_embeddings.shape[0] == len(train), \
        "mismatch in number of train embeddings and samples"
    assert test_embeddings.shape[0] == len(test), \
        "mismatch in number of test embeddings and samples"
    assert train_embeddings.shape[1] == test_embeddings.shape[1], \
        "train and test embeddings must have the same feature dimension"

    outdir = f"./data/sample_generations/{args.dataset}/logistic_regression/{args.embedding_types}/"
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{str(args.C).replace('.', 'p')}.jsonl")
    # clear / create file
    with open(outpath, "w") as f:
        pass

    # ------------------------------------------------------------------
    # TRAIN LOGISTIC REGRESSION
    # ------------------------------------------------------------------
    y_train = np.array([sample.y for sample in train]).ravel()

    clf = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter
    )
    clf.fit(train_embeddings, y_train)

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------
    predictions = clf.predict(test_embeddings)

    # ------------------------------------------------------------------
    # EVAL + SAVE
    # ------------------------------------------------------------------
    y_true = [extract_scalar(sample.y) for sample in test]
    y_pred = [extract_scalar(pred) for pred in predictions]

    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

    print(f"logistic_regression Results (embedding={args.embedding_types}, C={args.C}):")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (macro): {f1:.4f}")

    for row, pred in zip(test, predictions):
        line = {
            "idx": extract_scalar(row.idx),
            "gt": extract_scalar(row.y),
            "pred": extract_scalar(pred),
        }
        append_jsonl(outpath, line)

    print(f"Saved Logistic Regression results to {outpath}")
