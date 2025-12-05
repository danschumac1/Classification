'''
2025-12-04
Author: Dan Schumacher
How to run:
   python ./src/knn.py \
    --dataset har \
    --embedding_type ts_direct \
    --k 5
'''



import argparse
import os

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils.file_io import append_jsonl
from utils.loaders import load_train_test
from eval import accuracy_score, f1_score

def extract_scalar(x):
    return x.item() if hasattr(x, "item") else x

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate generated answers.")
    # REQUIRED arguments
    p.add_argument(
        "--dataset", type=str, required=True,
        help="dataset to run knn on")
    p.add_argument(
        "--embedding_type", type=str, required=True,
        choices=[
            "ts_direct",
            "letsc_direct",
            "vis_direct",
            # "ts_special", # TODO
            "text_summary",
            "letsc_summary",
            "vis_summary"],
        help="type of embedding to use")
    p.add_argument(
        "--k", type=int, required=True,
        help="number of neighbors for KNN"
    )
    # OPTIONAL arguments
    p.add_argument(
        "--normalize", type=int, default=1, choices=[0, 1], help="whether to normalize data")
    return p.parse_args()


if __name__ == "__main__":
    #region SETUP
    # ---------------------------------------------------
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
    trn_path = f"./data/features/{args.dataset}/embeddings/{args.embedding_type}/train_embeddings.npy"
    test_path = trn_path.replace("train_embeddings.npy", "test_embeddings.npy")
    train_embeddings = np.load(trn_path)
    test_embeddings = np.load(test_path)
    
    # make sure same number of embeddings as samples
    assert train_embeddings.shape[0] == len(train), \
        "mismatch in number of train embeddings and samples"
    assert test_embeddings.shape[0] == len(test), \
        "mismatch in number of test embeddings and samples"
    assert train_embeddings.shape[1] == test_embeddings.shape[1], \
        "train and test embeddings must have the same feature dimension"

    outdir = f"./data/sample_generations/{args.dataset}/knn/{args.embedding_type}/"
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"k{args.k}.jsonl")
    with open(outpath, "w") as f:
        pass  # clear file if exists

    #endregion
    #region COMPUTE SIMILARITIES
    # ---------------------------------------------------
    knn = KNeighborsClassifier(n_neighbors=args.k, metric='cosine')
    y_train = np.array([sample.y for sample in train]).ravel()
    knn.fit(train_embeddings, y_train)

    
    #endregion
    #region CLASSIFY WITH KNN
    predictions = knn.predict(test_embeddings)

    #endregion
    #region SAVE RESULTS
    y_true = [extract_scalar(sample.y) for sample in test]
    y_pred = [extract_scalar(pred) for pred in predictions]

    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

    print(f"KNN Results (k={args.k}, embedding={args.embedding_type}):")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (macro): {f1:.4f}")

    for row, pred in zip(test, predictions):
        line = {
            "idx": extract_scalar(row.idx),
            "gt": extract_scalar(row.y),
            "pred": extract_scalar(pred),
        }

        
        append_jsonl(outpath, line)

    #endregion
    print(f"Saved KNN results to {outpath}")
