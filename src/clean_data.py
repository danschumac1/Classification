"""
2025-11-19
Author: Dan Schumacher

Convert raw .ts time-series datasets into reproducible,
clean NumPy splits for downstream TSQA and visual prompting.

How to run:
    python ./src/clean_data.py --dataset har
"""

import os
import json
import argparse
from typing import Any, Dict, List, Tuple
import numpy as np
from sktime.datasets import load_from_tsfile

from utils.build_questions import (
    LABEL_MAPPING, _letters, _sort_key_for_label_id, build_question_text
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset cleaning."""
    parser = argparse.ArgumentParser(description="Convert .ts time-series dataset to clean NumPy format.")
    parser.add_argument(
        "--dataset",
        choices=["ctu", "emg", "har", "tee"],
        required=True,
        help="Dataset name (folder under data/raw_data/).",
    )
    parser.add_argument(
        "--train_sample_length",
        type=int,
        default=500,
        help="Number of training samples to include in the SMALL 'sample' split.",
    )
    parser.add_argument(
        "--test_sample_length",
        type=int,
        default=100,
        help="Number of test samples to include in the SMALL 'sample' split.",
    )
    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def ensure_directory_exists(directory_path: str) -> None:
    """Create directory if it does not already exist."""
    os.makedirs(directory_path, exist_ok=True)


def ts_dataframe_to_numpy(ts_dataframe: Any) -> np.ndarray:
    """
    Convert sktime's DataFrame-of-Series into a pure NumPy array.

    Parameters
    ----------
    ts_dataframe : Any
        A sktime panel-format dataframe where each cell contains a pd.Series.

    Returns
    -------
    np.ndarray
        Shape (num_samples, sequence_length, num_channels)
    """
    num_channels: int = ts_dataframe.shape[1]
    list_of_samples: List[np.ndarray] = []

    for _, row in ts_dataframe.iterrows():
        channels: List[np.ndarray] = [row.iloc[i].to_numpy() for i in range(num_channels)]
        sample: np.ndarray = np.stack(channels, axis=1)  # (T, D)
        list_of_samples.append(sample)

    return np.stack(list_of_samples, axis=0)


def build_label_letter_mappings(dataset_name: str) -> Tuple[Dict[int, str], Dict[str, int], Dict[int, str]]:
    """
    Construct mappings between:
    - class_id → letter ("A", "B", "C"...)
    - letter → class_id
    - class_id → human-readable class name

    All derived from LABEL_MAPPING defined in utils/build_questions.py.
    """
    dataset_key = dataset_name.strip().upper()
    class_id_to_name = LABEL_MAPPING[dataset_key]

    sorted_items = sorted(class_id_to_name.items(), key=lambda item: _sort_key_for_label_id(item[0]))
    class_id_to_letter = {cid: _letters(i + 1) for i, (cid, _) in enumerate(sorted_items)}
    letter_to_class_id = {ltr: cid for cid, ltr in class_id_to_letter.items()}

    return class_id_to_letter, letter_to_class_id, class_id_to_name


def save_split_folder(
    output_directory: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_shots: Dict[int, List[int]],
    general_question: str,
    label_maps: Dict[str, Dict[str, str]],
    dataset_statistics: List[str],
) -> None:
    """
    Save all artifacts for one dataset split (full or sample).

    Files written:
        train.npz
        test.npz
        class_shots.json
        general_question.txt
        label_maps.json
        data_statistics.txt
    """
    ensure_directory_exists(output_directory)

    # ---- TRAIN SPLIT ----
    np.savez(
        os.path.join(output_directory, "train.npz"),
        X_train=X_train,
        y_train=y_train,
    )

    # ---- TEST SPLIT ----
    np.savez(
        os.path.join(output_directory, "test.npz"),
        X_test=X_test,
        y_test=y_test,
    )

    # ---- FEW-SHOT SAMPLES (JSON) ----
    with open(os.path.join(output_directory, "class_shots.json"), "w") as f:
        json.dump(class_shots, f, indent=2)

    # ---- GENERAL QUESTION ----
    with open(os.path.join(output_directory, "general_question.txt"), "w") as f:
        f.write(general_question)

    # ---- LABEL MAPS ----
    with open(os.path.join(output_directory, "label_maps.json"), "w") as f:
        json.dump(label_maps, f, indent=2)

    # ---- DATA STATISTICS ----
    with open(os.path.join(output_directory, "data_statistics.txt"), "w") as f:
        for line in dataset_statistics:
            f.write(str(line) + "\n")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    """Main execution: load raw TS data, clean, split, save full+sample."""
    args = parse_args()
    dataset_name: str = args.dataset

    # -------------------------
    # Construct input paths
    # -------------------------
    train_ts_path = f"data/raw_data/{dataset_name}/{dataset_name.upper()}_TRAIN.ts"
    test_ts_path  = f"data/raw_data/{dataset_name}/{dataset_name.upper()}_TEST.ts"

    # -------------------------
    # Load raw SKTime .ts files
    # -------------------------
    train_ts_df, train_labels = load_from_tsfile(train_ts_path)
    test_ts_df, test_labels   = load_from_tsfile(test_ts_path)

    train_labels = train_labels.astype(int)
    test_labels  = test_labels.astype(int)

    # Force labels to start at 0 if necessary
    if train_labels.min() != 0:
        shift = train_labels.min()
        train_labels -= shift
        test_labels  -= shift

    # -------------------------
    # Convert X to NumPy
    # -------------------------
    X_train_full = ts_dataframe_to_numpy(train_ts_df)
    X_test_full  = ts_dataframe_to_numpy(test_ts_df)

    # -------------------------
    # Shuffle full dataset
    # -------------------------
    perm_train = np.random.permutation(len(train_labels))
    perm_test  = np.random.permutation(len(test_labels))

    X_train_full = X_train_full[perm_train]
    train_labels = train_labels[perm_train]

    X_test_full = X_test_full[perm_test]
    test_labels = test_labels[perm_test]

    # -------------------------
    # Build sample splits first
    # (few-shot selection must come from sample to avoid OOB)
    # -------------------------
    X_train_sample = X_train_full[: args.train_sample_length]
    y_train_sample = train_labels[: args.train_sample_length]

    X_test_sample = X_test_full[: args.test_sample_length]
    y_test_sample = test_labels[: args.test_sample_length]

    # -------------------------
    # Few-shot selection (from sample split)
    # -------------------------
    few_shot_indices: Dict[int, List[int]] = {}

    for label in np.unique(y_train_sample):
        candidate_idxs = np.where(y_train_sample == label)[0]
        selected_idxs = np.random.choice(
            candidate_idxs,
            size=min(5, len(candidate_idxs)),
            replace=False,
        )
        few_shot_indices[int(label)] = selected_idxs.tolist()

    # -------------------------
    # Build metadata
    # -------------------------
    general_question: str = build_question_text(dataset_name).strip()
    class_id_to_letter, letter_to_class_id, class_id_to_name = build_label_letter_mappings(dataset_name)

    label_maps: Dict[str, Dict[str, str]] = {
        "letter_to_id": {ltr: int(cid) for ltr, cid in letter_to_class_id.items()},
        "id_to_letter": {str(cid): ltr for cid, ltr in class_id_to_letter.items()},
        "id_to_name":   {str(cid): name for cid, name in class_id_to_name.items()},
    }

    dataset_statistics: List[str] = [
        f"Dataset: {dataset_name}",
        f"Train full: {len(train_labels)}",
        f"Test full:  {len(test_labels)}",
        f"Train sample: {len(y_train_sample)}",
        f"Test sample:  {len(y_test_sample)}",
        f"Train full shape: {X_train_full.shape}",
        f"Test full shape:  {X_test_full.shape}",
        f"Unique train labels: {np.unique(train_labels)}",
        f"Unique test labels:  {np.unique(test_labels)}",
        f"Few-shot indices: {few_shot_indices}",
    ]

    # -------------------------
    # Save FULL split
    # -------------------------
    full_output_dir = f"data/datasets/{dataset_name}"
    save_split_folder(
        output_directory=full_output_dir,
        X_train=X_train_full,
        y_train=train_labels,
        X_test=X_test_full,
        y_test=test_labels,
        class_shots=few_shot_indices,
        general_question=general_question,
        label_maps=label_maps,
        dataset_statistics=dataset_statistics,
    )

    # -------------------------
    # Save SAMPLE split
    # -------------------------
    sample_output_dir = f"data/samples/{dataset_name}"
    save_split_folder(
        output_directory=sample_output_dir,
        X_train=X_train_sample,
        y_train=y_train_sample,
        X_test=X_test_sample,
        y_test=y_test_sample,
        class_shots=few_shot_indices,
        general_question=general_question,
        label_maps=label_maps,
        dataset_statistics=dataset_statistics,
    )

    print(f"\n✔ Saved FULL split → {full_output_dir}")
    print(f"✔ Saved SAMPLE split → {sample_output_dir}\n")


if __name__ == "__main__":
    main()
