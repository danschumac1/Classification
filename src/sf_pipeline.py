#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025-11-20
Author: Dan Schumacher

How to run:
   python ./src/sf_pipeline.py --dataset emg

This script:
  1) Generates (or reloads) textual descriptions for TRAIN and TEST via vision model
  2) Embeds TRAIN descriptions
  3) Classifies TEST examples by nearest-neighbor in embedding space
"""
#region IMPORTS
import json
import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------------------------------
# Project imports
# ----------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.visualization import plot_time_series
from utils.build_questions import (
    HELP_STRING,
    TITLE_MAPPINGS,
    X_MAPPINGS,
    Y_MAPPINGS,
    LEGEND_MAPPINGS,
)
from utils.loaders import Split, load_train_test
from utils.file_io import append_jsonl, load_jsonl
from utils.image_prompter import ImagePrompter, Prompt
from utils.loggers import MasterLogger

#endregion
#region HELPERS
# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
def get_embeddings_batch(
    texts: List[str],
    client: OpenAI,
    model: str = "text-embedding-3-small",
) -> np.ndarray:
    """
    Embed a list of texts with OpenAI and return an array of
    shape (N, D) with row-wise L2 normalization.
    """
    if len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    resp = client.embeddings.create(
        model=model,
        input=texts,
    )
    embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    # normalize each row to unit length
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs = embs / norms
    return embs


# ----------------------------------------------------
# Argument parsing
# ----------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "End-to-end pipeline:\n"
            "  - Generate / load TRAIN + TEST visual descriptions\n"
            "  - Embed TRAIN descriptions\n"
            "  - Similarity-based classification of TEST."
        )
    )
    p.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g., emg.")
    p.add_argument(
        "--input_folder",
        type=str,
        default=None,
        help="Root folder for dataset. If not set, defaults to ./data/samples/{dataset}",
    )
    p.add_argument(
        "--visualization_method",
        type=str,
        default="line",
        choices=["line", "spectrogram"],
        help="Visualization method for time series.",
    )
    # Model + batching
    p.add_argument("--embedding_model", type=str, default="text-embedding-3-small")
    p.add_argument("--model_name", type=str, default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for vision prompts / API calls.",
    )
    # Data loading / normalization
    p.add_argument(
        "--normalize",
        type=int,
        choices=[0, 1],
        default=0,
        help="If set, normalizes the time series before visualization.",
    )
    # Classification
    p.add_argument(
        "--top_k",
        type=int,
        default=5,
        choices=[i for i in range(1,11)],
        help="Number of nearest neighbors for voting.",
    )
    # Logging & control
    p.add_argument("--print_to_console", type=int, default=1)
    p.add_argument(
        "--regen_train_desc",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, regenerate TRAIN descriptions even if they exist.",
    )
    p.add_argument(
        "--regen_test_desc",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, regenerate TEST descriptions even if they exist.",
    )
    return p.parse_args()


# ----------------------------------------------------
# Setup
# ----------------------------------------------------
def setup() -> Tuple[argparse.Namespace, MasterLogger, OpenAI, Split, Split, ImagePrompter]:
    args = parse_args()

    # Resolve dataset + input folder
    dataset = args.dataset
    if args.input_folder is None:
        args.input_folder = f"./data/samples/{dataset}"

    # Logging
    logs_dir = "./logs/sf_pipeline/"
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{dataset}.log")

    logger = MasterLogger(
        log_path=log_path,
        init=True,
        clear=True,
        print_to_console=args.print_to_console,
    )

    # Load train + test with artifacts
    train, test = load_train_test(
        args.input_folder,
        0,  # shots NA here
        mmap=False,
        attach_artifacts=True,
        normalize=bool(args.normalize),
    )

    logger.info(f"Loaded dataset: {train.dataset}")
    logger.info(f"Train size={len(train)}, Test size={len(test)}")

    # Setup OpenAI client (for embeddings)
    load_dotenv("./resources/.env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in ./resources/.env")
    client = OpenAI(api_key=api_key)

    # Setup vision prompter
    prompter = ImagePrompter()
    prompter.model_name = args.model_name
    prompter.system_prompt = (
        "You will be given a time series visualization."
        " Your job is to create a detailed textual description of the time series"
        " using the visualization."
        " Think step by step and explain your reasoning."
        f" Your contribution is valuable and will later be used to help {HELP_STRING[dataset.upper()]}."
    )

    return args, logger, client, train, test, prompter


# ----------------------------------------------------
# Description generation helpers
# ----------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def generate_descriptions_for_split(
    split: Split,
    dataset: str,
    split_name: str,
    prompter: ImagePrompter,
    args: argparse.Namespace,
    logger: MasterLogger,
) -> List[dict]:
    """
    Generate descriptions for all rows in a Split (train or test).
    Returns a list of dicts: {idx, gt, model_output}.
    """
    if split_name not in ("train", "test"):
        raise ValueError("split_name must be 'train' or 'test'.")

    img_root = f"./data/images/{dataset}/{split_name}/"
    ensure_dir(img_root)

    all_rows: List[dict] = []

    # For optional Markdown example dump (only for train)
    saved_example_prompt = False

    for start in tqdm(
        range(0, len(split), args.batch_size),
        desc=f"Generating {split_name} descriptions",
    ):
        batch = split[start : start + args.batch_size]
        logger.info(f"{split_name} batch rows: {start} → {start + len(batch)}")

        queries: List[Prompt] = []
        meta_batch: List[Tuple[int, int]] = []  # (idx, gt)

        for row in batch:
            idx_val = int(np.asarray(row.idx).item())
            gt_val = int(np.asarray(row.y).item())

            img_save_path = os.path.join(img_root, f"{idx_val}.png")
            _ = plot_time_series(
                row.X,
                method=args.visualization_method,
                title=TITLE_MAPPINGS[dataset.upper()],
                xlabs=X_MAPPINGS[dataset.upper()],
                ylabs=Y_MAPPINGS[dataset.upper()],
                legends=LEGEND_MAPPINGS[dataset.upper()],
                save_path=img_save_path,
                recreate=True,
            )

            prompt = Prompt(
                user={
                    "question": (
                        "Create a detailed textual description of the time series "
                        "using the visualization. Explain your reasoning."
                    )
                },
                img_path=img_save_path,
                img_detail="auto",
            )

            queries.append(prompt)
            meta_batch.append((idx_val, gt_val))

        all_messages = [prompter.format_prompt([], q) for q in queries]

        # Save one example prompt markdown (train only)
        if split_name == "train" and (not saved_example_prompt) and len(all_messages) > 0:
            md_dir = os.path.join(f"./data/features/{dataset}", "prompt_examples")
            ensure_dir(md_dir)
            md_path = os.path.join(md_dir, "simple.md")

            prompter.export_prompt_markdown(
                examples=[],
                query=queries[0],
                out_md_path=md_path,
                save_images=False,
            )
            logger.info(f"Saved train prompt markdown to: {md_path}")
            saved_example_prompt = True

        # Model inference
        logger.info(f"Sending {len(all_messages)} {split_name} prompts to model...")
        results = prompter.get_completion(
            all_messages,
            temperature=args.temperature,
        )

        if not isinstance(results, list):
            results = [results]

        batch_descs = [r["content"].strip() for r in results]

        for (idx_val, gt_val), text in zip(meta_batch, batch_descs):
            all_rows.append(
                {
                    "idx": int(idx_val),
                    "gt": int(gt_val),
                    "model_output": text,
                }
            )

    return all_rows

#endregion
#region MAIN

# ----------------------------------------------------
# Main
# ----------------------------------------------------
if __name__ == "__main__":
    args, logger, client, train, test, prompter = setup()
    dataset = args.dataset

    # ---------------------------------------------------
    # TRAIN descriptions: load or generate
    # ---------------------------------------------------
    train_feat_dir = f"./data/features/{dataset}/"
    ensure_dir(train_feat_dir)
    train_desc_path = os.path.join(train_feat_dir, "simple.jsonl")

    if args.regen_train_desc == 0 and os.path.exists(train_desc_path):
        logger.info(f"Loading TRAIN descriptions from {train_desc_path}")
        train_rows = load_jsonl(train_desc_path)
    else:
        logger.info("Generating TRAIN descriptions...")
        train_rows = generate_descriptions_for_split(
            train, dataset, "train", prompter, args, logger
        )
        # Save to JSONL
        logger.info(f"Saving TRAIN descriptions to {train_desc_path}")
        with open(train_desc_path, "w") as f:
            for row in train_rows:
                f.write(json.dumps(row) + "\n")

    # Build TRAIN arrays
    train_text_strs = np.array(
        [row["model_output"] for row in train_rows],
        dtype=object,
    )
    train_labels = np.array([row["gt"] for row in train_rows], dtype=int)
    train_idxs = np.array([row["idx"] for row in train_rows], dtype=int)

    n_train = len(train_text_strs)
    logger.info(f"Loaded {n_train} TRAIN feature strings.")

    # ---------------------------------------------------
    # Embed TRAIN descriptions once
    # ---------------------------------------------------
    logger.info(f"Embedding TRAIN texts using model: {args.embedding_model}")
    train_embs = get_embeddings_batch(
        train_text_strs.tolist(),
        client,
        model=args.embedding_model,
    )

    # ---------------------------------------------------
    # TEST descriptions: load or generate
    # ---------------------------------------------------
    sf_dir = f"./data/sample_generations/{dataset}/sf_classification/"
    ensure_dir(sf_dir)
    test_desc_path = os.path.join(sf_dir, "test_descs.jsonl")

    if args.regen_test_desc == 0 and os.path.exists(test_desc_path):
        logger.info(f"Loading TEST descriptions from {test_desc_path}")
        test_rows = load_jsonl(test_desc_path)
    else:
        logger.info("Generating TEST descriptions...")
        test_rows = generate_descriptions_for_split(
            test, dataset, "test", prompter, args, logger
        )
        logger.info(f"Saving TEST descriptions to {test_desc_path}")
        with open(test_desc_path, "w") as f:
            for row in test_rows:
                f.write(json.dumps(row) + "\n")

    desc_texts = [row["model_output"] for row in test_rows]
    meta = [(row["idx"], row["gt"]) for row in test_rows]

    # ---------------------------------------------------
    # Embed TEST descriptions
    # ---------------------------------------------------
    logger.info("Embedding TEST descriptions...")
    test_embs = get_embeddings_batch(desc_texts, client, model=args.embedding_model)

    # ---------------------------------------------------
    # Similarity-based classification
    # ---------------------------------------------------
    # Output file
    top_k_tag = f"0{args.top_k}" if args.top_k <10 else args.top_k
    outfile = os.path.join(sf_dir, f"top-{top_k_tag}.jsonl")
    # If a directory exists *with the same path*, remove it
    if os.path.isdir(outfile):
        import shutil

        shutil.rmtree(outfile)

    with open(outfile, "w") as _:
        pass

    logger.info(f"FINAL CLASSIFICATION OUTPUTS → {outfile}")

    correct = 0
    total = 0
    top_k = max(1, min(args.top_k, n_train))

    for (test_idx, gt), test_emb, desc_text in zip(meta, test_embs, desc_texts):
        # cos sims since embs are normalized
        sims = train_embs @ test_emb
        nn_pos = np.argsort(sims)[::-1][:top_k]

        nn_labels = train_labels[nn_pos]
        vals, counts = np.unique(nn_labels, return_counts=True)
        pred = int(vals[np.argmax(counts)])

        correct += int(pred == gt)
        total += 1

        line = {
            "idx": int(test_idx),
            "gt": int(gt),
            "pred": int(pred),
            "nearest_neighbors": train_idxs[nn_pos].tolist(),
            "nn_labels": nn_labels.tolist(),
            "sim_scores": sims[nn_pos].tolist(),
            "model_output": desc_text,
        }
        append_jsonl(outfile, line)

    acc = correct / total if total else 0.0
    logger.info(f"Similarity-based accuracy (k={top_k}): {acc*100:.2f}%")
    print(f"Similarity-based accuracy (k={top_k}): {acc*100:.2f}%")
#endregion
#region HELPERS
