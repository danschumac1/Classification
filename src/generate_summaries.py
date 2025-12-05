#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025-11-20
Author: Dan Schumacher

How to run:
   python ./src/sf_pipeline.py --dataset emg
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
from utils.processing import letcs_transform
from utils.visualization import plot_time_series
from utils.build_questions import (
    HELP_STRING,
    TITLE_MAPPINGS,
    X_MAPPINGS,
    Y_MAPPINGS,
    LEGEND_MAPPINGS,
    get_dim_names,   # assuming you defined this there
)
from utils.file_io import append_jsonl, load_jsonl
from utils.loaders import Split, load_train_test
from utils.image_prompter import ImagePrompter, Prompt
from utils.loggers import MasterLogger

#endregion
#region HELPERS
# ----------------------------------------------------
# Helpers
# ----------------------------------------------------

def build_text_queries(
    batch: List,
    dataset: str,
    use_letsc: bool = False,
) -> Tuple[List[Prompt], List[Tuple[int, int]]]:
    """
    Build text-only queries.

    For each example:
      - Each dimension of row.X becomes a separate field in the user dict:
        e.g. time_series_x_axis, time_series_y_axis, ...
      - If use_letsc=False: each field is a JSON list of floats.
      - If use_letsc=True: each field is a LETS-C-encoded string.
    """
    queries: List[Prompt] = []
    meta_batch: List[Tuple[int, int]] = []

    dataset_key = dataset.upper()

    for row in batch:
        idx_val = int(np.asarray(row.idx).item())
        gt_val = int(np.asarray(row.y).item())

        # ------------------------------------------------------------
        # Normalize X shape to (T, V)
        # ------------------------------------------------------------
        X = np.asarray(row.X)

        # Optional: leave this print in while debugging shapes
        # print("X shape:", X.shape)

        if X.ndim == 3:
            # Expect shape (1, T, V)
            if X.shape[0] != 1:
                raise ValueError(f"Expected X.shape[0] == 1 for 3D input, got {X.shape}")
            X = X[0]                       # (1, T, V) -> (T, V)
        elif X.ndim == 2:
            pass                           # already (T, V) or (V, T)
        elif X.ndim == 1:
            X = X[:, None]                 # (T,) -> (T, 1)
        else:
            raise ValueError(f"Unsupported X.ndim={X.ndim} with shape {X.shape}")

        T, V = X.shape

        # If you still have legacy datasets where X might be (V, T),
        # you can keep this flip; otherwise you can remove it.
        if T < V:
            X = X.T
            T, V = X.shape

        # ------------------------------------------------------------
        # Dimension names (from LEGEND_MAPPINGS or fallback)
        # ------------------------------------------------------------
        dim_names = get_dim_names(dataset_key)

        question = (
            "Create a detailed textual description of this time series. "
            "Explain your reasoning."
        )

        user_payload: dict = {"question": question}

        for i, name in enumerate(dim_names):
            ts_1d = X[:, i].tolist()

            if use_letsc:
                ts_repr = letcs_transform(ts_1d, precision=3)
            else:
                ts_repr = json.dumps(ts_1d)

            user_payload[name] = ts_repr

        prompt = Prompt(
            user=user_payload,
        )

        queries.append(prompt)
        meta_batch.append((idx_val, gt_val))

    return queries, meta_batch


def build_vis_queries(
    batch: List, dataset: str, split_name: str, args, img_root: str
) -> Tuple[List[Prompt], List[Tuple[int, int]]]:
    """
    Build visualization-based queries:
      - Saves one PNG per row
      - Asks the model to describe the plot.
    """
    queries: List[Prompt] = []
    meta_batch: List[Tuple[int, int]] = []

    dataset_key = dataset.upper()

    for row in batch:
        idx_val = int(np.asarray(row.idx).item())
        gt_val = int(np.asarray(row.y).item())

        img_save_path = os.path.join(img_root, f"{idx_val}.png")

        _ = plot_time_series(
            row.X,
            method=args.vis_method,
            title=TITLE_MAPPINGS[dataset_key],
            xlabs=X_MAPPINGS[dataset_key],
            ylabs=Y_MAPPINGS[dataset_key],
            legends=LEGEND_MAPPINGS[dataset_key],
            save_path=img_save_path,
            recreate=True,
        )

        user_text = (
            "Create a detailed textual description of the time series "
            "using the visualization. Explain your reasoning."
        )

        prompt = Prompt(
            user={"question": user_text},
            img_path=img_save_path,
            img_detail="auto",
        )

        queries.append(prompt)
        meta_batch.append((idx_val, gt_val))

    return queries, meta_batch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def generate_summary_for_split(
    dataset: str,
    summary_type: str,
    split: Split,
    split_name: str,
    prompter: ImagePrompter,
    args: argparse.Namespace,
    logger: MasterLogger,
    output_path: str,
    existing_idxs: set[int] | None = None,
):
    """
    Generate summaries for a split and append each one to a JSONL file immediately.
    If existing_idxs is provided, rows whose idx is in that set are skipped.
    """
    if split_name not in ("train", "test"):
        raise ValueError("split_name must be 'train' or 'test'.")

    if existing_idxs is None:
        existing_idxs = set()

    saved_example_prompt = False

    for start in tqdm(
        range(0, len(split), args.batch_size),
        desc=f"Generating {split_name} {summary_type} summaries",
    ):
        batch = split[start : start + args.batch_size]

        # Filter out already-processed rows
        batch_to_do = []
        for row in batch:
            idx_val = int(np.asarray(row.idx).item())
            if idx_val in existing_idxs:
                logger.info(
                    f"Skipping {split_name} idx={idx_val} (already present in {output_path})"
                )
                continue
            batch_to_do.append(row)

        if not batch_to_do:
            logger.info(
                f"All rows in batch {start}–{start+len(batch)} already processed; skipping."
            )
            continue

        logger.info(
            f"{split_name} batch rows: {start} → {start + len(batch_to_do)} (after skipping existing)"
        )

        # Pick query builder
        if summary_type == "vis":
            img_root = f"./data/images/{dataset}/{split_name}/"
            ensure_dir(img_root)
            queries, meta_batch = build_vis_queries(
                batch=batch_to_do,
                dataset=dataset,
                split_name=split_name,
                args=args,
                img_root=img_root,
            )
        elif summary_type in ("text", "letsc_like"):
            use_letsc = (summary_type == "letsc_like")
            queries, meta_batch = build_text_queries(
                batch=batch_to_do,
                dataset=dataset,
                use_letsc=use_letsc,
            )
        else:
            raise ValueError(f"Unknown summary_type: {summary_type}")

        # Format messages
        all_messages = [prompter.format_prompt([], q) for q in queries]

        # Save one example prompt (only first new batch of train)
        if split_name == "train" and not saved_example_prompt and len(all_messages) > 0:
            md_dir = os.path.join(f"./data/features/{dataset}", "prompt_examples")
            ensure_dir(md_dir)
            md_path = os.path.join(md_dir, f"{summary_type}_example.md")
            prompter.export_prompt_markdown(
                examples=[], query=queries[0], out_md_path=md_path, save_images=False
            )
            logger.info(f"Saved train prompt markdown to: {md_path}")
            saved_example_prompt = True

        # Call the model
        logger.info(f"Sending {len(all_messages)} {split_name} prompts to model...")
        results = prompter.get_completion(
            all_messages,
            temperature=args.temperature,
        )
        if not isinstance(results, list):
            results = [results]

        # Extract model outputs & append
        for (idx_val, gt_val), r in zip(meta_batch, results):
            idx_val = int(idx_val)
            row_dict = {
                "idx": idx_val,
                "gt": int(gt_val),
                "summary": r["content"].strip(),
            }

            append_jsonl(output_path, row_dict)
            existing_idxs.add(idx_val)

            logger.info(f"Saved summary idx={idx_val} to {output_path}")



#endregion
#region MAIN
# ----------------------------------------------------
# Main
# ----------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate text summaries of time series for SF pipeline."
    )
    p.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g., emg.")
    p.add_argument(
        "--summary_type",
        type=str,
        default="text",
        nargs="?",
        choices=[
            "text",         # passes time series directly
            "vis",          # passes visualization of time series
            "letsc_like",   # passes formatted time series like LETS-C
        ],
        help="Type of summary to generate.",
    )
    p.add_argument(
        "--vis_method",
        type=str,
        default="line",
        choices=["line", "spectrogram"],
        help="Visualization method for time series. (doesn't matter what you pass in if not using 'vis' summaries)",
    )
    # Model + batching
    p.add_argument("--model_name", type=str, default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for prompts / API calls.",
    )
    # Data loading / normalization
    p.add_argument(
        "--normalize",
        type=int,
        choices=[0, 1],
        default=0,
        help="If set, normalizes the time series before visualization.",
    )
    # Logging & control
    p.add_argument("--print_to_console", type=int, default=1)
    p.add_argument("--read_first", type=int, default=0, choices=[0,1],
                   help="If set, reads existing summaries from disk before generating new ones.")
    return p.parse_args()


if __name__ == "__main__":
    #region SETUP
    # ---------------------------------------------------
    args = parse_args()
    args.input_folder = f"./data/samples/{args.dataset}"

    # DATA
    train, test = load_train_test(
        args.input_folder,
        0,  # shots NA here
        mmap=False,
        attach_artifacts=True,
        normalize=bool(args.normalize),
    )

    # PROMPTER
    load_dotenv("./resources/.env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in ./resources/.env")
    client = OpenAI(api_key=api_key)

    prompter = ImagePrompter()
    prompter.model_name = args.model_name
    ts_str = "time series visualization" if args.summary_type == "vis" else "time series"
    prompter.system_prompt = (
        f"You will be given a {ts_str}. "
        f"Your job is to create a detailed textual description that describes the {ts_str}. "
        "Think step by step and explain your reasoning. "
        f"Your contribution is valuable and will later be used to help "
        f"{HELP_STRING[args.dataset.upper()]}"
    )

    # LOGGER
    logger = MasterLogger(
        log_path=f"./logs/{args.dataset}/summaries/{args.summary_type}.log",
        init=True,
        clear=True,
        print_to_console=args.print_to_console,
    )

    outdir = f"./data/features/{args.dataset}/summaries/{args.summary_type}"
    ensure_dir(outdir)
    train_out = os.path.join(outdir, "train_summaries.jsonl")
    test_out  = os.path.join(outdir, "test_summaries.jsonl")

    #endregion
    #region MAIN LOGIC
    # ---------------------------------------------------
    # TRAIN descriptions (resume-safe)
    # ---------------------------------------------------
    if args.read_first and os.path.exists(train_out):
        logger.info(f"Reading existing TRAIN summaries from {train_out}")
        existing_train_rows = load_jsonl(train_out)
        existing_train_idxs = {int(row["idx"]) for row in existing_train_rows}
        logger.info(f"Found {len(existing_train_idxs)} existing TRAIN summaries.")
    else:
        existing_train_idxs = set()

    logger.info("Generating TRAIN descriptions (crash-safe + resume)...")
    generate_summary_for_split(
        dataset=args.dataset,
        summary_type=args.summary_type,
        split=train,
        split_name="train",
        prompter=prompter,
        args=args,
        logger=logger,
        output_path=train_out,
        existing_idxs=existing_train_idxs,
    )

    logger.info("Generating TEST descriptions (crash-safe)...")
    # ---------------------------------------------------
    # TEST descriptions (resume-safe)
    # ---------------------------------------------------
    if args.read_first and os.path.exists(test_out):
        logger.info(f"Reading existing TEST summaries from {test_out}")
        existing_test_rows = load_jsonl(test_out)
        existing_test_idxs = {int(row["idx"]) for row in existing_test_rows}
        logger.info(f"Found {len(existing_test_idxs)} existing TEST summaries.")
    else:
        existing_test_idxs = set()

    logger.info("Generating TEST descriptions (crash-safe + resume)...")
    generate_summary_for_split(
        dataset=args.dataset,
        summary_type=args.summary_type,
        split=test,
        split_name="test",
        prompter=prompter,
        args=args,
        logger=logger,
        output_path=test_out,
        existing_idxs=existing_test_idxs,
    )

    print("Done generating summaries.")

    # At this point you can pass `train_text_strs`, `train_labels`, `desc_texts`, `meta`
    # to your embedding + nearest-neighbor classification code.

#endregion
