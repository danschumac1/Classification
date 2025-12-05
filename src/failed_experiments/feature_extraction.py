#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025-08-06
Author: Dan Schumacher

How to run:
    see ./bin/feature_extraction.sh
"""

from typing import Dict, List, Tuple
import os
import sys
import argparse
import random
import numpy as np
from tqdm import tqdm

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
from utils.file_io import append_jsonl
from utils.loggers import MasterLogger
from utils.image_prompter import Prompt, ImagePrompter


# ----------------------------------------------------
# Argument parsing
# ----------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Feature extraction via visual prompting (images + Prompt objects)."
    )
    p.add_argument("--input_folder", type=str, required=True)
    p.add_argument(
        "--visualization_method",
        type=str,
        default="line",
        choices=["line", "spectrogram"],
        help="Visualization method for time series.",
    )
    # Model + batching
    p.add_argument("--model_name", type=str, default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=4)
    # Data loading / normalization
    p.add_argument(
        "--normalize",
        type=int,
        choices=[0, 1],
        default=0,
        help="If set, normalizes the time series before visualization.",
    )
    # Logging
    p.add_argument("--print_to_console", type=int, default=1)
    return p.parse_args()


# ----------------------------------------------------
# Setup: load dataset + logger
# ----------------------------------------------------
def set_up() -> Tuple[Split, argparse.Namespace, MasterLogger]:
    args = parse_args()

    random.seed(42)
    np.random.seed(42)

    dataset = os.path.basename(os.path.normpath(args.input_folder))
    args.dataset = dataset

    # Logging
    logs_dir = "./logs/feature_extraction/"
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{dataset}.log")

    logger = MasterLogger(
        log_path=log_path,
        init=True,
        clear=True,
        print_to_console=args.print_to_console,
    )

    # Output file
    out_dir = f"./data/features/{dataset}/"
    os.makedirs(out_dir, exist_ok=True)
    args.out_file = os.path.join(out_dir, "simple.jsonl")
    logger.info(f"Clearing output file {args.out_file}")
    with open(args.out_file, "w") as _:
        pass

    # Load train with artifacts
    train, _ = load_train_test(
        args.input_folder,
        0, # shots NA for train
        mmap=False,
        attach_artifacts=True,
        normalize=bool(args.normalize),
    )

    logger.info(f"Loaded dataset: {train.dataset}")
    logger.info(f"Train size={len(train)}")
    logger.info(f"Output file: {args.out_file}")

    return train, args, logger


# ----------------------------------------------------
# Main execution
# ----------------------------------------------------
if __name__ == "__main__":
    saved_example_prompt = False
    train, args, logger = set_up()
    train = train[:101]  # clip to save compute

    # Convert to arrays (not strictly needed, but harmless)
    _, X_tr, y_tr = (
        np.asarray(train.idx),
        np.asarray(train.X),
        np.asarray(train.y).ravel(),
    )

    prompter = ImagePrompter()
    prompter.model_name = args.model_name
    prompter.system_prompt = (
        f"You will be given a time series visualization"
        f" Your job is to create a detailed textual description of the time series"
        f' using the visualization.'
        " Think step by step and explain your reasoning."
        f" Your contribution is valuable and will later be used to help {HELP_STRING[args.dataset.upper()]}."
    )

    for start_of_batch in tqdm(range(0, len(train), args.batch_size), desc="Processing batches"):
        batch = train[start_of_batch : start_of_batch + args.batch_size]

        logger.info(f"Batch rows: {start_of_batch} → {start_of_batch + len(batch)}")

        queries: List[Prompt] = []
        meta_batch: List[Tuple[int, int]] = []  # (idx, gt)

        for row in batch:
            user_kwargs = {
                "question": (
                    "Create a detailed textual description of the time series using the "
                    "visualization. Explain your reasoning."
                ),
            }
            prompt_kwargs = {"user": user_kwargs}

            img_save_path = f"./data/images/{args.dataset}/train/{row.idx}.png"
            path_to_plot = plot_time_series(
                row.X,
                method=args.visualization_method,
                title=TITLE_MAPPINGS[args.dataset.upper()],
                xlabs=X_MAPPINGS[args.dataset.upper()],
                ylabs=Y_MAPPINGS[args.dataset.upper()],
                legends=LEGEND_MAPPINGS[args.dataset.upper()],
                save_path=img_save_path,
                recreate=True,
            )

            # attach image path to the Prompt so ImagePrompter can include it
            prompt_kwargs["img_path"] = img_save_path
            prompt_kwargs["img_detail"] = "auto"

            queries.append(Prompt(**prompt_kwargs))
            # store scalar idx and gt
            meta_batch.append((int(np.asarray(row.idx).item()),
                   int(np.asarray(row.y).item())))


        all_messages: List[List[Dict[str, object]]] = [
            prompter.format_prompt([], q) for q in queries
        ]

        # Save ONE example prompt in Markdown for inspection
        if not saved_example_prompt and len(all_messages) > 0:
            md_dir = os.path.join(os.path.dirname(args.out_file), "prompt_examples")
            os.makedirs(md_dir, exist_ok=True)
            file_ext = os.path.basename(args.out_file).replace(".jsonl", ".md")
            md_path = os.path.join(md_dir, file_ext)

            prompter.export_prompt_markdown(
                examples=[],  # no few-shots in this script
                query=queries[0],
                out_md_path=md_path,
                save_images=False,
            )
            logger.info(f"Saved prompt markdown to: {md_path}")
            saved_example_prompt = True

        # Model inference
        logger.info(f"Sending {len(all_messages)} train prompts to model...")
        results = prompter.get_completion(
            all_messages,
            temperature=args.temperature,
        )

        if not isinstance(results, list):
            results = [results]

        # Write results to JSONL
        for (idx_val, y_val), result in zip(meta_batch, results):
            out_text = result.get("content", "").strip()
            line = {
                "idx": int(np.asarray(idx_val).reshape(-1)[0]),
                "gt": int(np.asarray(y_val).reshape(-1)[0]),
                "model_output": out_text,
            }
            append_jsonl(args.out_file, line)

    logger.info(f"✅ Finished. Results saved → {args.out_file}")
