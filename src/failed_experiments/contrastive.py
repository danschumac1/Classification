#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025-11-21
Author: Dan Schumacher

Contrastive summarization of time series classes using multimodal LLMs.

For each unordered pair of labels (A, B) in the training split, we:
  - Run `n_sample_rounds` independent sampling rounds.
  - In each round, sample up to `n_imgs` examples from class A and B.
  - Render each time series as an image.
  - Build a grouped multimodal prompt: one group per class.
  - Ask the vision model for a contrastive explanation of the differences.

The final JSON file has the structure:

{
  "CLASSA_CLASSB": [
    "contrastive summary round 1",
    "contrastive summary round 2",
    ...
  ],
  ...
}

How to run:
   python ./src/contrastive.py \
       --dataset har \
       --n_imgs 5 \
       --n_sample_rounds 5
"""

import argparse
import itertools
import os
import random
from typing import Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from utils.build_questions import (
    HELP_STRING,
    LABEL_MAPPING,
    TITLE_MAPPINGS,
    X_MAPPINGS,
    Y_MAPPINGS,
    LEGEND_MAPPINGS,
)
from utils.image_prompter import ImagePrompter
from utils.loaders import Split, load_train_test
from utils.loggers import MasterLogger
from utils.file_io import save_json
from utils.visualization import plot_time_series


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Contrastive summarization of features in time series images."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ctu", "emg", "har", "tee"],
        help="Dataset name (must match preprocessed samples under ./data/samples/).",
    )
    parser.add_argument(
        "--n_imgs",
        type=int,
        choices=[i for i in range(1, 11)],
        default=5,
        help="How many example images per class PER ROUND to include in the prompt.",
    )
    parser.add_argument(
        "--n_sample_rounds",
        type=int,
        default=5,
        help=(
            "How many independent sampling rounds per class-pair. "
            "Each round re-samples up to n_imgs examples per class and "
            "adds another contrastive summary to the list."
        ),
    )
    parser.add_argument(
        "--print_to_console",
        type=int,
        choices=[0, 1],
        default=1,
        help="When set to 1, logger.info(...) also prints to the console.",
    )
    parser.add_argument(
        "--normalize",
        type=int,
        choices=[0, 1],
        default=0,
        help="If set, normalizes the time series before visualization.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name for contrastive summaries.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the LLM.",
    )
    parser.add_argument(
        "--img_detail",
        type=str,
        choices=["auto", "low", "high"],
        default="auto",
        help="Image detail level for the vision model.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup() -> Tuple[argparse.Namespace, MasterLogger, Split, Split, ImagePrompter]:
    args = parse_args()

    # Reproducibility
    random.seed(42)
    np.random.seed(42)

    # Resolve dataset + input/output folders
    args.input_folder = f"./data/samples/{args.dataset}"
    contrastive_summaries_dir = f"./data/features/{args.dataset}/"
    os.makedirs(contrastive_summaries_dir, exist_ok=True)

    # JSON output encodes n_imgs and n_sample_rounds in the filename
    args.contrastive_summaries_path = os.path.join(
        contrastive_summaries_dir,
        f"{args.n_imgs}-imgs_{args.n_sample_rounds}-rounds_contrastive.json",
    )

    # Directory for markdown prompt examples
    args.prompt_examples_dir = os.path.join(
        contrastive_summaries_dir, "prompt_examples"
    )
    os.makedirs(args.prompt_examples_dir, exist_ok=True)

    # Logging
    logs_dir = "./logs/contrastive/"
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{args.dataset}.log")
    logger = MasterLogger(
        log_path=log_path,
        init=True,
        clear=True,
        print_to_console=args.print_to_console,
    )

    # Load train + test with artifacts (e.g., metadata, label maps, etc.)
    train, test = load_train_test(
        args.input_folder,
        0,  # shots NA here
        mmap=False,
        attach_artifacts=True,
        normalize=bool(args.normalize),
    )
    logger.info(f"Loaded dataset: {train.dataset}")
    logger.info(f"Train size={len(train)}, Test size={len(test)}")

    # Ensure API key is loaded (ImagePrompter will also check)
    load_dotenv("./resources/.env")

    # Setup vision prompter
    prompter = ImagePrompter()
    prompter.model_name = args.model_name

    return args, logger, train, test, prompter


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def sample_indices_for_label(split: Split, label_id: int, n_imgs: int) -> np.ndarray:
    """
    Sample up to n_imgs indices from split.y == label_id.

    Returns an array of indices (possibly smaller than n_imgs if the class is rare).
    """
    y = np.asarray(split.y).ravel()
    idxs = np.where(y == label_id)[0]
    if len(idxs) == 0:
        return np.array([], dtype=int)
    if len(idxs) <= n_imgs:
        return idxs
    return np.random.choice(idxs, size=n_imgs, replace=False)


def _norm_class_name(name: str) -> str:
    """Normalize class names so they are safe as dict keys / filenames."""
    return name.replace(" ", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# Contrastive generation (grouped layout + multiple rounds)
# ---------------------------------------------------------------------------

def generate_contrastive_descriptions_for_split(
    split: Split,
    split_name: str,
    prompter: ImagePrompter,
    args: argparse.Namespace,
    logger: MasterLogger,
) -> Dict[str, List[str]]:
    """
    Generate contrastive descriptions for all class pairs on a given split.

    For each unordered pair of labels (A, B), we run n_sample_rounds:
      - sample up to n_imgs examples from class A and class B
      - render those time series as images
      - build a grouped multimodal prompt
      - get a contrastive description from the LLM
    and append each description to data_dict["A_B"].

    Returns:
        Dict[str, List[str]]:
            {
              "CLASS1_CLASS2": [summary_round1, summary_round2, ...],
              ...
            }
    """
    assert split_name in ["train", "test"], (
        f"Split must be 'train' or 'test' but got {split_name}"
    )

    # Root where images for this split will live
    img_root_template = "./data/images/SPLIT/"
    img_root = img_root_template.replace("SPLIT", split_name)
    os.makedirs(img_root, exist_ok=True)

    # LABEL_MAPPING[DATASET] maps label_id -> human-readable class name
    raw_label_map = LABEL_MAPPING[args.dataset.upper()]
    label_map: Dict[int, str] = {int(k): v for k, v in raw_label_map.items()}

    label_ids = sorted(label_map.keys())
    logger.info(f"Label ids: {label_ids}")

    pairwise_combinations = list(itertools.combinations(label_ids, 2))
    logger.info(f"Number of class pairs: {len(pairwise_combinations)}")

    # Desired structure:
    #   { "CLASS1_CLASS2": [summary1, summary2, ...], ... }
    data_dict: Dict[str, List[str]] = {}

    X = np.asarray(split.X)
    saved_example_prompt = False

    # System prompt template (will be formatted per pair and round)
    generic_system_prompt = (
        "You will be given two groups of time series visualizations, each group "
        "corresponding to a different class.\n\n"
        "GROUP 1: {n1} images of class '{class1}'.\n"
        "GROUP 2: {n2} images of class '{class2}'.\n\n"
        "Carefully compare these groups. Identify features that belong to one class "
        "but not the other, focusing on overall level, variability, spikes, "
        "periodicity, drift, and other temporal structure. "
        "Think step by step and explain your reasoning. "
        "Your contribution is valuable and will later be used to help "
        f"{HELP_STRING[args.dataset.upper()]}."
    )

    for label1, label2 in tqdm(pairwise_combinations, desc="Class pairs"):
        class1_name = label_map[label1]
        class2_name = label_map[label2]

        k1 = _norm_class_name(class1_name)
        k2 = _norm_class_name(class2_name)
        pair_key = f"{k1}-{k2}"

        logger.info(
            f"Processing pair: {label1} ({class1_name}) vs {label2} ({class2_name})"
        )

        # Ensure key exists
        if pair_key not in data_dict:
            data_dict[pair_key] = []

        # Multiple independent sampling rounds for this pair
        for round_idx in range(args.n_sample_rounds):
            logger.info(
                f"  Round {round_idx + 1}/{args.n_sample_rounds} for pair {pair_key}"
            )

            # ---------------------------------------------------------------
            # 1. Sample indices for both classes
            # ---------------------------------------------------------------
            idxs1 = sample_indices_for_label(split, label1, args.n_imgs)
            idxs2 = sample_indices_for_label(split, label2, args.n_imgs)

            if len(idxs1) == 0 or len(idxs2) == 0:
                logger.info(
                    f"  Skipping pair ({label1}, {label2}) this round due to "
                    f"insufficient examples (len1={len(idxs1)}, len2={len(idxs2)})"
                )
                # If a class has no examples at all in this split, continuing further
                # rounds for this pair will not help, so break out of the rounds loop.
                if len(idxs1) == 0 or len(idxs2) == 0:
                    break
                continue

            # ---------------------------------------------------------------
            # 2. Create images for both classes (this round)
            # ---------------------------------------------------------------
            img_paths_class1: List[str] = []
            img_paths_class2: List[str] = []

            for i, idx in enumerate(idxs1):
                save_path = os.path.join(
                    img_root,
                    f"{int(idx)}_{i}_c{label1}_r{round_idx}.png",
                )
                img_path = plot_time_series(
                    X[idx],
                    method="line",
                    title=TITLE_MAPPINGS[args.dataset.upper()],
                    xlabs=X_MAPPINGS[args.dataset.upper()],
                    ylabs=Y_MAPPINGS[args.dataset.upper()],
                    legends=LEGEND_MAPPINGS.get(args.dataset.upper(), None),
                    save_path=save_path,
                    recreate=True,
                )
                img_paths_class1.append(img_path)

            for i, idx in enumerate(idxs2):
                save_path = os.path.join(
                    img_root,
                    f"{int(idx)}_{i}_c{label2}_r{round_idx}.png",
                )
                img_path = plot_time_series(
                    X[idx],
                    method="line",
                    title=TITLE_MAPPINGS[args.dataset.upper()],
                    xlabs=X_MAPPINGS[args.dataset.upper()],
                    ylabs=Y_MAPPINGS[args.dataset.upper()],
                    legends=LEGEND_MAPPINGS.get(args.dataset.upper(), None),
                    save_path=save_path,
                    recreate=True,
                )
                img_paths_class2.append(img_path)

            # ---------------------------------------------------------------
            # 3. Build grouped messages using explicit class names
            # ---------------------------------------------------------------
            n1 = len(img_paths_class1)
            n2 = len(img_paths_class2)

            # Specialize the system prompt for this pair / round
            sys_prompt = generic_system_prompt.format(
                class1=class1_name,
                class2=class2_name,
                n1=n1,
                n2=n2,
            )
            prompter.system_prompt = sys_prompt

            # Grouped layout: one group per class with a header and its images
            groups = [
                (class1_name, img_paths_class1),
                (class2_name, img_paths_class2),
            ]

            messages = prompter.build_grouped_image_messages(
                groups,
                extra_instruction=(
                    f"Now carefully compare the two groups. Identify patterns that appear "
                    f"in the {class1_name} images but not the {class2_name} images, and "
                    f"patterns that appear in the {class2_name} images but not the "
                    f"{class1_name} images. Think step by step."
                ),
                detail=args.img_detail,
            )

            # ---------------------------------------------------------------
            # 3.5 Save the FIRST grouped prompt as Markdown for inspection
            # ---------------------------------------------------------------
            if not saved_example_prompt:
                md_name = (
                    f"{split_name}_{args.n_imgs}-imgs_"
                    f"{args.n_sample_rounds}-rounds_contrastive_prompt.md"
                )
                md_path = os.path.join(args.prompt_examples_dir, md_name)
                prompter.export_messages_markdown(
                    messages,
                    out_md_path=md_path,
                    save_images=False,   # set True if you want copied image files
                    images_dirname="images",
                )
                logger.info(f"Saved example prompt markdown to: {md_path}")
                saved_example_prompt = True

            # ---------------------------------------------------------------
            # 4. Call the model
            # ---------------------------------------------------------------
            logger.info("  Sending contrastive prompt to model...")

            result = prompter.get_completion(
                messages,
                temperature=args.temperature,
            )
            # get_completion returns a dict in single-prompt mode
            text = result.get("content", "").strip()

            # ---------------------------------------------------------------
            # 5. Store result in the requested structure
            # ---------------------------------------------------------------
            data_dict[pair_key].append(text)

            # ---------------------------------------------------------------
            # 6. SAVE-AS-YOU-GO: dump full dict after each completion
            # ---------------------------------------------------------------
            save_json(args.contrastive_summaries_path, data_dict)
            logger.info(
                f"  Intermediate save → {args.contrastive_summaries_path} "
                f"(len={len(data_dict[pair_key])} for {pair_key})"
            )

    return data_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args, logger, train, test, prompter = setup()

    logger.info("Starting contrastive summarization on TRAIN split...")
    train_results = generate_contrastive_descriptions_for_split(
        train,
        "train",
        prompter,
        args,
        logger,
    )

    # Final save (already saved incrementally, but this is a clean final dump)
    logger.info("Saving final contrastive summaries to JSON...")
    save_json(args.contrastive_summaries_path, train_results)
    logger.info(f"✅ Finished. Results saved → {args.contrastive_summaries_path}")
