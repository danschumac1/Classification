#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025-11-24
Author: Dan Schumacher

This script classifies directly using the contrastive summaries.

(Another file will use cosine similarities and then clarify with the contrastive summaries.)

How to run:
   python ./src/cf_classification_v1.py \
       --dataset har \
       --visualization_method line \
       --batch_size 20 \
       --temperature 0.0
"""

# --------------------------------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------------------------------
# STANDARD
import argparse
import json
import os
import random
import re
from typing import Dict, List, Tuple

# NEED INSTALLS
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm

# USER DEFINED
from utils.build_questions import (
    HELP_STRING,
    LEGEND_MAPPINGS,
    TITLE_MAPPINGS,
    X_MAPPINGS,
    Y_MAPPINGS,
)
from utils.image_prompter import ImagePrompter, Prompt
from utils.loaders import Split, load_train_test
from utils.loggers import MasterLogger
from utils.file_io import append_jsonl, load_json
from utils.visualization import plot_time_series


# --------------------------------------------------------------------------------------------------
# ARGPARSE
# --------------------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Using the contrastive summaries to do classification"
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to be used.",
    )
    p.add_argument(
        "--print_to_console",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to print logs to console.",
    )
    p.add_argument(
        "--normalize",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to normalize the data.",
    )
    p.add_argument(
        "--visualization_method",
        type=str,
        default="line",
        help="Visualization method for plot_time_series (e.g., 'line', 'spectrogram').",
    )
    p.add_argument(
        "--img_detail",
        type=str,
        choices=["auto", "low", "high"],
        default="auto",
        help="Image detail level for the vision model.",
    )
    # We set out_file in setup(), so this is optional and can be overridden if desired.
    p.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="Optional override for output JSONL file path.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the model.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for test examples.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------------------
def setup() -> Tuple[argparse.Namespace, MasterLogger, Split, Split, ImagePrompter]:
    args = parse_args()

    # Reproducibility
    random.seed(42)
    np.random.seed(42)

    # Resolve dataset + input/output folders
    args.data_in = f"./data/samples/{args.dataset}"
    args.features_in = f"./data/features/{args.dataset}"
    output_folder = f"./data/sample_generations/{args.dataset}/contrastive_classification/"
    os.makedirs(output_folder, exist_ok=True)

    # If user did not override out_file, set default
    if args.out_file is None:
        args.out_file = os.path.join(output_folder, "cf.jsonl")

    # Directory for markdown prompt examples
    args.prompt_examples_dir = os.path.join(output_folder, "prompt_examples")
    os.makedirs(args.prompt_examples_dir, exist_ok=True)

    # Logging
    logs_dir = "./logs/contrastive_classification/"
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{args.dataset}.log")
    logger = MasterLogger(
        log_path=log_path,
        init=True,
        clear=True,
        print_to_console=bool(args.print_to_console),
    )

    # Load train + test with artifacts (e.g., metadata, label maps, etc.)
    train, test = load_train_test(
        f"./data/samples/{args.dataset}",
        0,  # shots NA here
        mmap=False,
        attach_artifacts=True,
        normalize=bool(args.normalize),
    )
    logger.info(f"Loaded dataset: {test.dataset}")
    logger.info(f"Test size={len(test)}")

    # Ensure API key is loaded (ImagePrompter will also check)
    load_dotenv("./resources/.env")

    # Setup vision prompter
    prompter = ImagePrompter()
    prompter.model_name = "gpt-4o-mini"

    return args, logger, train, test, prompter


def dict_to_prompt_block(features: Dict[str, str]) -> str:
    """
    Turn a dict of {label: description} into a big text block
    to stuff into the prompt.
    """
    chunks: List[str] = []
    for k, v in features.items():
        chunks.append(
            f"LET'S ZOOM IN ON THE LABEL [{k.upper()}]\n{v}\n"
        )
    # Add some spacing between label blocks
    return "\n\n".join(chunks)


def load_feature_str(test: Split) -> str:
    feature_path = f"./data/features/{test.dataset}/contrastive_generation/final_summaries.json"
    features = load_json(feature_path)
    return dict_to_prompt_block(features)


def extract_letter_to_idx(model_output: str, mapping: Dict[str, int]) -> Tuple[str, int]:
    """
    Extract the predicted label (e.g., 'A', 'B', 'C', ...) from a model output string
    and map it to an integer id using `mapping`.

    Heuristics:
      1. Prefer explicit patterns like "The answer is [X]" or "Answer: [X]".
      2. If none found, fall back to the *last* bracketed token [X] in the text
         that appears in `mapping`.
      3. If nothing valid is found, return ("no valid letter found", -1).
    """
    # 1. Look for explicit "answer" patterns
    answer_patterns = [
        r"[Tt]he answer is\s*\[([^\[\]]+)\]",
        r"[Ff]inal answer\s*[:\-]?\s*\[([^\[\]]+)\]",
        r"[Aa]nswer\s*[:\-]?\s*\[([^\[\]]+)\]",
    ]
    for pat in answer_patterns:
        m = re.search(pat, model_output)
        if m:
            cand = m.group(1).strip()
            if cand in mapping:
                return cand, mapping[cand]

    # 2. Fallback: scan all bracketed tokens and use the LAST one that maps
    candidates = re.findall(r"\[([^\[\]]+)\]", model_output)
    for cand in reversed(candidates):
        cand = cand.strip()
        if cand in mapping:
            return cand, mapping[cand]

    # 3. Nothing matched
    return "no valid letter found", -1


# --------------------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args, logger, train, test, prompter = setup()
    features_str = load_feature_str(test)

    test_img_dir = f"./data/images/{args.dataset}/"
    os.makedirs(test_img_dir, exist_ok=True)
    # Convert arrays
    idx_tr, X_tr, y_tr = np.asarray(train.idx), np.asarray(train.X), np.asarray(train.y).ravel()
    idx_te, X_te, y_te = np.asarray(test.idx), np.asarray(test.X), np.asarray(test.y).ravel()
    general_question = test.general_question or train.general_question
    general_question = general_question.strip()
    fewshot_dir = f"./data/images/examples/{args.dataset}"


    # AUTO-CoT
    prompter.system_prompt = (
        "You are a time series classification expert."
        " You will be given a time series visualization, and an extensive description of each"
        " possible classification label, and a ground truth label."
        f' Your job is to use the visualization and provided class descriptions to explain why the provided answer is correct.'
        f' Think step by step and explain your reasoning.'
        " Your contribution is valuable and will later be used to help "
        f"{HELP_STRING[args.dataset.upper()]}."
    )

    example_indices = train.shot_idxs
    print(train.shot_idxs)
    exit()
    zero_shot_batch_requests = []
    example_image_paths = []

    # Handle None or empty few-shot set
    if example_indices is None or len(example_indices) == 0:
        logger.info("No few-shot examples selected.")
        example_indices = []           # <-- important fix!
        reasoning_list = []            # <-- also safe
    else:
        logger.info(f"Generating reasoning for {len(example_indices)} few-shot examples...")

        for ex_idx in example_indices:
            ex_X = X_tr[ex_idx]
            letter = train.label_maps['id_to_letter'][str(int(y_tr[ex_idx]))]

            user_kwargs = {
                "question": general_question,
                "correct_answer": f"The answer is [{letter}]",
                "feature_string": features_str
            }


            prompt_kwargs = {"user": user_kwargs}
            legends = LEGEND_MAPPINGS[args.dataset.upper()]
            example_image_path = plot_time_series(
                ex_X,
                method=args.visualization_method,
                title=TITLE_MAPPINGS[args.dataset.upper()],
                xlabs=X_MAPPINGS[args.dataset.upper()],
                ylabs=Y_MAPPINGS[args.dataset.upper()],
                legends=LEGEND_MAPPINGS[args.dataset.upper()],
                save_path=(
                    f"{fewshot_dir}/idx_{ex_idx}_"
                    f"{args.visualization_method[:5]}_{args.img_detail}.png"
                ),
                recreate=True,
            )
            prompt_kwargs["img_path"] = example_image_path
            example_image_paths.append(example_image_path)
            prompt_kwargs["img_detail"] = args.img_detail

            prompt = Prompt(**prompt_kwargs)
            msgs = prompter.format_prompt([], prompt)
            zero_shot_batch_requests.append(msgs)#!/usr/bin/env python3

        batch_results = prompter.get_completion(
            zero_shot_batch_requests,
            temperature=0.0,
        )
        reasoning_list = [r["content"] for r in batch_results]

    # update system prompt for main task
    prompter.system_prompt = (
        ' You will be given a multiple choice question, a time series visualization, and a detailed description of each class.'
        ' Your job is to use the visualization and the description to answer the multiple choice question.'
        ' Think step by step and explain your reasoning. Then, provide a final answer.'
        ' The final answer must use the wording "The answer is [x]" where x is the answer to the multiple choice question.'
        ' Be sure to encapsulate x in square brackets.'
        " Your contribution is valuable and will later be used to help "
        f"{HELP_STRING[args.dataset.upper()]}."
    )

    
    few_shot_examples = []
    for i, ex_idx in enumerate(example_indices):
        ex_X = X_tr[ex_idx]
        example_image_path = example_image_paths[i]
        # User fields must match the query structure
        user_kwargs = {
            "question": general_question,
            "features": features_str
        }

        reasoning = reasoning_list[i]
        letter = train.label_maps['id_to_letter'][str(int(y_tr[ex_idx]))]
        assistant_kwargs = {
            "answer": reasoning + f" The answer is [{letter}]"
        }

        prompt = Prompt(
            user=user_kwargs,
            assistant=assistant_kwargs,
            img_path=example_image_path,
            img_detail=args.img_detail,
        )
        few_shot_examples.append(prompt)



    # ----------------------------------------------------
    # Process test set in batches
    # ----------------------------------------------------
    saved_example_prompt = False
    running_acc = []
    for start_of_batch in tqdm(range(0, len(test), args.batch_size), desc="Processing batches"):
        batch = test[start_of_batch : start_of_batch + args.batch_size]
        
        logger.info(f"Batch rows: {start_of_batch} → {start_of_batch+len(batch)}")

        # ------------------------------------------------
        # 1. Build test queries (Prompt objects)
        # ------------------------------------------------

        queries: List[Prompt] = []
        meta_batch: List[Tuple[int, int]] = []  # (idx, gt)

        for row in batch:
            user_kwargs = {
                "question": general_question,
            }
            prompt_kwargs = {
                "user": user_kwargs,
            }

            prompt_kwargs["img_path"] = plot_time_series(
                row.X, 
                method=args.visualization_method,
                title=TITLE_MAPPINGS[args.dataset.upper()],
                xlabs=X_MAPPINGS[args.dataset.upper()],
                ylabs=Y_MAPPINGS[args.dataset.upper()],
                legends=LEGEND_MAPPINGS.get(f"{args.dataset.upper()}", None),
                save_path=f"{test_img_dir}/idx_{row.idx}_{args.visualization_method[:5]}_{args.img_detail}.png",
                recreate=True
                )
            prompt_kwargs["img_detail"] = args.img_detail

            queries.append(Prompt(**prompt_kwargs))
            # row.idx and row.y are usually 0-d arrays or scalars; we keep raw here and cast later
            meta_batch.append((row.idx, row.y))

        all_messages: List[List[Dict[str, any]]] = [
            prompter.format_prompt(few_shot_examples, q) for q in queries
        ]


        # Save ONE example prompt in Markdown for inspection
        if not saved_example_prompt and len(all_messages) > 0:
            md_dir = os.path.join(os.path.dirname(args.out_file), "prompt_examples")
            os.makedirs(md_dir, exist_ok=True)
            file_ext = os.path.basename(args.out_file).replace(".jsonl", ".md")
            md_path = os.path.join(md_dir, file_ext)

            prompter.export_prompt_markdown(
                examples=few_shot_examples,
                query=queries[0],
                out_md_path=md_path,
                save_images=False,
            )
            logger.info(f"Saved prompt markdown to: {md_path}")
            saved_example_prompt = True


        # ------------------------------
        # 3. Model inference
        # ------------------------------
        logger.info(f"Sending {len(all_messages)} test prompts to model...")
        results = prompter.get_completion(
            all_messages,
            temperature=args.temperature,
        )

        # get_completion returns a list in batch mode, but guard anyway
        if not isinstance(results, list):
            results = [results]

        # ------------------------------
        # 4. Write results to JSONL + track accuracy
        # ------------------------------
        for (idx_val, gt_val), result, messages in zip(
            meta_batch, results, all_messages
        ):
            out_text = result.get("content", "").strip()

            letter, pred = extract_letter_to_idx(
                out_text, test.label_maps["letter_to_id"]
            )

            gt_scalar = int(np.asarray(gt_val).item())
            correctness = gt_scalar == pred
            running_acc.append(correctness)

            line = {
                "idx": int(np.asarray(idx_val).item()),
                "gt": gt_scalar,
                "pred": int(pred),
                "letter": letter,
                "model_output": out_text,
            }

            append_jsonl(args.out_file, line)

        curr_acc = float(np.mean(running_acc))
        last_idx = int(np.asarray(meta_batch[-1][0]).item())
        logger.info(
            f"Current accuracy at idx {last_idx}:\t\t\t{curr_acc * 100:.2f}%"
        )

    logger.info(f"✅ Finished. Results saved → {args.out_file}")
