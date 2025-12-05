#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025-08-06
Author: Dan Schumacher

How to run:
    see ./bin/prompting.sh
"""

import json
from typing import Dict, List, Tuple
import os
import sys
import argparse
import random
import numpy as np
from tqdm import tqdm
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.visualization import plot_time_series
from utils.build_questions import (
    TASK_DESCRIPTION, TITLE_MAPPINGS, X_MAPPINGS, Y_MAPPINGS, LEGEND_MAPPINGS, EXTRA_INFO_MAPPINGS)
from utils.loaders import Split, load_train_test
from utils.file_io import append_jsonl
from utils.loggers import MasterLogger
from utils.image_prompter import Prompt, ImagePrompter


# ----------------------------------------------------
# Argument parsing
# ----------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Visual prompting using images + few-shot Prompt objects.")
    p.add_argument("--input_folder", type=str, required=True)
    p.add_argument("--n_shots", type=int, choices=[0,1,2,3,4,5], default=1)
    p.add_argument("--visualization_method", type=str, choices=["line", "spectrogram"])
    p.add_argument("--img_detail", type=str, choices=["auto","low", "high"], default="auto")
    p.add_argument("--normalize", type=int, choices=[0,1], default=0, help="If set, normalizes the time series before visualization.")
    p.add_argument("--include_ts", type=int, choices=[0,1], default=0, help="If set, includes the raw time series values in the prompt.")
    p.add_argument("--model_name", type=str, default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=2) # TODO make default larger
    p.add_argument("--print_to_console", type=int, default=1)
    p.add_argument("--clear_images", type=int, choices=[0,1], default=1, help="If set, clears existing images for the dataset.")
    p.add_argument("--stop_early_idx", type=int, default=-1, help="If >0, stops processing after this many test examples (for debugging).")
    return p.parse_args()



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
    #    This avoids picking [A] from the option list when the final line is "The answer is [B]".
    candidates = re.findall(r"\[([^\[\]]+)\]", model_output)
    for cand in reversed(candidates):
        cand = cand.strip()
        if cand in mapping:
            return cand, mapping[cand]

    # 3. Nothing matched
    return "no valid letter found", -1

# ----------------------------------------------------
# Setup: load dataset + logger
# ----------------------------------------------------
def set_up() -> Tuple[Split, Split, argparse.Namespace, MasterLogger]:
    args = parse_args()

    random.seed(42)
    np.random.seed(42)

    dataset = os.path.basename(os.path.normpath(args.input_folder))
    args.dataset = dataset

    if args.clear_images:
        img_dir = f"./data/images/{dataset}"
        if os.path.exists(img_dir):
            import shutil

            print(f"Clearing image directory: {img_dir}")
            shutil.rmtree(img_dir)

    # Logging
    logs_dir = "./logs/visual_prompting/"
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{dataset}.log")

    logger = MasterLogger(
        log_path=log_path,
        init=True,
        clear=True,
        print_to_console=args.print_to_console,
    )

    # ---- NEW: tags for normalization + TS presence ----
    norm_tag = "norm" if args.normalize else "raw"
    ts_tag = "ts" if args.include_ts else "no_ts"
    args.norm_tag = norm_tag
    args.ts_tag = ts_tag
    # TAG="${viz}_ID-${img_detail}_TS-${include_ts}_NORM-${NORM_TAG}"
    # OUT_JSONL="./data/sample_generations/${dataset}/visual_prompting/${TAG}.jsonl"
    tag = args.visualization_method[:5] + f"-{args.img_detail}-" + \
        f"{ts_tag}-" + f"{norm_tag}-" + f"{args.n_shots}_shot"
    out_dir = f"./data/sample_generations/{dataset}/visual_prompting/"
    os.makedirs(out_dir, exist_ok=True)
    args.out_file = os.path.join(out_dir, f"{tag}.jsonl")
    logger.info(f"Clearing output file {args.out_file}")
    with open(args.out_file, "w") as _:
        pass

    # Load train/test with artifacts
    train, test = load_train_test(
        args.input_folder,
        args.n_shots,
        mmap=False,
        attach_artifacts=True,
        normalize=bool(args.normalize),
    )
    test = test[: args.stop_early_idx] if args.stop_early_idx > 0 else test

    logger.info(f"Loaded dataset: {train.dataset}")
    logger.info(f"Train size={len(train)}, Test size={len(test)}")
    logger.info(f"Output file: {args.out_file}")

    return train, test, args, logger

# ----------------------------------------------------
# Main execution
# ----------------------------------------------------
if __name__ == "__main__":

    train, test, args, logger = set_up()
    # Few shot examples location
    fewshot_dir = f"./data/images/examples/{args.dataset}"
    test_img_dir = f"./data/images/{args.dataset}/"
    os.makedirs(fewshot_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

 
    # Convert arrays
    idx_tr, X_tr, y_tr = np.asarray(train.idx), np.asarray(train.X), np.asarray(train.y).ravel()
    idx_te, X_te, y_te = np.asarray(test.idx), np.asarray(test.X), np.asarray(test.y).ravel()

    # Question used across dataset
    general_question = test.general_question or train.general_question
    general_question = general_question.strip()
    assert general_question, "General Question not available"


    if args.include_ts:
        input_string = 'visualization and its corresponding values'
    else:
        input_string = 'visualization'


    prompter = ImagePrompter()
    prompter.model_name = args.model_name
    prompter.system_prompt = (
        TASK_DESCRIPTION[args.dataset.upper()] + \
        f' You will be given a multiple choice question, the correct answer to that question, and a time series {input_string}.'
        f' Your job is to use the time series {input_string} to explain why the provided answer is correct.'
        f' Think step by step and explain your reasoning.'
        f' Here is some additional information that may help you:\n'
        + EXTRA_INFO_MAPPINGS[args.dataset.upper()]
    )


    # -----------------------------
    # FEW-SHOT EXAMPLE SELECTION
    # -----------------------------
    example_indices = train.shot_idxs
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
            }

            if args.include_ts:
                user_kwargs["time_series"] = json.dumps(ex_X.tolist())

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
            zero_shot_batch_requests.append(msgs)

        batch_results = prompter.get_completion(
            zero_shot_batch_requests,
            temperature=0.0,
        )
        reasoning_list = [r["content"] for r in batch_results]

    # update system prompt for main task
    prompter.system_prompt = (
        TASK_DESCRIPTION[args.dataset.upper()] + \
        f' You will be given a multiple choice question and a time series {input_string}.'
        f' Your job is to use the time series {input_string} to answer the multiple choice question.'
        f' Think step by step and explain your reasoning. Then, provide a final answer.'
        f' The final answer must use the wording "The answer is [x]" where x is the answer to the multiple choice question.'
        f' Be sure to encapsulate x in square brackets.'
        f' \n\nHere is some additional information that may help you:\n'
        + EXTRA_INFO_MAPPINGS[args.dataset.upper()]
    )

    
    few_shot_examples = []
    for i, ex_idx in enumerate(example_indices):
        ex_X = X_tr[ex_idx]
        example_image_path = example_image_paths[i]
        # User fields must match the query structure
        user_kwargs = {
            "question": general_question,
        }
        if args.include_ts:
            user_kwargs["time_series"] = json.dumps(ex_X.tolist())

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

    logger.info(f"Model = {args.model_name}")


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
            if args.include_ts:
                user_kwargs["time_series"] = json.dumps(row.X.tolist())
            
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
