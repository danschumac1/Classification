'''
2025-12-11
Author: Dan Schumacher
How to run:
   python ./src/llamaEmbedv1.py --input_path path/to/file.jsonl
'''

import argparse
import json
import os

import numpy as np
from utils.build_questions import EXTRA_INFO_MAPPINGS, LEGEND_MAPPINGS, QUESTION_TAG, TASK_DESCRIPTION, TITLE_MAPPINGS, X_MAPPINGS, Y_MAPPINGS
from utils.llamaPrompter import LlamaVisionPrompter, VisPrompt
from utils.loaders import Split
from utils.visualization import plot_time_series

def build_classification_prompt(
    prompter: LlamaVisionPrompter,
    row: Split,
    *,
    dataset: str,
    img_dir: str,
    include_ts: bool = False,
    include_vis: bool = True,
    viz_method: str = "line",
) -> VisPrompt:


    # # SYSTEM PROMPT
    # system_prompt = (
    #     TASK_DESCRIPTION[dataset.upper()]
    #     + " You will be given a multiple choice question, the correct answer to that question,"
    #     " and a time series visualization. Your job is to use the time series visualization to"
    #     " explain why the provided answer is correct. Think step by step and explain your reasoning."
    #     " At the end of your explanation, restate the answer using the wording"
    #     ' \"The answer is [x]\" where x is the correct choice.'
    #     " Here is some additional information that may help you:\n"
    #     + EXTRA_INFO_MAPPINGS[dataset.upper()]
    # )

    # USER PROMPT
    general_question = row.general_question 
    general_question = general_question.strip()
    general_question = general_question + "\n\n" + QUESTION_TAG
















    # Optionally append raw TS as JSON blob
    if include_ts:
        ts_json = json.dumps(np.asarray(row.X).tolist())
        user_text += "\n\nHere are the raw time series values (JSON):\n" + ts_json
    
    letcs_transform_multivar(serialize_ts(
    
    image_path: str | None = None
    if include_vis:
        # Use a stable filename based on the row index
        idx_scalar = int(np.asarray(row.idx).item())
        image_path = os.path.join(img_dir, f"{idx_scalar}.png")

        plot_time_series(
            row.X,
            method=viz_method,
            title=TITLE_MAPPINGS[dataset.upper()],
            xlabs=X_MAPPINGS[dataset.upper()],
            ylabs=Y_MAPPINGS[dataset.upper()],
            legends=LEGEND_MAPPINGS.get(dataset.upper(), None),
            save_path=image_path,
            recreate=True,
        )

    return VisPrompt(
        image_path=image_path,
        user_text=user_text,
    )


def compile_prompts(
    prompter: LlamaVisionPrompter,
    rows: Split,
    *,
    dataset: str,
    question: str,
    img_dir: str,
    include_ts: bool = False,
    include_vis: bool = True,
    viz_method: str = "line",
) -> list[VisPrompt]:
    """
    Build a list of VisPrompt objects for a collection of rows (e.g., a batch).

    Example usage:
        batch_prompts = compile_prompts(
            prompter,
            batch_rows,
            dataset=args.dataset,
            question=general_question,
            img_dir=test_img_dir,
            include_ts=False,
            include_vis=True,
        )
    """
    os.makedirs(img_dir, exist_ok=True)

    prompts: list[VisPrompt] = []
    for row in rows:
        vp = build_prompt(
            prompter,
            row,
            dataset=dataset,
            question=question,
            img_dir=img_dir,
            include_ts=include_ts,
            include_vis=include_vis,
            viz_method=viz_method,
        )
        prompts.append(vp)

    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated answers.")
    # REQUIRED arguments
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to the generated file"
    )
    # OPTIONAL arguments
    # parser.add_argument("--verbose", action="store_true", help="Enable verbose logging") 
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Input: {args.input_path}")

if __name__ == "__main__":
    main()