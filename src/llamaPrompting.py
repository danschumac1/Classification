"""
2025-12-11
Author: Dan Schumacher
How to run:
   CUDA_VISIBLE_DEVICES=2 python ./src/llamaPrompting.py --dataset har
"""

import argparse
import os
import random
import re
from typing import Tuple, Dict, List

from tqdm import tqdm
import numpy as np

from utils.build_questions import (
    EXTRA_INFO_MAPPINGS,
    LEGEND_MAPPINGS,
    TASK_DESCRIPTION,
    TITLE_MAPPINGS,
    X_MAPPINGS,
    Y_MAPPINGS,
    QUESTION_TAG,
)
from utils.loaders import Split, load_train_test
from utils.loggers import MasterLogger
from utils.llamaPrompter import LlamaVisionPrompter, VisPrompt
from utils.visualization import plot_time_series
from utils.file_io import append_jsonl


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


def infer_model_stem(model_name: str) -> str:
    if "llama" in model_name.lower():
        if "instruct" in model_name.lower():
            return "llamaInstruct"
        return "llama"
    if "gpt" in model_name.lower():
        return "gpt"
    else:
        raise NotImplementedError(f"Model {model_name} not supported yet.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated answers (vision LLaMA).")
    # REQUIRED arguments
    parser.add_argument(
        "--dataset",
        choices=["ctu", "emg", "har", "tee"],
        type=str,
        required=True,
        help="Dataset name (must match ./data/samples/{dataset})",
    )
    parser.add_argument(
        "--model",
        choices=["meta-llama/Llama-3.2-11B-Vision-Instruct"],
        type=str,
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="Model to use for prompting",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing samples (used when n_shots == 0)",
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        default=5,
        help="Number of few-shot examples to use in the prompt (0 = zero-shot)",
    )
    return parser.parse_args()


def setup() -> Tuple[argparse.Namespace, Split, Split, MasterLogger, LlamaVisionPrompter]:
    random.seed(42)
    np.random.seed(42)

    args = parse_args()
    args.model_stem = infer_model_stem(args.model)

    prompter = LlamaVisionPrompter(model_id=args.model)

    # Load train/test; we assume loader attaches label_maps & shot_idxs
    train, test = load_train_test(input_folder=f"./data/samples/{args.dataset}", n_shots=args.n_shots)

    logs_dir = "./logs/prompting/"
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{args.dataset}.log")
    logger = MasterLogger(
        log_path=log_path,
        init=True,
        clear=True,
        print_to_console=True,
    )

    out_dir = f"./data/sample_generations/{args.model_stem}/{args.dataset}/visual_prompting/"
    os.makedirs(out_dir, exist_ok=True)
    args.out_file = os.path.join(out_dir, f"{args.n_shots}-shot.jsonl")
    logger.info(f"Clearing output file {args.out_file}")
    with open(args.out_file, "w") as _:
        pass

    return args, train, test, logger, prompter


# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():
    args, train, test, logger, prompter = setup()
    print(f"Output JSONL: {args.out_file}")

    # Directories
    fewshot_dir = f"./data/images/examples/{args.dataset}"
    test_img_dir = f"./data/images/{args.dataset}/test"
    os.makedirs(fewshot_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    # Question used across dataset
    general_question = test.general_question or train.general_question
    general_question = general_question.strip()
    assert general_question, "General Question not available"
    # Add standardized question suffix/tag
    general_question = general_question + "\n\n" + QUESTION_TAG

    # ------------------------------------------------
    # Few-shot example selection & reasoning stage
    # ------------------------------------------------
    few_shot_vis_prompts: List[VisPrompt] = []
    few_shot_indices = train.shot_idxs

    if few_shot_indices is None:
        logger.info("Zero-shot prompting (no few-shot examples).")
    else:
        logger.info(
            f"Using {len(few_shot_indices)} few-shot examples "
            f"({args.n_shots} per label × {len(train.unique_classes)} labels)"
        )

        # Reasoning system prompt (like old script)
        prompter.system_prompt = (
            TASK_DESCRIPTION[args.dataset.upper()]
            + " You will be given a multiple choice question, the correct answer to that question,"
            " and a time series visualization. Your job is to use the time series visualization to"
            " explain why the provided answer is correct. Think step by step and explain your reasoning."
            " At the end of your explanation, restate the answer using the wording"
            ' \"The answer is [x]\" where x is the correct choice.'
            " Here is some additional information that may help you:\n"
            + EXTRA_INFO_MAPPINGS[args.dataset.upper()]
        )

        reasoning_prompts: List[VisPrompt] = []
        example_image_paths: List[str] = []

        for ex_idx in few_shot_indices:
            ex_X = train.X[ex_idx]
            y_val = int(np.asarray(train.y[ex_idx]).item())
            letter = train.label_maps["id_to_letter"][str(y_val)]

            # Create and save example image
            img_path = plot_time_series(
                ex_X,
                method="line",
                title=TITLE_MAPPINGS[args.dataset.upper()],
                xlabs=X_MAPPINGS[args.dataset.upper()],
                ylabs=Y_MAPPINGS[args.dataset.upper()],
                legends=LEGEND_MAPPINGS.get(args.dataset.upper(), None),
                save_path=os.path.join(
                    fewshot_dir,
                    f"idx_{ex_idx}_line_example.png",
                ),
                recreate=True,
            )
            example_image_paths.append(img_path)

            user_text = (
                general_question
                + f"\n\nThe correct answer is [{letter}]."
                " Explain why this is correct using the time series visualization."
            )

            reasoning_prompts.append(
                VisPrompt(
                    image_path=img_path,
                    user_text=user_text,
                )
            )

        # Generate reasoning for few-shot examples
        logger.info("Generating few-shot reasoning examples...")
        reasonings = prompter.get_completion(reasoning_prompts, batch=True)

        # Build few-shot VisPrompts (user = question, assistant = reasoning+answer)
        for ex_idx, img_path, reasoning in zip(few_shot_indices, example_image_paths, reasonings):
            y_val = int(np.asarray(train.y[ex_idx]).item())
            letter = train.label_maps["id_to_letter"][str(y_val)]

            assistant_text = reasoning.strip()
            if f"[{letter}]" not in assistant_text:
                assistant_text = assistant_text + f" The answer is [{letter}]."

            vp = VisPrompt(
                image_path=img_path,
                user_text=general_question,
                assistant_text=assistant_text,
            )
            few_shot_vis_prompts.append(vp)

    # ------------------------------------------------
    # Main system prompt for test-time answering
    # ------------------------------------------------
    prompter.system_prompt = (
        TASK_DESCRIPTION[args.dataset.upper()]
        + " You will be given a multiple choice question and a time series visualization."
        " Your job is to use the time series visualization to answer the multiple choice question."
        " Think step by step and explain your reasoning. Then, provide a final answer."
        ' The final answer must use the wording "The answer is [x]" where x is the answer to the'
        " multiple choice question. Be sure to encapsulate x in square brackets."
        " \n\nHere is some additional information that may help you:\n"
        + EXTRA_INFO_MAPPINGS[args.dataset.upper()]
    )

    logger.info(f"Model = {args.model}")
    logger.info(f"Few-shot examples used: {len(few_shot_vis_prompts)}")

    # ------------------------------------------------
    # Process test set
    # ------------------------------------------------
    acc = 0
    total = 0

    for start_idx in tqdm(
        range(0, len(test), args.batch_size),
        desc="Batch prompting",
    ):
        batch_rows = test[start_idx : start_idx + args.batch_size]

        # Build query VisPrompts (just the query images + question text)
        query_prompts: List[VisPrompt] = []
        for j, row in enumerate(batch_rows):
            idx_global = start_idx + j
            img_path = os.path.join(test_img_dir, f"{idx_global}.png")

            plot_time_series(
                row.X,
                method="line",
                title=TITLE_MAPPINGS[args.dataset.upper()],
                xlabs=X_MAPPINGS[args.dataset.upper()],
                ylabs=Y_MAPPINGS[args.dataset.upper()],
                legends=LEGEND_MAPPINGS.get(args.dataset.upper(), None),
                save_path=img_path,
                recreate=True,
            )

            query_prompts.append(
                VisPrompt(
                    image_path=img_path,
                    user_text=general_question,
                )
            )

        # ----------------------------------------
        # Inference:
        #   - If no few-shot examples: use batched LLaMA call.
        #   - If few-shot examples: for each query, prepend examples and call one by one.
        # ----------------------------------------
        if len(few_shot_vis_prompts) == 0:
            # Zero-shot, true batching
            outputs = prompter.get_completion(query_prompts, batch=True)
        else:
            # Few-shot: build conversation per query (examples + query),
            # use single-sample mode per query.
            outputs = []
            for qp in query_prompts:
                convo_prompts = few_shot_vis_prompts + [qp]
                out = prompter.get_completion(convo_prompts, batch=False)
                outputs.append(out)

        # ----------------------------------------
        # Write results + track accuracy
        # ----------------------------------------
        for row, model_output in zip(batch_rows, outputs):
            idx_scalar = int(np.asarray(row.idx).item())
            y_scalar = int(np.asarray(row.y).item())

            letter, pred = extract_letter_to_idx(
                model_output, test.label_maps["letter_to_id"]
            )

            pred_scalar = int(pred)
            correct = int(pred_scalar == y_scalar) if pred_scalar >= 0 else 0

            total += 1
            acc += correct
            running_acc = acc / max(total, 1)

            line = {
                "idx": idx_scalar,
                "correct": correct,
                "gt": y_scalar,
                "pred": pred_scalar,
                "letter": letter,
                "model_output": model_output,
            }
            append_jsonl(args.out_file, line)

        logger.info(f"Running Acc after idx {int(np.asarray(batch_rows[-1].idx).item())}: {running_acc:.4f}")

    logger.info(f"✅ Finished. Results saved → {args.out_file}")


if __name__ == "__main__":
    main()
