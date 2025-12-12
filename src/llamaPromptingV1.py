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

import numpy as np
from tqdm import tqdm

from utils.loaders import Split, load_train_test
from utils.loggers import MasterLogger
from utils.llamaPrompter import LlamaVisionPrompter, VisPrompt
from utils.file_io import append_jsonl

from utils.prompt_builders import (
    build_prompt,
    build_classification_system_prompt,
    build_reasoning_system_prompt,
)


# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
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

    # Load train/test; loader attaches label_maps & shot_idxs
    train, test = load_train_test(
        input_folder=f"./data/samples/{args.dataset}",
        n_shots=args.n_shots,
    )

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

    # Directories (images will be handled inside build_prompt)
    fewshot_split_name = "train"
    test_split_name = "test"

    # Question used across dataset (RAW, no QUESTION_TAG here)
    general_question = test.general_question or train.general_question
    general_question = general_question.strip()
    assert general_question, "General Question not available"

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

        # Reasoning system prompt
        prompter.system_prompt = build_reasoning_system_prompt(args.dataset)

        reasoning_prompts: List[VisPrompt] = []

        for ex_idx in few_shot_indices:
            row_ex = train[ex_idx]
            y_val = int(np.asarray(row_ex.y).item())
            letter = train.label_maps["id_to_letter"][str(y_val)]

            # For reasoning, question includes the correct answer text
            reasoning_question = (
                general_question
                + f"\n\nThe correct answer is [{letter}]."
                " Explain why this is correct using the time series visualization."
            )

            reasoning_prompts.append(
                build_prompt(
                    row=row_ex,
                    split_name=fewshot_split_name,
                    dataset=args.dataset,
                    model=args.model,
                    general_question=reasoning_question,
                    include_ts=False,
                    include_LETSCLike=False,
                    include_vis=True,
                    assistant_msg="",        # no assistant for reasoning stage
                    viz_method="line",
                )
            )

        # Generate reasoning for few-shot examples
        logger.info("Generating few-shot reasoning examples...")
        reasonings = prompter.get_completion(reasoning_prompts, batch=True)

        # Build few-shot VisPrompts (user = plain question, assistant = reasoning+answer)
        for ex_idx, reasoning in zip(few_shot_indices, reasonings):
            row_ex = train[ex_idx]
            y_val = int(np.asarray(row_ex.y).item())
            letter = train.label_maps["id_to_letter"][str(y_val)]

            assistant_text = reasoning.strip()
            if f"[{letter}]" not in assistant_text:
                assistant_text = assistant_text + f" The answer is [{letter}]."

            vp = build_prompt(
                row=row_ex,
                split_name=fewshot_split_name,
                dataset=args.dataset,
                model=args.model,
                general_question=general_question,  # plain question here
                include_ts=False,
                include_LETSCLike=False,
                include_vis=True,
                assistant_msg=assistant_text,
                viz_method="line",
            )
            few_shot_vis_prompts.append(vp)

    # ------------------------------------------------
    # Main system prompt for test-time answering
    # ------------------------------------------------
    prompter.system_prompt = build_classification_system_prompt(args.dataset)
    logger.info(f"Model = {args.model}")
    logger.info(f"Few-shot examples used: {len(few_shot_vis_prompts)}")
    acc = 0
    total = 0

    for start_idx in tqdm(
        range(0, len(test), args.batch_size),
        desc="Batch prompting",
    ):
        batch_rows = test[start_idx : start_idx + args.batch_size]

        # Build query VisPrompts (just the query images + question text)
        query_prompts: List[VisPrompt] = []
        for row in batch_rows:
            vp = build_prompt(
                row=row,
                split_name=test_split_name,
                dataset=args.dataset,
                model=args.model,
                general_question=general_question,
                include_ts=False,
                include_LETSCLike=False,
                include_vis=True,
                assistant_msg="",     # no assistant text for queries
                viz_method="line",
            )
            query_prompts.append(vp)

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

        last_idx_scalar = int(np.asarray(batch_rows[-1].idx).item())
        logger.info(f"Running Acc after idx {last_idx_scalar}: {running_acc:.4f}")

    logger.info(f"✅ Finished. Results saved → {args.out_file}")


if __name__ == "__main__":
    main()
