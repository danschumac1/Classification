'''
python ./src/blah.py \
    --dataset tee
'''

import argparse
import os
import re

from dotenv import load_dotenv
import numpy as np

from utils.build_questions import (
    HELP_STRING,
    LEGEND_MAPPINGS,
    TITLE_MAPPINGS,
    X_MAPPINGS,
    Y_MAPPINGS,
)
from utils.image_prompter import ImagePrompter, Prompt
from utils.loggers import MasterLogger
from utils.loaders import Split, load_train_test
from utils.file_io import append_jsonl, load_json, load_jsonl
from utils.visualization import plot_time_series


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Few-shot visual + text prompting for time-series classification."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., har, emg, ctu).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="Name of the chat/vision model to use.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="./logs/blah.log",
        help="Path to save logs.",
    )
    parser.add_argument(
        "--print_to_console",
        default=1,
        type=int,
        choices=[0, 1],
        help="Whether to print logs to console.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of queries to send in one batch to the model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model.",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
def setup(
    args: argparse.Namespace,
) -> tuple[ImagePrompter, MasterLogger, Split, Split, dict]:
    """
    Sets up the prompter, logger, and loads train/test data.

    Returns
    -------
    (prompter, logger, train, test, mappings)
    """
    # Ensure env is loaded (ImagePrompter also does this internally, but this is harmless)
    load_dotenv("./resources/.env")

    # Prompter (this will load OPENAI_API_KEY from ./resources/.env)
    prompter = ImagePrompter()
    prompter.model_name = args.model_name

    # Logger
    logger = MasterLogger(
        log_path=args.log_path,
        init=True,
        clear=True,
        print_to_console=bool(args.print_to_console),
    )

    # Data
    train, test = load_train_test(
        f"./data/samples/{args.dataset}",
        n_shots=0,
    )

    # Label mappings (id→name, id→letter, etc.)
    mappings = load_json(f"./data/samples/{args.dataset}/label_maps.json")

    return prompter, logger, train, test, mappings


# ----------------------------------------------------------------------
# Helper: parse class name from model output
# ----------------------------------------------------------------------
def extract_class_name(model_output: str, valid_names: list[str]) -> str:
    """
    Extract CLASS_NAME from a string like: 'The answer is WALKING'

    - Prefer explicit 'The answer is ...'
    - Fall back to first exact match of any valid_names in the output
    - Return 'UNKNOWN' if nothing found
    """
    # 1) explicit pattern
    m = re.search(r"[Tt]he answer is\s+([^\n\.]+)", model_output)
    if m:
        cand = m.group(1).strip()
        # strip trailing punctuation
        cand = cand.rstrip(" .")
        if cand in valid_names:
            return cand

    # 2) fallback: scan for any valid class name substring
    for name in valid_names:
        if name in model_output:
            return name

    return "UNKNOWN"


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    prompter, logger, train, test, mappings = setup(args)

    # OUT FILE (like your other script)
    out_dir = f"./data/sample_generations/{args.dataset}/fewshot_mm/"
    os.makedirs(out_dir, exist_ok=True)
    args.out_file = os.path.join(out_dir, "fewshot_mm.jsonl")
    logger.info(f"Clearing output file {args.out_file}")
    with open(args.out_file, "w") as _:
        pass

    # Question string shared across dataset
    general_question = train.general_question or test.general_question
    general_question = (general_question or "").strip()
    assert general_question, "general_question is empty – check your dataset."

    # ------------------------------------------------------------------
    # System prompt: image + summary + multiple-choice question
    # ------------------------------------------------------------------
    prompter.system_prompt = (
        "You will be given:\n"
        "  1. A multiple choice question about a time series,\n"
        "  2. A time series visualization (image),\n"
        "  3. A short natural language summary of the time series.\n\n"
        "Use BOTH the visualization and the summary to answer the question.\n"
        "Respond in a single line of the form:\n"
        "The answer is CLASS_NAME\n"
        "where CLASS_NAME is exactly one of the options provided in the question.\n\n"
        f"Your contribution is valuable and will later be used to help "
        f"{HELP_STRING[args.dataset.upper()]}."
    )

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model:   {args.model_name}")

    # ----------------------------------------------------------------------
    # STEP 1: LOAD TRAIN/TEST SUMMARIES
    # ----------------------------------------------------------------------
    train_summaries_path = f"./data/features/{args.dataset}/simple.jsonl"
    test_summaries_path = (
        f"./data/sample_generations/{args.dataset}/sf_classification/top-5.jsonl"
    )

    train_summaries = load_jsonl(train_summaries_path)
    test_summaries = load_jsonl(test_summaries_path)

    logger.info(f"Loaded {len(train_summaries)} train summaries.")
    logger.info(f"Loaded {len(test_summaries)} test summaries.")

    # ----------------------------------------------------------------------
    # STEP 1D: FIND TOP-K SIMILAR TRAINING SUMMARIES FOR EACH TEST SUMMARY
    # (We trust `nearest_neighbors` in test_summaries as precomputed neighbors)
    # ----------------------------------------------------------------------
    similar_texts: list[list[str]] = []
    similar_labels: list[list[int]] = []

    for row in test:
        # Safely convert idx to Python int
        row_idx = int(np.asarray(row.idx).item())

        neighbor_indices = test_summaries[row_idx]["nearest_neighbors"]

        train_texts = [
            train_summaries[int(np.asarray(nn_idx).item())]["model_output"]
            for nn_idx in neighbor_indices
        ]
        train_gts = [
            train_summaries[int(np.asarray(nn_idx).item())]["gt"]
            for nn_idx in neighbor_indices
        ]

        assert len(train_texts) == len(train_gts)
        similar_texts.append(train_texts)
        similar_labels.append(train_gts)

    assert len(similar_texts) == len(test)
    assert len(similar_labels) == len(test)
    logger.info("Built neighbor-based similar_texts and similar_labels.")

    # ----------------------------------------------------------------------
    # STEP 2: BUILD PROMPT OBJECTS (QUERY + FEW-SHOT EXAMPLES PER TEST ITEM)
    # ----------------------------------------------------------------------
    few_shot_examples_per_query: list[list[Prompt]] = []  # shape: [num_test][num_neighbors]
    query_prompts: list[Prompt] = []

    example_images_root = f"./data/images/{args.dataset}/train"
    query_images_root = f"./data/images/{args.dataset}/test"
    os.makedirs(example_images_root, exist_ok=True)
    os.makedirs(query_images_root, exist_ok=True)

    # Gather list of valid class names for parsing
    if "id_to_name" in mappings:
        valid_names = [mappings["id_to_name"][k] for k in sorted(mappings["id_to_name"].keys(), key=int)]
    else:
        # fallback: use letters as "names"
        valid_names = [mappings["id_to_letter"][k] for k in sorted(mappings["id_to_letter"].keys(), key=int)]

    # Build for each test row
    for i, (row, sim_texts, sim_labels) in enumerate(
        zip(test, similar_texts, similar_labels)
    ):
        row_idx = int(np.asarray(row.idx).item())

        # ----------------- Query prompt ----------------- #
        query_img_path = plot_time_series(
            X=row.X,
            method="line",
            title=TITLE_MAPPINGS[args.dataset.upper()],
            xlabs=X_MAPPINGS[args.dataset.upper()],
            ylabs=Y_MAPPINGS[args.dataset.upper()],
            legends=LEGEND_MAPPINGS.get(args.dataset.upper(), None),
            save_path=os.path.join(query_images_root, f"{row_idx}.png"),
        )

        query_summary = test_summaries[row_idx]["model_output"]

        query_prompt = Prompt(
            user={
                "question": general_question,
                "summary": query_summary,
            },
            img_path=query_img_path,
        )
        query_prompts.append(query_prompt)

        # --------------- Few-shot examples --------------- #
        neighbor_indices = test_summaries[row_idx]["nearest_neighbors"]
        assert len(sim_texts) == len(sim_labels) == len(neighbor_indices)

        example_prompts: list[Prompt] = []

        for ex_text, ex_label, ex_idx in zip(sim_texts, sim_labels, neighbor_indices):
            ex_idx_int = int(np.asarray(ex_idx).item())

            example_img_path = plot_time_series(
                X=train[ex_idx_int].X,
                method="line",
                title=TITLE_MAPPINGS[args.dataset.upper()],
                xlabs=X_MAPPINGS[args.dataset.upper()],
                ylabs=Y_MAPPINGS[args.dataset.upper()],
                legends=LEGEND_MAPPINGS.get(args.dataset.upper(), None),
                save_path=os.path.join(example_images_root, f"{ex_idx_int}.png"),
            )

            # If available, map numeric label → class name or letter
            ex_label_int = int(ex_label)
            ex_label_str = str(ex_label_int)

            if "id_to_name" in mappings and ex_label_str in mappings["id_to_name"]:
                class_name = mappings["id_to_name"][ex_label_str]
            else:
                # fallback to letter
                class_name = mappings["id_to_letter"][ex_label_str]

            example_prompt = Prompt(
                user={
                    "question": general_question,
                    "summary": ex_text,
                },
                img_path=example_img_path,
                assistant={"answer": f"The answer is {class_name}"},
            )
            example_prompts.append(example_prompt)

        few_shot_examples_per_query.append(example_prompts)

    logger.info(
        f"Constructed {len(query_prompts)} query prompts "
        f"with neighbor-based few-shot examples."
    )

    # ----------------------------------------------------------------------
    # STEP 3: BATCHED COMPLETIONS OVER QUERIES + APPEND_JSONL LINE OUT
    # ----------------------------------------------------------------------
    saved_example_prompt = False
    batch_messages: list[list[dict]] = []
    batch_indices: list[int] = []
    running_acc: list[bool] = []

    # helper: name -> id
    name_to_id = mappings.get("name_to_id", None)
    if name_to_id is None and "id_to_name" in mappings:
        # build inverse just in case
        name_to_id = {v: int(k) for k, v in mappings["id_to_name"].items()}

    for q_idx, (query_prompt, example_prompts) in enumerate(
        zip(query_prompts, few_shot_examples_per_query)
    ):
        # Build OpenAI-style messages for this query
        messages = prompter.format_prompt(example_prompts, query_prompt)
        batch_messages.append(messages)
        batch_indices.append(q_idx)

        # Export a single prompt to Markdown for inspection (once)
        if not saved_example_prompt and example_prompts:
            md_dir = f"./data/sample_generations/{args.dataset}/prompt_examples"
            os.makedirs(md_dir, exist_ok=True)
            md_path = os.path.join(md_dir, "fewshot_example.md")

            prompter.export_prompt_markdown(
                examples=example_prompts,
                query=query_prompt,
                out_md_path=md_path,
                save_images=False,
            )
            logger.info(f"Saved example prompt markdown to: {md_path}")
            saved_example_prompt = True

        # If batch is full OR we're at the last query, send to model
        is_last = (q_idx == len(query_prompts) - 1)
        if len(batch_messages) == args.batch_size or is_last:
            logger.info(
                f"Sending batch of size {len(batch_messages)} to model "
                f"(up to query index {q_idx})."
            )

            results = prompter.get_completion(
                batch_messages,
                temperature=args.temperature,
            )

            # Ensure results is a list
            if not isinstance(results, list):
                results = [results]

            # ------------------------------
            # Line out per result + append_jsonl
            # ------------------------------
            for local_idx, result in zip(batch_indices, results):
                row = test[local_idx]
                row_idx = int(np.asarray(row.idx).item())
                gt_scalar = int(np.asarray(row.y).item())

                if "id_to_name" in mappings:
                    gt_name = mappings["id_to_name"][str(gt_scalar)]
                else:
                    gt_name = mappings["id_to_letter"][str(gt_scalar)]

                out_text = result.get("content", "").strip()

                # parse predicted class name
                pred_class_name = extract_class_name(out_text, valid_names)

                if name_to_id and pred_class_name in name_to_id:
                    pred_class_id = int(name_to_id[pred_class_name])
                else:
                    pred_class_id = -1  # unknown / parse fail

                correct = int(pred_class_id == gt_scalar)

                running_acc.append(bool(correct))

                line = {
                    "idx": row_idx,
                    "gt": gt_scalar,
                    "gt_class_name": gt_name,
                    "pred": pred_class_id,
                    "pred_class_name": pred_class_name,
                    "correct": correct,
                    "model_output": out_text,
                }

                append_jsonl(args.out_file, line)

            # simple running accuracy log (optional)
            curr_acc = float(np.mean(running_acc)) if running_acc else 0.0
            logger.info(f"Running accuracy over {len(running_acc)} items: {curr_acc * 100:.2f}%")

            # Reset batch
            batch_messages = []
            batch_indices = []

    logger.info(f"✅ Finished batched few-shot prompting run. Results → {args.out_file}")
