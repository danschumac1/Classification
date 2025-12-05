"""
python ./src/c_pipeline.py \
    --dataset har \
    --contrastive_input_path ./data/features/har/10-imgs_5-rounds_contrastive.json

"""

# GENERIC
import os
import argparse
import json
import random
from typing import Tuple
import textwrap


# NEED INSTALLS
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

# CUSTOM
from utils.build_questions import LABEL_MAPPING
from utils.file_io import save_json
from utils.loaders import load_train_test
from utils.image_prompter import ImagePrompter, Prompt
from utils.loaders import Split
from utils.loggers import MasterLogger

# -------------------------------------------------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Contrastive summarization of features in time series images."
    )
    p.add_argument("--dataset", type=str, required=True, help="Name of the dataset to be used.")
    p.add_argument("--contrastive_input_path", type=str, required=True, help="Path to the contrastive input data.")
    p.add_argument("--print_to_console", type=int, default=1, choices=[0, 1], help="Whether to print logs to console.")
    p.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Name of the vision model to use.")
    p.add_argument("--normalize", action="store_true", help="Whether to normalize the data.")

    return p.parse_args()

def setup() -> Tuple[argparse.Namespace, MasterLogger, Split, Split, ImagePrompter]:
    args = parse_args()

    # Reproducibility
    random.seed(42)
    np.random.seed(42)

    # Resolve dataset + input/output folders
    args.input_folder = f"./data/samples/{args.dataset}"
    args.output_folder = f"./data/sample_generations/{args.dataset}/contrastive_generation/"
    os.makedirs(args.output_folder, exist_ok=True)

    # Directory for markdown prompt examples
    args.prompt_examples_dir = os.path.join(
        args.output_folder, "prompt_examples"
    )
    os.makedirs(args.prompt_examples_dir, exist_ok=True)

    # Logging
    logs_dir = "./logs/contrastive_generation/"
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{args.dataset}.log")
    logger = MasterLogger(
        log_path=log_path,
        init=True,
        clear=True,
        print_to_console=args.print_to_console,
    )

    # Load train + test with artifacts (e.g., metadata, label maps, etc.)
    _, test = load_train_test(
        args.input_folder,
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
    prompter.model_name = args.model_name

    return args, logger,  test, prompter


def main():
    args, logger, test, prompter = setup()
    # Load contrastive input data
    with open(args.contrastive_input_path, "r") as f:
        contrastive_data = json.load(f)
    
    logger.info(f"Processing dataset: {args.dataset}")

    # Map numeric labels -> class names (e.g., 0 -> "WALKING")
    unique_class_names = [
        LABEL_MAPPING[args.dataset.upper()][cls_id] 
        for cls_id in test.unique_classes
    ]
    logger.info(f"Unique classes in the dataset: {unique_class_names}")

    # Initialize nested dict:
    #   contrastive_dict[cls_a][cls_b] = list of summaries comparing cls_a vs cls_b
    contrastive_dict: dict[str, dict[str, list[str]]] = {
        cls_a: {cls_b: [] for cls_b in unique_class_names if cls_b != cls_a}
        for cls_a in unique_class_names
    }

    # Fill contrastive_dict from contrastive_data
    # Keys in contrastive_data are like "WALKING-WALKING_UPSTAIRS"
    for pair_key, summaries in contrastive_data.items():
        c1, c2 = pair_key.split("-")

        if c1 not in contrastive_dict or c2 not in contrastive_dict:
            # Skip any pair that involves a class not in this split
            continue

        # Ensure summaries is a list of strings
        if not isinstance(summaries, list):
            logger.warning(f"Expected list of summaries for key {pair_key}, got {type(summaries)}")
            continue

        # Add summaries in both directions so:
        #   contrastive_dict[c1][c2] and contrastive_dict[c2][c1]
        contrastive_dict[c1][c2].extend(summaries)
        contrastive_dict[c2][c1].extend(summaries)
    logger.info("Constructed contrastive summary dictionary.")
    logger.info(f"Contrastive dict keys: {list(contrastive_dict.keys())}")

    system_prompt = (
        "You will be given a class label and several contrastive summaries between that class label and another. "
        "Your task is to generate a verbose summary highlighting the distinctive features of the given class "
        "based on the provided contrastive information. Focus on the key differences that set this class apart from the others."
    )
    prompter.system_prompt = system_prompt

    generic_user_prompt_header = (
        "#" * 50
        + "\nGenerate a verbose summary describing the unique "
        "characteristics of the class **UNIQUE_CLASS** based on the "
        "following contrastive summaries:\n"
        + "#" * 50
    )

    saved_example_prompt = False
    final_summaries = {unique_class: "" for unique_class in unique_class_names} # to hold final summaries
    for unique_class in tqdm(unique_class_names, desc="Summarizing classes"):
        others_dict = contrastive_dict[unique_class]
        question = generic_user_prompt_header.replace("**UNIQUE_CLASS**", unique_class)

        for other_class_name, contrastive_summaries in others_dict.items():
            section_header = (
                f'{"-" * 50}'
                f"\n\nContrastive summaries between **{unique_class}** and **{other_class_name}**:\n"
                )
            question += section_header

            # Indent each summary by 4 spaces for readability
            for summary in contrastive_summaries:
                indented = textwrap.indent(f"- {summary.strip()}\n", "    ")
                question += indented

        user_kwargs = {"question": question}
        prompt_kwargs = {"user": user_kwargs}
        target = Prompt(**prompt_kwargs)


        messages = prompter.format_prompt([], target)

        # Save example prompt markdown for first class only
        if not saved_example_prompt:
            md_name = f"{args.dataset}_class_summary_prompt_example.md"
            md_path = os.path.join(args.prompt_examples_dir, md_name)
            prompter.export_messages_markdown(
                messages,
                out_md_path=md_path,
                save_images=False,
                images_dirname="images",
            )
            logger.info(f"Saved example class-summary prompt markdown to: {md_path}")
            saved_example_prompt = True

        result = prompter.get_completion(
            messages
            )
        
        # get_completion may return dict (single-prompt mode) in your setup
        if isinstance(result, dict):
            content = result.get("content", "").strip()
        else:
            content = str(result).strip()

        final_summaries[unique_class] = content
        # intermediate save
        save_json(f"{args.output_folder}/final_summaries.json", final_summaries)


    # -----------------------------------------------------------------------
    # Save final summaries to JSON
    # -----------------------------------------------------------------------
    logger.info("Saving final class-level summaries to JSON...")
    logger.info(f"✅ Finished. Results saved → {args.output_folder}/final_summaries.json")


if __name__ == "__main__":
    main()
