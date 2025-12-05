'''
2025-11-25
Author: Dan Schumacher
How to run:
   python ./src/pipe.py \
    --dataset har
'''
import json
from typing import Any
from utils.full_pipe.setup import setup, parse_pipe_args    
from utils.full_pipe.generation import batch_embed, generate_summaries
from utils.full_pipe.io import load_summary, save_results
from utils.full_pipe.processing import build_nn_packagedtsdata, build_prompts, build_summary_prompts

SUMMARY_QUESTION =\
"""
...
"""

CLASSIFICATION_QUESTION =\
"""
...
"""

def main():
    args = parse_pipe_args()
    prompter, logger, train, test, mapping = setup(args)
    with open(f"./data/samples/{args.dataset}/label_maps.json", "r") as fi:
        mappings = json.load(fi)
    summary_path  = f"./data/features/{args.dataset}/summaries/"

    # Step 1 | generate or load the appropriate train & test summaries
    if args.load_summary:   # load the summary
        train_summaries, test_summaries = load_summary(
            summary_path, args.summary_type, summary_path)
    else:                   # generate the summary
        train_prompts, test_prompts = build_summary_prompts(train, test, SUMMARY_QUESTION, args.dataset)
        train_summaries, test_summaries = generate_summaries(
            train_prompts, test_prompts, prompter, args.summary_type, summary_path, args.batch_size)

    # Step 2 | Embed train and test summaries. Find the top-k most similar train examples for each test query
    train_embeddings,test_embeddings = batch_embed(train_summaries, test_summaries)

    nn_packages = build_nn_packagedtsdata(
        train_embeddings, train_summaries, test_embeddings, test_summaries, train, test, mappings)
    prompts = build_prompts(CLASSIFICATION_QUESTION, nn_packages, args.dataset)
    
    # batch completion
    for start_idx in range(0, len(prompts), args.batch_size):
        end_idx = start_idx + args.batch_size
        batch_rows = prompts[start_idx:end_idx]

        # Build this batch’s request payload:
        # each item is a list[dict[str, Any]] for one query
        batch_requests: list[list[dict[str, Any]]] = [
            prompter.format_prompt(row.examples, row.query)
            for row in batch_rows
        ]

        # Call the model once per batch
        batch_results = prompter.get_completion(batch_requests)
        save_results(batch_results)


if __name__ == "__main__":
    main()