#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025-11-20
Author: Dan Schumacher

How to run:
   python ./src/similar_feature_classification.py --dataset emg
"""

import json
import os
import argparse

from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from tqdm import tqdm

# CUSTOM IMPORTS
from utils.build_questions import HELP_STRING, LEGEND_MAPPINGS, TITLE_MAPPINGS, X_MAPPINGS, Y_MAPPINGS
from utils.file_io import append_jsonl, load_jsonl
from utils.image_prompter import ImagePrompter, Prompt
from utils.loggers import MasterLogger
from utils.loaders import load_train_test
from utils.visualization import plot_time_series

# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
def get_embeddings_batch(texts, client: OpenAI, model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Embed a list of texts with OpenAI and return an array of
    shape (N, D) with row-wise L2 normalization.
    """
    if len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    resp = client.embeddings.create(
        model=model,
        input=texts
    )
    embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    # normalize each row to unit length
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs = embs / norms
    return embs


def cosine_similarity_matrix(embs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix for normalized embeddings.
    Assumes each row is already L2-normalized, so cos = dot.
    Returns an (N, N) matrix.
    """
    return embs @ embs.T


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute similarity-based classification over text features using OpenAI embeddings."
    )
    p.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., emg).")
    p.add_argument("--embedding_model", type=str, default="text-embedding-3-small")
    p.add_argument("--model_name", type=str, default="gpt-4o-mini")
    p.add_argument("--visualization_method", type=str, default="line",
               choices=["line", "spectrogram"])
    p.add_argument("--top_k", type=int, default=5, help="Number of nearest neighbors for voting.")
    p.add_argument("--batch_size", type=int, default=5, help="number of simultanous msgs to send to api")
    p.add_argument("--print_to_console", type=int, default=1)
    p.add_argument('--generate_descriptions',type=int, choices=[0,1], default=0)
    return p.parse_args()


def setup():
    args = parse_args()

    # SETUP OUTPUT FILE
    out_folder = f"./data/sample_generations/{args.dataset}/sf_classification/"
    os.makedirs(out_folder, exist_ok=True)

    args.outfile = os.path.join(out_folder, f"top-{args.top_k}.jsonl")

    # If a directory exists *with the same path*, remove it
    if os.path.isdir(args.outfile):
        # user accidentally has a directory named "top-5.jsonl/"
        import shutil
        shutil.rmtree(args.outfile)

    # Clear output file (create empty file)
    with open(args.outfile, "w") as _:
        pass

    logs_dir = "./logs/sf_classification/"
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{args.dataset}.log")

    logger = MasterLogger(
        log_path=log_path,
        init=True,
        clear=True,
        print_to_console=args.print_to_console,
    )

    # Load feature file produced by feature_extraction.py
    feature_path = f"./data/features/{args.dataset}/simple.jsonl"
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found at {feature_path}. "
                                f"Run feature_extraction.py first.")

    train_texts = load_jsonl(feature_path)
    logger.info(f"Loaded {len(train_texts)} feature rows from {feature_path}")

    # LOAD TEST DATA
    _, test = load_train_test(f"./data/samples/{args.dataset}", 0)

    # OpenAI client
    load_dotenv("./resources/.env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in ./resources/.env")
    client = OpenAI(api_key=api_key)

    # SETUP PROMPTER
    prompter = ImagePrompter()
    prompter.model_name = args.model_name
    prompter.system_prompt = (
        f"You will be given a time series visualization"
        f" Your job is to create a detailed textual description of the time series"
        f' using the visualization.'
        " Think step by step and explain your reasoning."
        f" Your contribution is valuable and will later be used to help {HELP_STRING[args.dataset.upper()]}."
    )

    logger.info(f"FINAL OUTPUTS APPEAR AT: {args.outfile}")

    return args, logger, client, train_texts, test, prompter


if __name__ == "__main__":
    args, logger, client, train_texts, test, prompter = setup()

    # ---------------------------------------------------------------
    # Build TRAIN arrays: texts, labels, idxs
    # ---------------------------------------------------------------
    train_text_strs = [row["model_output"] for row in train_texts]
    train_labels = [row["gt"] for row in train_texts]
    train_idxs = [row["idx"] for row in train_texts]
    # ---
    train_text_strs = np.array(train_text_strs, dtype=object)
    train_labels = np.array(train_labels, dtype=int)
    train_idxs = np.array(train_idxs, dtype=int)

    n_train = len(train_text_strs)
    logger.info(f"Loaded {n_train} train feature strings.")

    # ---------------------------------------------------------------
    # Embed TRAIN descriptions once
    # ---------------------------------------------------------------
    logger.info(f"Embedding TRAIN texts using model: {args.embedding_model}")
    train_embs = get_embeddings_batch(train_text_strs.tolist(), client, model=args.embedding_model)

    # ---------------------------------------------------------------
    # TEST batching: generate descriptions with the vision model
    # ---------------------------------------------------------------
    img_root = f"./data/images/{args.dataset}/test/"
    os.makedirs(img_root, exist_ok=True)

    saved_desc_path = f"./data/sample_generations/{args.dataset}/sf_classification/test_descs.jsonl"

    if args.generate_descriptions == 0:
        logger.info(f"Loading saved descriptions from {saved_desc_path}")
        if not os.path.exists(saved_desc_path):
            raise FileNotFoundError(
                "Saved descriptions not found. Run with --generate_descriptions 1 first."
            )

        saved_rows = load_jsonl(saved_desc_path)
        desc_texts = [row["model_output"] for row in saved_rows]
        meta = [(row["idx"], row["gt"]) for row in saved_rows]

    else:
        logger.info("Generating descriptions with API...")
        desc_texts = []
        meta = []

        for start in tqdm(range(0, len(test), args.batch_size), desc="Batches"):
            batch = test[start : start + args.batch_size]
            queries = []
            meta_batch = []

            # ----------- Build prompts & images -----------
            for row in batch:
                img_save_path = os.path.join(img_root, f"{row.idx}.png")
                _ = plot_time_series(
                    row.X,
                    method=args.visualization_method,
                    title=TITLE_MAPPINGS[args.dataset.upper()],
                    xlabs=X_MAPPINGS[args.dataset.upper()],
                    ylabs=Y_MAPPINGS[args.dataset.upper()],
                    legends=LEGEND_MAPPINGS[args.dataset.upper()],
                    save_path=img_save_path,
                    recreate=True,
                )

                prompt = Prompt(
                    user={
                        "question": (
                            "Create a detailed textual description of the time series "
                            "using the visualization. Explain your reasoning."
                        )
                    },
                    img_path=img_save_path,
                    img_detail="auto",
                )

                queries.append(prompt)
                meta_batch.append((
                    int(np.asarray(row.idx).item()),
                    int(np.asarray(row.y).item())
                ))

            # ----------- Send batched requests -----------
            all_messages = [prompter.format_prompt([], q) for q in queries]
            results = prompter.get_completion(all_messages, temperature=0.0)

            if not isinstance(results, list):
                results = [results]

            # ----------- Save descriptions -----------
            batch_descs = [r["content"].strip() for r in results]
            desc_texts.extend(batch_descs)
            meta.extend(meta_batch)

        # ----------- Save completed descriptions to disk -----------
        saved_descs = [
            {"idx": idx, "gt": gt, "model_output": text}
            for (idx, gt), text in zip(meta, desc_texts)
        ]

        with open(saved_desc_path, "w") as f:
            for row in saved_descs:
                f.write(json.dumps(row) + "\n")

    # ---------------------------------------------------------------
    #  Embed descriptions (no API cost if reload mode)
    # ---------------------------------------------------------------
    logger.info("Embedding test descriptions...")
    test_embs = get_embeddings_batch(desc_texts, client, model=args.embedding_model)

    # ---------------------------------------------------------------
    #  CLASSIFY
    # ---------------------------------------------------------------
    correct = 0
    total = 0
    top_k = max(1, min(args.top_k, n_train))

    for (test_idx, gt), test_emb, desc_text in zip(meta, test_embs, desc_texts):
        sims = train_embs @ test_emb
        nn_pos = np.argsort(sims)[::-1][:top_k]

        nn_labels = train_labels[nn_pos]
        vals, counts = np.unique(nn_labels, return_counts=True)
        pred = vals[np.argmax(counts)]

        correct += int(pred == gt)
        total += 1

        line = {
            "idx": int(test_idx),
            "gt": int(gt),
            "pred": int(pred),
            "nearest_neighbors": train_idxs[nn_pos].tolist(),
            "nn_labels": nn_labels.tolist(),
            "sim_scores": sims[nn_pos].tolist(),
            "model_output": desc_text,
        }
        append_jsonl(args.outfile, line)

    acc = correct / total if total else 0.0
    logger.info(f"Similarity-based accuracy (k={args.top_k}): {acc*100:.2f}%")
    print(f"Similarity-based accuracy (k={args.top_k}): {acc*100:.2f}%")
