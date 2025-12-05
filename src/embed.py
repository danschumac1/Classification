'''
2025-12-03
Author: Dan Schumacher
How to run:
   see ./bin/embed.sh
'''
# -------------------------------
# IMPORTS
# -------------------------------
# STANDARD LIBRARIES
import os 
import json
from enum import Enum
import argparse
# PIP INSTALLS
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
import torch
# pip install git+https://github.com/openai/CLIP.git
import clip
from PIL import Image
from tqdm import tqdm
# USER DEFINED FUNCTIONS
from utils.full_pipe.generation import batch_embed, serialize_ts
from utils.loaders import load_train_test
from utils.loggers import MasterLogger
from utils.processing import letcs_transform_multivar
from utils.visualization import plot_time_series
from utils.build_questions import TITLE_MAPPINGS, X_MAPPINGS, Y_MAPPINGS, LEGEND_MAPPINGS

# -------------------------------
# HELPERS
# -------------------------------
class EmbeddingType(Enum):
    ts_direct = 1
    vis_direct = 2
    letsc_direct = 3
    text_summary = 4
    vis_summary = 5
    letsc_summary = 6
    ts_special = 7

MODEL_CONTEXT_LIMITS = {
    "text-embedding-3-small": 8192,
    "text-embedding-3-large": 8192,
    "text-embedding-3-large-v1": 8192,
    "clip-ViT-L/14": None,  # not token-based
    "ViT-L/14": None,
}


def save_results(train_embed: np.ndarray, test_embed: np.ndarray, save_path: str):
    os.makedirs(save_path, exist_ok=True)
    # TODO later we may want more specific naming conventions (model name etc.)
    np.save(os.path.join(save_path, "train_embeddings.npy"), train_embed)
    np.save(os.path.join(save_path, "test_embeddings.npy"), test_embed)


def embed_images_in_batches(paths, model, preprocess, device, batch_size=64):
    """
    Efficiently embed a list of image paths using CLIP with batching + tqdm.
    Returns a (N, D) numpy array.
    """
    all_embeds = []

    for i in tqdm(range(0, len(paths), batch_size), desc="Embedding images"):
        batch_paths = paths[i : i + batch_size]

        # Load + preprocess images
        imgs = [preprocess(Image.open(p)) for p in batch_paths]
        imgs = torch.stack(imgs).to(device)

        with torch.no_grad():
            embs = model.encode_image(imgs)
            embs = embs / embs.norm(dim=-1, keepdim=True)

        all_embeds.append(embs.cpu().numpy())

    return np.vstack(all_embeds)



# -------------------------------
# MAIN
# -------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed time-series data using various methods."
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., har, emg, ctu).",
    )
    p.add_argument(
        "--embedding_type",
        type=str,
        choices=[e.name for e in EmbeddingType],  # ts_direct, vis_direct, ...
        required=True,
        help="Type of embedding to perform.",
    )
    p.add_argument(
        "--embed_model",
        type=str,
        default="text-embedding-3-small",
        choices=["text-embedding-3-small", "clip-ViT-L/14", "ViT-L/14", "text-embedding-3-large"],
        help="Name of the chat/vision model to use for embedding.",
    )
    p.add_argument(
        "--vis_method",
        type=str,
        default="line",
        help="Method to visualize time-series data (for vis_direct embedding).",
    )
    p.add_argument(
        "--normalize",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether to normalize the time-series data.",
    )
    p.add_argument(
        "--print_to_console",
        default=1,
        type=int,
        choices=[0, 1],
        help="Whether to print logs to console.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for embedding.",
    )
    args = p.parse_args()
    args.embedding_type = EmbeddingType[args.embedding_type]
    return args


if __name__ == "__main__":
    # SETUP
        #region SETUP
    # ---------------------------------------------------
    args = parse_args()
    args.input_folder = f"./data/samples/{args.dataset}"

    # DATA
    train, test = load_train_test(
        args.input_folder,
        0,  # shots NA here
        mmap=False,
        attach_artifacts=True,
        normalize=bool(args.normalize),
    )

    # PROMPTER
    load_dotenv("./resources/.env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in ./resources/.env")
    client = OpenAI(api_key=api_key)
    max_tokens = MODEL_CONTEXT_LIMITS[args.embed_model]

    # LOGGER
    logger = MasterLogger(
        log_path=f"./logs/{args.dataset}/embeds/{args.embedding_type.name}.log",
        init=True,
        clear=True,
        print_to_console=args.print_to_console,
    )

    outdir = f"./data/features/{args.dataset}/embeddings/{args.embedding_type.name}"
    os.makedirs(outdir, exist_ok=True)

    train_out = os.path.join(outdir, "train_summaries.jsonl")
    test_out  = os.path.join(outdir, "test_summaries.jsonl")

    # args.embedding_type = EmbeddingType[args.embedding_type]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary_types = [EmbeddingType.text_summary, EmbeddingType.vis_summary, EmbeddingType.letsc_summary]

    # ----------------------------------------------------------------------------------------------
    # EMBEDDING LOGIC
    # ----------------------------------------------------------------------------------------------
    # DIRECTLY EMBED THE LIST OF NUMBERS
    if args.embedding_type in (EmbeddingType.ts_direct, EmbeddingType.letsc_direct):
        if args.embedding_type == EmbeddingType.ts_direct:
            X_train = [serialize_ts(item.X, max_tokens) for item in train]
            X_test  = [serialize_ts(item.X, max_tokens) for item in test]

        elif args.embedding_type == EmbeddingType.letsc_direct:
            # letcs_transform_multivar returns *text*, so we don't need JSON list-of-floats,
            # just apply it and optionally truncate if it's insanely long.
            X_train = [letcs_transform_multivar(serialize_ts(item.X, max_tokens, return_as_list=True)) for item in train]
            X_test  = [letcs_transform_multivar(serialize_ts(item.X, max_tokens, return_as_list=True)) for item in test]

        train_embed, test_embed = batch_embed(
            train_text=X_train,
            test_text=X_test,
            client=client,
            model=args.embed_model,
            batch_size=args.batch_size,
        )



    # EMBED THE GENERATED SUMMARIES
    elif args.embedding_type in summary_types:
        summary_stem = args.embedding_type.name.split("_summary")[0]
        summary_dir = f"./data/features/{args.dataset}/summaries/{summary_stem}/"
        with open(os.path.join(summary_dir, "train_summaries.jsonl"), "r") as f:
            train_summaries = [json.loads(line) for line in f]
        with open(os.path.join(summary_dir, "test_summaries.jsonl"), "r") as f:
            test_summaries = [json.loads(line) for line in f]
        train_texts = [item["summary"] for item in train_summaries]
        test_texts  = [item["summary"] for item in test_summaries]
        # print()
        # print("Example train summary:")
        # print(train_texts[0])
        # print()

        train_embed, test_embed = batch_embed(
            train_text=train_texts,
            test_text=test_texts,
            client=client,
            model=args.embed_model,
        )


    # EMBED THE VISUAL REPRESENTATION DIRECTLY
    elif args.embedding_type == EmbeddingType.vis_direct:
        assert args.embed_model in ["clip-ViT-L/14", "ViT-L/14"], \
            "Only clip-ViT-L/14 is supported for vis_direct embedding type"

        dataset_key = args.dataset.upper()

        print(f"Loading CLIP ViT-L/14 on device={device}...")
        model, preprocess = clip.load("ViT-L/14", device=device)

        # ---------------------------------------------------
        # Generate TRAIN image paths
        # ---------------------------------------------------
        print("Generating train images...")
        train_paths = []
        for idx, item in enumerate(tqdm(train, desc="Saving train plots")):
            img_path = f"./data/images/{args.dataset}/train/{idx}.png"
            img_path = plot_time_series(
                X=item.X,
                method=args.vis_method,
                title=TITLE_MAPPINGS[dataset_key],
                xlabs=X_MAPPINGS[dataset_key],
                ylabs=Y_MAPPINGS[dataset_key],
                legends=LEGEND_MAPPINGS[dataset_key],
                save_path=img_path,
                recreate=True,
            )
            train_paths.append(img_path)

        # ---------------------------------------------------
        # Generate TEST image paths
        # ---------------------------------------------------
        print("Generating test images...")
        test_paths = []
        for idx, item in enumerate(tqdm(test, desc="Saving test plots")):
            img_path = f"./data/images/{args.dataset}/test/{idx}.png"
            img_path = plot_time_series(
                X=item.X,
                method=args.vis_method,
                title=TITLE_MAPPINGS[dataset_key],
                xlabs=X_MAPPINGS[dataset_key],
                ylabs=Y_MAPPINGS[dataset_key],
                legends=LEGEND_MAPPINGS[dataset_key],
                save_path=img_path,
                recreate=True,
            )
            test_paths.append(img_path)

        # ---------------------------------------------------
        # Batch-embed using GPU efficiently
        # ---------------------------------------------------
        print("Embedding TRAIN images...")
        train_embed = embed_images_in_batches(
            train_paths, model, preprocess, device, batch_size=64
        )

        print("Embedding TEST images...")
        test_embed = embed_images_in_batches(
            test_paths, model, preprocess, device, batch_size=64
        )

        print("Finished CLIP embeddings for vis_direct.")


    # USE A SPECIALIZED TIME SERIES EMBEDDING MODEL
    elif args.embedding_type == EmbeddingType.ts_special:
        raise NotImplementedError("ts_special embedding type is not yet implemented.")
    else:
        raise ValueError(f"Unknown embedding type: {args.embedding_type}")


    # ----------------------------------------------------------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------------------------------------------------------
    save_results(
        train_embed=train_embed,
        test_embed=test_embed,
        save_path=f"./data/features/{args.dataset}/embeddings/{args.embedding_type.name}/",
    )