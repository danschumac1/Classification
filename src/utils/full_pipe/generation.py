import json
from tqdm import tqdm
from typing import Any
import numpy as np
from openai import OpenAI
from utils.file_io import append_jsonl
from utils.full_pipe.processing import PromptRow
from utils.image_prompter import ImagePrompter


# def batch_embed(
#     train_text: list[str],
#     test_text: list[str],
#     client: OpenAI,
#     model: str = "text-embedding-3-small",
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Embed a list of texts with OpenAI and return arrays of shape (N, D)
#     with row-wise L2 normalization.
#     """
#     # handle empty inputs cleanly
#     if len(train_text) == 0 or len(test_text) == 0:
#         empty = np.zeros((0, 0), dtype=np.float32)
#         return empty, empty

#     embeds_list = []

#     for summary_texts in [train_text, test_text]:
#         resp = client.embeddings.create(
#             model=model,
#             input=summary_texts,
#         )

#         embs = np.array([d.embedding for d in resp.data], dtype=np.float32)

#         # normalize each row
#         norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
#         embs = embs / norms

#         embeds_list.append(embs)

#     return embeds_list[0], embeds_list[1]

import warnings


import json
import numpy as np
import warnings


def serialize_ts(
    X: np.ndarray,
    max_chars: int = 24000,
    decimals: int = 3,
    return_as_list: bool = False,
) -> str | list:
    """
    Serialize a time series to JSON, downsampling along the *longest* dimension
    until the string is under `max_chars` characters.

    Emits warnings if downsampling is required.
    """

    # Round floats to fewer decimals → shorter strings → fewer tokens
    X_round = np.round(X, decimals=decimals)

    # First serialization attempt: no downsampling
    s = json.dumps(X_round.tolist())
    orig_len = len(s)

    if orig_len <= max_chars:
        return s  # fits fine

    # -------------------------------------------------------------
    # Too large → downsample along the *longest* axis (likely time)
    # -------------------------------------------------------------
    if X_round.ndim == 1:
        time_axis = 0
    else:
        # assume the longest dimension is time
        time_axis = int(np.argmax(X_round.shape))

    T = X_round.shape[time_axis]

    # Estimate factor: length scales ~linearly with T
    est_factor = int(np.ceil(orig_len / max_chars))
    factor = max(est_factor, 1)

    def downsample(X_arr: np.ndarray, step: int) -> np.ndarray:
        slicers = [slice(None)] * X_arr.ndim
        slicers[time_axis] = slice(0, X_arr.shape[time_axis], step)
        return X_arr[tuple(slicers)]

    X_ds = downsample(X_round, factor)
    s_ds = json.dumps(X_ds.tolist())

    # -------------------------------------------------------------
    # Safety net: still too long → increase factor incrementally
    # -------------------------------------------------------------
    while len(s_ds) > max_chars and X_ds.shape[time_axis] > 2:
        factor += 1
        X_ds = downsample(X_round, factor)
        s_ds = json.dumps(X_ds.tolist())

    final_len = len(s_ds)
    final_T = X_ds.shape[time_axis]

    warnings.warn(
        f"[serialize_ts] Time series too long → downsampled.\n"
        f"  Original: T={T}, chars={orig_len}\n"
        f"  Final:    T={final_T}, chars={final_len}\n"
        f"  Axis:     time_axis={time_axis}\n"
        f"  Factor:   every {factor}-th timestep kept\n"
        f"  Threshold: max_chars={max_chars}\n",
        category=UserWarning,
        stacklevel=2,
    )

    if return_as_list:
        return X_ds.tolist()

    return s_ds


def batch_embed(
    train_text: list[str],
    test_text: list[str],
    client: OpenAI,
    model: str = "text-embedding-3-small",
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed lists of texts in batches with OpenAI + tqdm progress bars.
    Returns (train_embeds, test_embeds).
    """

    def _embed_list(texts: list[str], desc: str) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        all_embs = []

        # tqdm for nice progress visibility
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch = texts[i : i + batch_size]

            resp = client.embeddings.create(
                model=model,
                input=batch,
            )

            embs = np.array([d.embedding for d in resp.data], dtype=np.float32)

            # row-wise L2 normalization
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
            embs = embs / norms

            all_embs.append(embs)

        return np.vstack(all_embs)

    train_embs = _embed_list(train_text, desc="Embedding train")
    test_embs  = _embed_list(test_text,  desc="Embedding test")

    return train_embs, test_embs


def generate_summaries(
    train_prompts:list[PromptRow], 
    test_prompts:list[PromptRow], 
    prompter:ImagePrompter, 
    summary_path:str, 
    batch_size:int
    ):
    
    for split, split_name in zip([train_prompts, test_prompts], "train", "test"):
        for start_idx in range(0, len(split), batch_size):
            end_idx = start_idx + batch_size
            batch_rows = split[start_idx:end_idx]


            batch_requests: list[list[dict[str, Any]]] = [
                prompter.format_prompt([], row.query)
                for row in batch_rows
            ]

            # Call the model once per batch
            batch_results = prompter.get_completion(batch_requests)
            lines = []
            for row, result in zip(batch_rows, batch_results):
                lines.append({
                    "idx": row.idx,
                    "gt": row.gt,
                    "model_output": result[0]["text"]
                })
            append_jsonl(summary_path, lines)

