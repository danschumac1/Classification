import json
import os
from typing import Optional
import warnings

import numpy as np
from utils.build_questions import (
    EXTRA_INFO_MAPPINGS, LEGEND_MAPPINGS, QUESTION_TAG, 
    TASK_DESCRIPTION, TITLE_MAPPINGS, X_MAPPINGS, Y_MAPPINGS)
from utils.llamaPrompter import VisPrompt
from utils.loaders import Split
from utils.visualization import plot_time_series

# -------------------------------------------
# HELPERS
# -------------------------------------------
MODEL_CONTEXT_LIMITS = {
    "text-embedding-3-small": 8192,
    "text-embedding-3-large": 8192,
    "text-embedding-3-large-v1": 8192,
    # "clip-ViT-L/14": None,  # not token-based
    # "ViT-L/14": None,
    "meta-llama/Llama-3.2-11B-Vision-Instruct": 128000,
    "meta-llama/Llama-3.2-11B-Vision": 128000
}


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

def letcs_transform(ts_list: list, precision: int = 3) -> str:
    """
    Convert a time series list (list of floats or ints) into LETCS-style
    digit-space formatting.
    """
    formatted_steps = []

    for x in ts_list:
        s = f"{float(x):.{precision}f}"
        s = s.replace(".", "")
        s = s.replace(",", "")
        digit_spaced = " ".join(list(s))
        formatted_steps.append(digit_spaced)

    return " , ".join(formatted_steps)


def letcs_transform_multivar(ts_2d, precision: int = 3) -> str:
    """
    Convert a multivariate time series (any ndim >= 1) into a single
    LETCS-style string by flattening all dimensions.
    """
    arr = np.asarray(ts_2d, dtype=float)   # ensures numeric + handles nested lists
    flat = arr.flatten().tolist()          # 1D list of floats
    return letcs_transform(flat, precision=precision)

# -------------------------------------------
# SYSTEM PROMPT BUILDERS
# -------------------------------------------
def build_classification_system_prompt(
    dataset: str,
) -> str:
    system_prompt = (
        TASK_DESCRIPTION[dataset.upper()]
        + " You will be given a multiple choice question, the correct answer to that question,"
        " and a time series visualization. Your job is to use the time series visualization to"
        " explain why the provided answer is correct. Think step by step and explain your reasoning."
        " At the end of your explanation, restate the answer using the wording"
        ' \"The answer is [x]\" where x is the correct choice.'
        " Here is some additional information that may help you:\n"
        + EXTRA_INFO_MAPPINGS[dataset.upper()]
    )
    return system_prompt

def build_reasoning_system_prompt(
    dataset: str,
) -> str:
    system_prompt = (
        TASK_DESCRIPTION[dataset.upper()]
        + " You will be given a multiple choice question, the correct answer to that question,"
        " and a time series visualization. Your job is to use the time series visualization to"
        " explain why the provided answer is correct. Think step by step and explain your reasoning."
        " At the end of your explanation, restate the answer using the wording"
        ' \"The answer is [x]\" where x is the correct choice.'
        " Here is some additional information that may help you:\n"
        + EXTRA_INFO_MAPPINGS[dataset.upper()]
    )
    return system_prompt

# -------------------------------------------
# GENERAL PROMPT BUILDER
# -------------------------------------------
def build_prompt(
    row: Split,
    split_name: str,
    *,
    dataset: str,
    model: str,
    include_ts: bool = False,
    include_LETSCLike: bool = False,
    include_vis: bool = False,
    assistant_msg: str = "",
    viz_method: str = "line",
) -> VisPrompt:
    """
    Build a VisPrompt for a single classification example.

    - `include_ts`: append raw time series values as JSON to the user text.
    - `include_LETSCLike`: placeholder flag to add LETSC-like text features (TODO).
    - `include_vis`: render and attach an image of the time series.
    """

    # Make sure we are not building a completely empty prompt
    assert include_ts or include_vis or include_LETSCLike, \
        "At least one of include_ts, include_vis, or include_LETSCLike must be True."
    # can only do include_ts OR include_LETSCLike; not both
    assert not (include_ts and include_LETSCLike), \
        "include_ts and include_LETSCLike cannot both be True."
    assert model in MODEL_CONTEXT_LIMITS, f"Model {model} not recognized."
    assert split_name in ["train", "test"], f"split_name {split_name} not recognized."

    user_text = row.general_question
    user_text = user_text.strip()
    user_text = user_text + "\n\n" + QUESTION_TAG

    if include_ts:
        ts_str = serialize_ts(row.X, MODEL_CONTEXT_LIMITS[model])
        user_text += "\n\nHere are the raw time series values (JSON):\n" + ts_str

    if include_LETSCLike:
            letsclike_str = letcs_transform_multivar(serialize_ts(row.X, MODEL_CONTEXT_LIMITS[model], return_as_list=True))
            user_text += "\n\nHere are the time series values formatted in a special style:\n" + letsclike_str


    image_path: Optional[str] = None
    if include_vis:
        img_dir = f"./data/images/{dataset}/{split_name}/"
        os.makedirs(img_dir, exist_ok=True)

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

    payload = {}
    payload["user_text"] = user_text
    if assistant_msg:
        payload["assistant_text"] = assistant_msg
    if image_path is not None:
        payload["image_path"] = image_path

    # -----------------------------
    # Final VisPrompt
    # -----------------------------
    vis_prompt = VisPrompt(
        **payload
    )
    return vis_prompt

# -------------------------------------------
# REASONING BUILDING PROMPT BUILDERS
# -------------------------------------------
