#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Dan Schumacher

How to run:
    python ./src/utils/image_prompter.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import base64
import json
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from openai import OpenAI


# ----------------------------- Data Structures ----------------------------- #


@dataclass
class Prompt:
    """
    A single multimodal prompt example (few-shot or query).

    This object:
        1) Accepts a `user` dict of fields (e.g., {"question": "...", "context": "..."}).
        2) Optionally embeds an image by inlining a data URL into a special key "__image_b64__".
        3) Optionally stores an `assistant` reply for few-shot examples.
        4) Provides a pretty, truncated string representation safe for printing (no huge Base64 dumps).

    Fields
    ------
    user : Dict[str, Any]
        Arbitrary user fields. If an image is attached, we inject a special
        key "__image_b64__" with {"type": "image_url", "image_url": {"url": "data:..."}}
    img_path : Optional[Union[str, List[str]]] = None
        Path or list of paths to image(s) to encode as Base64 and attach to the user content. If None, prompt is text-only.
    assistant : Optional[Dict[str, Any]]
        An optional assistant message for few-shot demonstrations.

    Computed Fields (post-init)
    ---------------------------
    mime : Optional[str]
        The guessed image MIME type (e.g., "image/png") if img_path is set.
    b64 : Optional[str]
        The Base64-encoded image content if img_path is set.
    list_of_msgs : List[Dict[str, str]]
        A compact internal representation used when building printable strings.
    str_representation : str
        Cached pretty JSON string with truncated Base64 for safe logging/printing.
    """

    user: Dict[str, Any]
    img_path: Optional[Union[str, List[str]]] = None
    assistant: Optional[Dict[str, Any]] = None
    img_detail:str ="auto"
    mime: Optional[str] = field(init=False, default=None)
    b64: Optional[str] = field(init=False, default=None)
    list_of_msgs: List[Dict[str, str]] = field(init=False)
    str_representation: str = field(init=False)

    def __post_init__(self) -> None:
        assert self.img_detail in ["low", "high", "auto"], "detail must be 'low', 'high', or 'auto'"

        # Attach image(s) as OpenAI-style `image_url` blocks inside the user dict.
        if self.img_path:
            # Normalize to a list of paths
            if isinstance(self.img_path, list):
                paths: List[str] = self.img_path
            else:
                paths = [self.img_path]

            img_blocks: List[Dict[str, Any]] = []
            for i, path in enumerate(paths):
                mime = self._guess_mime(path)
                b64 = self._encode_b64(path)
                if i == 0:
                    # Keep first image in the legacy fields for backwards compatibility
                    self.mime = mime
                    self.b64 = b64

                data_url = f"data:{mime};base64,{b64}"
                img_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": self.img_detail},
                    }
                )

            # For backwards compatibility:
            # - single image → dict (old behavior)
            # - multiple images → list of dicts
            if len(img_blocks) == 1:
                self.user["__image_b64__"] = img_blocks[0]
            else:
                self.user["__image_b64__"] = img_blocks

        # Build a minimal message ledger for readable string output.
        self.list_of_msgs = [{"user": json.dumps(self.user)}]
        if self.assistant is not None:
            self.list_of_msgs.append({"assistant": json.dumps(self.assistant)})

        self.str_representation = self._build_pretty_str()

    # ---- Private helpers (static) ---- #

    @staticmethod
    def _guess_mime(path: str) -> str:
        """Guess MIME type from filename; default to 'image/png' if unknown."""
        mime, _ = mimetypes.guess_type(path)
        return mime or "image/png"

    @staticmethod
    def _encode_b64(path: str) -> str:
        """Read a file from disk and return its Base64-encoded contents (UTF-8 string)."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # ---- Private helpers (instance) ---- #

    def _build_pretty_str(self) -> str:
        """
        Build an indented JSON representation of prompt messages with Base64 truncated.

        Returns
        -------
        str
            A human-readable JSON string where any embedded Base64 image is
            shortened to prevent log spam while preserving structure.
        """
        pretty: List[Dict[str, Any]] = []

        for message in self.list_of_msgs:
            for role, content_json in message.items():
                # Parse the stored JSON for readability and potential truncation
                content_obj: Dict[str, Any] = json.loads(content_json)

                if role == "user" and "__image_b64__" in self.user:
                    # Safely truncate the large `data:` URL so printing is concise.
                    user_with_img = dict(content_obj)
                    try:
                        url: str = user_with_img["__image_b64__"]["image_url"]["url"]
                        prefix, b64_payload = url.split("base64,", 1)
                        truncated_url = prefix + "base64," + b64_payload[:48] + "..."
                        user_with_img["__image_b64__"]["image_url"]["url"] = truncated_url
                    except Exception:
                        # If something about the payload is off, keep the original (rare).
                        pass

                    pretty.append({"role": role, "value": user_with_img})
                else:
                    pretty.append({"role": role, "value": content_obj})

        return json.dumps(pretty, indent=3, ensure_ascii=False)

    # ---- Display hooks ---- #

    def __str__(self) -> str:
        return self.str_representation

    def __repr__(self) -> str:
        return self.str_representation


# ------------------------------- Core Prompter ------------------------------ #


class ImagePrompter:
    """
    Helper for constructing OpenAI multimodal messages and fetching completions.

    Usage
    -----
    1) Instantiate and set `model_name` and `system_prompt`.
    2) Build Prompt objects for few-shot examples and a query.
    3) Call `format_prompt(examples, query)` to get OpenAI-style messages.
    4) Call `get_completion(messages)` (single) or `get_completion([messages, ...])` (batch).

    Attributes
    ----------
    client : OpenAI
        OpenAI API client, created using an API key loaded from `.env`.
    system_prompt : str
        The system instruction for the chat model.
    model_name : str
        The chat/completions model identifier (e.g., "gpt-4o-mini").
    """

    def __init__(self) -> None:
        # Load API key and initialize an OpenAI client instance.
        api_key = self._load_api_key_from_env()
        self.client: OpenAI = OpenAI(api_key=api_key)

        # These are set by the caller.
        self.system_prompt: str = ""
        self.model_name: str = ""

    # ---- Environment & configuration ---- #

    def _load_api_key_from_env(self) -> str:
        """
        Load the OpenAI API key from `./resources/.env`.

        Returns
        -------
        str
            The API key string.

        Raises
        ------
        ValueError
            If no API key is found at OPENAI_API_KEY.
        """
        load_dotenv("./resources/.env")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API Key not found. Set OPENAI_API_KEY=xxxx in ./resources/.env"
            )
        return api_key

    # ---- Prompt formatting ---- #

    def format_prompt(self, examples: List[Prompt], query: Prompt) -> List[Dict[str, Any]]:
        """
        Convert few-shot examples + a query `Prompt` into OpenAI chat `messages`.

        The strategy preserves the *key order* from `query.user` and requires
        each example's `user` dict to have the same ordered keys (including
        the optional image token key "__image_b64__" if present).

        Parameters
        ----------
        examples : List[Prompt]
            Zero or more `Prompt` objects used as few-shot demonstrations.
        query : Prompt
            The target query `Prompt` to be answered.

        Returns
        -------
        List[Dict[str, Any]]
            Messages ready for `client.chat.completions.create`.

        Raises
        ------
        AssertionError
            If any example's `user` keys do not match the query's `user` keys
            (order-sensitive) which helps prevent schema drift in few-shot prompts.
        """
        # Enforce exact key order agreement between examples and query.
        query_key_order: List[str] = list(query.user.keys())
        for i, example in enumerate(examples):
            example_keys = list(example.user.keys())
            assert (
                example_keys == query_key_order
            ), (
                "USER KEYS MUST MATCH FOR FEW-SHOT EXAMPLES (INCLUDING IMAGES)\n"
                f"- example #{i} keys: {example_keys}\n"
                f"- expected (query):  {query_key_order}"
            )

        # Start with a system message.
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

        def user_dict_to_mm_content(user_obj: Dict[str, Any], key_order: Sequence[str]) -> List[Dict[str, Any]]:
            """
            Convert a `user` dict into a list of multimodal content blocks.
            Text keys become {"type":"text","text": "..."}.
            The special "__image_b64__" key can be:
            - a single dict  {"type":"image_url","image_url":{...}}
            - a list of such dicts
            """
            content_blocks: List[Dict[str, Any]] = []
            for key in key_order:
                value = user_obj[key]

                if key == "__image_b64__":
                    # Single image block
                    if isinstance(value, dict) and "image_url" in value:
                        content_blocks.append(
                            {"type": "image_url", "image_url": value["image_url"]}
                        )
                    # Multiple image blocks
                    elif isinstance(value, list):
                        for img_dict in value:
                            if isinstance(img_dict, dict) and "image_url" in img_dict:
                                content_blocks.append(
                                    {"type": "image_url", "image_url": img_dict["image_url"]}
                                )
                    # else: silently ignore weird shapes
                else:
                    text_value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                    content_blocks.append({"type": "text", "text": f"{key}: {text_value}"})

            return content_blocks


        # Add few-shot examples (user → assistant).
        for example in examples:
            messages.append(
                {"role": "user", "content": user_dict_to_mm_content(example.user, query_key_order)}
            )
            if example.assistant is not None:
                # Coerce assistant content to string (common: {"answer": "..."}).
                assistant_payload = example.assistant
                if isinstance(assistant_payload, str):
                    assistant_text = assistant_payload
                elif isinstance(assistant_payload, dict) and "answer" in assistant_payload:
                    assistant_text = cast(str, assistant_payload["answer"])
                else:
                    assistant_text = json.dumps(assistant_payload, ensure_ascii=False)
                messages.append({"role": "assistant", "content": assistant_text})

        # Add query (final user message).
        messages.append(
            {"role": "user", "content": user_dict_to_mm_content(query.user, query_key_order)}
        )

        return messages

    def build_grouped_image_messages(
        self,
        groups: List[Tuple[str, List[str]]],
        *,
        extra_instruction: str = "",
        detail: str = "auto",
    ) -> List[Dict[str, Any]]:
        """
        Build a single user message that lays out images in named groups, like:

            HERE ARE THE WALKING IMAGES:
            <img1> <img2> ...

            HERE ARE THE WALKING_UPSTAIRS IMAGES:
            <img1> <img2> ...

        Parameters
        ----------
        groups : List[Tuple[str, List[str]]]
            Each tuple is (group_name, list_of_image_paths).
        extra_instruction : str, optional
            Optional final text after all groups (e.g., "Compare the two groups...").
        detail : str, optional
            OpenAI vision detail level ("low", "high", "auto").

        Returns
        -------
        List[Dict[str, Any]]
            A `messages` list ready to pass directly to `get_completion`.
        """
        assert detail in ["low", "high", "auto"], "detail must be 'low', 'high', or 'auto'"

        content_blocks: List[Dict[str, Any]] = []

        for group_name, paths in groups:
            # Header text for this group
            header = f"HERE ARE THE {group_name.upper()} IMAGES:"
            content_blocks.append({"type": "text", "text": header})

            # Images for this group
            for path in paths:
                mime = Prompt._guess_mime(path)
                b64 = Prompt._encode_b64(path)
                data_url = f"data:{mime};base64,{b64}"
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                            "detail": detail,
                        },
                    }
                )

            # Blank line between groups (optional, but helps readability)
            content_blocks.append({"type": "text", "text": ""})

        # Optional trailing instruction
        if extra_instruction:
            content_blocks.append({"type": "text", "text": extra_instruction})

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content_blocks},
        ]
        return messages


    # ---- Completions ---- #

    def get_completion(
        self,
        prompts: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        *,
        max_workers: int = 10,
        request_timeout: float = 60.0,
        temperature: float = 0.0,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Fetch a completion for a single prompt or a batch of prompts.

        Parameters
        ----------
        prompts : Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]
            - Single prompt: a `messages` list like: [{"role":"system",...}, {"role":"user",...}]
            - Batch: a list of such `messages` lists, one per request.
        max_workers : int, optional
            Max worker threads for batch execution, by default 10.
        request_timeout : float, optional
            Timeout (seconds) applied to waiting for futures to finish, by default 60.0.
        temperature : float, optional
            Sampling temperature for generation, by default 0.0.

        Returns
        -------
        Union[Dict[str, Any], List[Dict[str, Any]]]
            - Single dict: {"content": "..."} or {"error": "..."} for single prompt.
            - List of dicts for batch, in the same order as input.

        Notes
        -----
        - Order is preserved for batch results.
        - Exceptions are caught and returned as {"error": "..."} entries
          so that a single failure does not crash the whole batch.
        """
        # Normalize to a batch: List[List[Dict[str, Any]]]
        is_single_prompt = (
            isinstance(prompts, list)
            and (len(prompts) == 0 or isinstance(prompts[0], dict))
        )
        batch: List[List[Dict[str, Any]]]
        if is_single_prompt:
            batch = [cast(List[Dict[str, Any]], prompts)]
        else:
            batch = cast(List[List[Dict[str, Any]]], prompts)

        # Some sanity checks to catch obvious type issues early.
        assert all(isinstance(m, list) for m in batch), "Each item in batch must be a list of message dicts."
        for idx, msgs in enumerate(batch):
            assert all(isinstance(x, dict) for x in msgs), f"Batch item {idx} must be a list of dict messages."

        results: List[Optional[Dict[str, Any]]] = [None] * len(batch)

        def _call_single(i: int, messages: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
            """
            Worker function: send one chat.completions request and extract the first message content.
            Returns
            -------
            (index, result_dict)
            """
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                )
                content: str = response.choices[0].message.content or ""
                return i, {"content": content}
            except Exception as exc:
                raise ValueError(f'{i}, {"error": f"{type(exc).__name__}: {exc}"}')

            # Note: if you want tokens/logprobs etc., extract more fields here.

        # Fire all requests concurrently (or the one request, if single).
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_call_single, i, messages)
                for i, messages in enumerate(batch)
            ]
            for future in as_completed(futures, timeout=request_timeout):
                index, result = future.result()
                results[index] = result

        # There should be no `None` left, but guard defensively:
        finalized = [r if r is not None else {"error": "UnknownError: missing result"} for r in results]
        return finalized[0] if is_single_prompt else finalized
    
    
    def export_messages_markdown(
        self,
        messages: List[Dict[str, Any]],
        out_md_path: str,
        *,
        save_images: bool = True,
        images_dirname: str = "images",
    ) -> str:
        """
        Export an arbitrary `messages` list (like the output of
        `build_grouped_image_messages`) to a readable Markdown file.

        This is what you want for TEST 8: system + one big grouped user message.
        """

        def _safe_stem(s: str) -> str:
            return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)

        os.makedirs(os.path.dirname(out_md_path), exist_ok=True)

        images_dir = os.path.join(os.path.dirname(out_md_path), images_dirname)
        if save_images:
            os.makedirs(images_dir, exist_ok=True)

        def _image_md_from_data_url(data_url: str, stem: str) -> str:
            try:
                header, b64 = data_url.split(",", 1)
                mime = header.split(";")[0].replace("data:", "").strip()
                ext = {
                    "image/png": "png",
                    "image/jpeg": "jpg",
                    "image/jpg": "jpg",
                    "image/webp": "webp",
                }.get(mime, "png")
                filename = f"{_safe_stem(stem)}.{ext}"
                path = os.path.join(images_dir, filename)
                with open(path, "wb") as f:
                    f.write(base64.b64decode(b64))
                return f"![{stem}]({images_dirname}/{filename})"
            except Exception:
                # Fallback: just embed the data URL directly
                return f"![{stem}]({data_url})"

        lines: List[str] = []
        ts = datetime.now().isoformat(timespec="seconds")
        lines.append(f"# Exported Messages ({ts})")
        lines.append("")

        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # -------------------------------
            # System: usually plain text
            # -------------------------------
            if role == "system":
                lines.append("## System Prompt")
                lines.append("")
                lines.append("```")
                # system content is typically a string
                lines.append(content if isinstance(content, str) else str(content))
                lines.append("```")
                lines.append("")
                continue

            # -------------------------------
            # User / assistant / other roles
            # -------------------------------
            lines.append(f"## {role.title()} message {i}")
            lines.append("")

            # For grouped prompts, user content is a list of blocks
            if isinstance(content, list):
                img_counter = 0
                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        txt = block.get("text", "")
                        if txt.strip():
                            lines.append(txt)
                            lines.append("")
                    elif btype == "image_url":
                        img_counter += 1
                        url = block.get("image_url", {}).get("url", "")
                        stem = f"{role}_{i}_img{img_counter}"
                        if url.startswith("data:") and save_images:
                            md_img = _image_md_from_data_url(url, stem)
                        else:
                            md_img = f"![{stem}]({url})"
                        lines.append(md_img)
                        lines.append("")
            else:
                # Fallback if content is just a string
                lines.append("```")
                lines.append(content if isinstance(content, str) else str(content))
                lines.append("```")
                lines.append("")

        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return os.path.abspath(out_md_path)
    
    
    def export_prompt_markdown(
        self,
        examples: List["Prompt"],
        query: "Prompt",
        out_md_path: str,
        *,
        save_images: bool = True,
        images_dirname: str = "images",
    ) -> str:
        """
        Write a readable Markdown file showing:
        - system prompt
        - each example (user fields, embedded/saved image if present, assistant answer)
        - the final query (user fields + image if present)

        Returns the absolute path to the written markdown file.
        """

        def _safe_stem(s: str) -> str:
            # Keep it simple: alnum, dash, underscore
            return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)

        # This ensures the system message is in messages[0]
        if examples:
            messages = self.format_prompt(examples, query)
        else:
            messages = self.format_prompt([], query)    
        os.makedirs(os.path.dirname(out_md_path), exist_ok=True)

        # Optionally create an images folder next to the MD file
        images_dir = os.path.join(os.path.dirname(out_md_path), images_dirname)
        if save_images:
            os.makedirs(images_dir, exist_ok=True)

        def _image_md_from_data_url(data_url: str, stem: str) -> str:
            """
            Kept for backwards compatibility if you ever pass data URLs directly.
            Not used in the current img_path-based flow, but harmless to keep.
            """
            try:
                header, b64 = data_url.split(",", 1)
                mime = header.split(";")[0].replace("data:", "").strip()
                ext = {
                    "image/png": "png",
                    "image/jpeg": "jpg",
                    "image/jpg": "jpg",
                    "image/webp": "webp",
                }.get(mime, "png")
                filename = f"{_safe_stem(stem)}.{ext}"
                path = os.path.join(images_dir, filename)
                with open(path, "wb") as f:
                    f.write(base64.b64decode(b64))
                return f"![{stem}]({images_dirname}/{filename})"
            except Exception:
                return f"![{stem}]({data_url})"

        lines: List[str] = []
        ts = datetime.now().isoformat(timespec="seconds")
        lines.append(f"# Exported Prompt ({ts})")
        lines.append("")

        # System message is always the first in messages (from format_prompt)
        if messages and messages[0].get("role") == "system":
            lines.append("## System Prompt")
            lines.append("")
            lines.append("```")
            lines.append(messages[0].get("content", "") or "")
            lines.append("```")
            lines.append("")

        def dump_prompt_block(prompt: "Prompt", key_order: Sequence[str], stem: str):
            """
            Dump the user block with NO collapsible dropdown:
            - all text fields in user (following key_order, except __image_b64__)
            - the image referenced by prompt.img_path, if any
            """
            user_dict = prompt.user

            # Header instead of collapsible details
            lines.append("### User")
            lines.append("")

            # ------------------------
            # Text fields (skip __image_b64__)
            # ------------------------
            for k in key_order:
                if k == "__image_b64__":
                    continue

                v = user_dict.get(k)
                if v is None:
                    continue

                text_val = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                lines.append(f"- **{k}**: {text_val}")

            # ------------------------
            # Image from img_path or base64
            # ------------------------
            img_path = getattr(prompt, "img_path", None)
            if img_path:
                lines.append("")

                # Normalize to list for convenience
                if isinstance(img_path, list):
                    paths = img_path
                else:
                    paths = [img_path]

                for i, path in enumerate(paths, start=1):
                    tag = f"{stem}_{i}" if len(paths) > 1 else stem

                    # Case 1: base64 data URL → convert & link
                    if isinstance(path, str) and path.startswith("data:"):
                        md_img = _image_md_from_data_url(path, tag)
                        lines.append(md_img)

                    # Case 2: normal file path → copy or link
                    else:
                        if save_images:
                            ext = os.path.splitext(path)[1] or ".png"
                            filename = f"{_safe_stem(tag)}{ext}"
                            target = os.path.join(images_dir, filename)

                            try:
                                shutil.copy2(path, target)
                                rel_path = os.path.join(images_dirname, filename)
                            except Exception:
                                rel_path = os.path.relpath(path, os.path.dirname(out_md_path))
                        else:
                            rel_path = os.path.relpath(path, os.path.dirname(out_md_path))

                        lines.append(f"![{tag}]({rel_path})")
                lines.append("")

        # Determine key order from query.user (format_prompt enforces exact match on examples)
        key_order = list(query.user.keys())

        # Few-shot examples
        if examples:
            lines.append("## Few-shot Examples")
            lines.append("")
            for i, ex in enumerate(examples, start=1):
                lines.append(f"### Example {i}")
                dump_prompt_block(ex, key_order, stem=f"example_{i}")
                if ex.assistant is not None:
                    lines.append("**Assistant**")
                    if isinstance(ex.assistant, dict) and "answer" in ex.assistant:
                        lines.append("")
                        lines.append("```")
                        lines.append(str(ex.assistant["answer"]))
                        lines.append("```")
                    else:
                        lines.append("")
                        lines.append("```json")
                        lines.append(json.dumps(ex.assistant, ensure_ascii=False, indent=2))
                        lines.append("```")
                lines.append("")

        # Query
        lines.append("## Query")
        dump_prompt_block(query, key_order, stem="query")

        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return os.path.abspath(out_md_path)
# ----------------------------------- Main ----------------------------------- #


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    system_prompt_text: str = (
        "You are a helpful assistant. Your job is to help users identify "
        "what is in various images."
    )
    default_question: str = "What animal is shown in the following image?"
    default_context: str = "My friend is making a pixel art game, and I can't tell what creature this is."

    # Initialize prompter (loads API key from ./resources/.env)
    prompter = ImagePrompter()
    prompter.model_name = "gpt-4o-mini"  # set your target model here
    prompter.system_prompt = system_prompt_text

    # -------------------------------------------------------------------------
    # TEST 1 — Prompt creation & truncated string representation
    # -------------------------------------------------------------------------
    print("\n=== TEST 1: Prompt creation with image ===")
    example_cat = Prompt(
        user={"question": default_question, "context": default_context},
        img_path="./demo/images/ex1.jpg",
        assistant={"answer": "Cat"},
    )
    print(example_cat)

    print("\n=== TEST 1B: Prompt creation with a different image ===")
    example_axolotl = Prompt(
        user={"question": default_question, "context": default_context},
        img_path="./demo/images/query.jpg",
    )
    print(example_axolotl)

    # -------------------------------------------------------------------------
    # TEST 2 — Verify base64 truncation happened
    # -------------------------------------------------------------------------
    print("\n=== TEST 2: Base64 truncation check ===")
    assert "..." in str(example_cat), "Base64 truncation did not occur."
    assert "base64," in str(example_cat), "Data URL missing."
    print("Truncation verified.")

    # -------------------------------------------------------------------------
    # TEST 3 — format_prompt correctness
    # -------------------------------------------------------------------------
    print("\n=== TEST 3: format_prompt output shape ===")
    fewshot_messages = prompter.format_prompt([example_cat], example_axolotl)

    assert isinstance(fewshot_messages, list), "format_prompt must return List[dict]"
    assert isinstance(fewshot_messages[0], dict) and fewshot_messages[0].get("role") == "system", (
        "First message must be a system role message"
    )
    print("format_prompt returned a valid prompt structure.")

    # -------------------------------------------------------------------------
    # TEST 3B — Zero-shot prompt formatting (no examples)
    # -------------------------------------------------------------------------
    print("\n=== TEST 3B: Zero-shot prompt format ===")
    zero_shot_messages = prompter.format_prompt([], example_axolotl)

    assert isinstance(zero_shot_messages, list), "Zero-shot prompt must be a list"
    assert zero_shot_messages[0].get("role") == "system", "Zero-shot first message must be system"
    assert zero_shot_messages[-1].get("role") == "user", "Zero-shot last message must be user"
    assert isinstance(zero_shot_messages[-1]["content"], list), "User content must be a multimodal list"

    print("Zero-shot formatting verified.")
    zero_shot_result = prompter.get_completion(zero_shot_messages)
    print(zero_shot_result)

    # -------------------------------------------------------------------------
    # TEST 3C — Prompt with no images should still work
    # -------------------------------------------------------------------------
    print("\n=== TEST 3C: Prompt without image ===")
    text_only_query = Prompt(
        user={"question": "What is 2 + 2?", "context": "Simple math question."}
        # No image attached
    )

    text_only_messages = prompter.format_prompt([], text_only_query)
    text_only_result = prompter.get_completion(text_only_messages)
    print(text_only_result)

    # Validate structure
    assert text_only_messages[-1]["role"] == "user", "Text-only prompt must end with user message"
    assert all(
        block.get("type") == "text" for block in text_only_messages[-1]["content"]
    ), "Text-only prompt must not contain image blocks"
    print("No-image formatting verified.")

    # -------------------------------------------------------------------------
    # TEST 4 — Single completion (cat example explaining axolotl)
    # -------------------------------------------------------------------------
    print("\n=== TEST 4: Single completion call ===")
    single_reply = prompter.get_completion(fewshot_messages)
    print("LLM:", single_reply)

    # -------------------------------------------------------------------------
    # TEST 5 — Batch completion with cat, dog, lion queries
    # -------------------------------------------------------------------------
    print("\n=== TEST 5: Batch completion with 3 images ===")
    example_dog = Prompt(
        user={"question": default_question, "context": default_context},
        img_path="./demo/images/ex2.jpg",
    )
    example_lion = Prompt(
        user={"question": default_question, "context": default_context},
        img_path="./demo/images/ex3.jpg",
    )

    # Build three independent prompts for the batch
    batch_requests: List[List[Dict[str, Any]]] = [
        prompter.format_prompt([example_cat], example_axolotl),  # cat → axolotl
        prompter.format_prompt([example_cat], example_dog),      # cat → dog
        prompter.format_prompt([example_cat], example_lion),     # cat → lion
    ]

    batch_results = prompter.get_completion(batch_requests)
    print("\nBatch results:")
    assert isinstance(batch_results, list), "Expected list of results for batch mode."
    for i, result in enumerate(batch_results):
        print(f"{i}: {result}")

    assert len(batch_results) == 3, "Batch size mismatch."
    print("\nBatch execution verified.")

    # -------------------------------------------------------------------------
    # TEST 6 — Order preservation under threading
    # -------------------------------------------------------------------------
    print("\n=== TEST 6: Thread ordering ===")
    for idx, item in enumerate(batch_results):
        assert isinstance(item, dict), f"Unexpected output type at index {idx}"
    print("Threaded order preserved.")

    print("\nAll tests completed successfully.")


    out_path = prompter.export_prompt_markdown(
        examples=[example_cat],          # or more: [example_cat, example_dog, ...]
        query=example_axolotl,           # whatever your current query Prompt is
        out_md_path="./prompt_export.md",
        save_images=False,                # set False to keep data URLs inline
        images_dirname="./"          # folder next to the .md
    )
    print(f"Markdown exported to: {out_path}")
    # -------------------------------------------------------------------------
    # TEST 7 — Batch completion with zero-shot prompts
    # -------------------------------------------------------------------------
    print("\n=== TEST 7: Batch zero-shot completion ===")

    # Reuse our existing prompts, but now with NO few-shot examples.
    zero_shot_batch_requests: List[List[Dict[str, Any]]] = [
        prompter.format_prompt([], example_axolotl),  # zero-shot axolotl
        prompter.format_prompt([], example_dog),      # zero-shot dog
        prompter.format_prompt([], example_lion),     # zero-shot lion
    ]

    zero_shot_batch_results = prompter.get_completion(zero_shot_batch_requests)

    print("\nZero-shot batch results:")
    assert isinstance(
        zero_shot_batch_results, list
    ), "Expected list of results for zero-shot batch mode."
    assert (
        len(zero_shot_batch_results) == len(zero_shot_batch_requests)
    ), "Zero-shot batch size mismatch."

    for i, result in enumerate(zero_shot_batch_results):
        print(f"Zero-shot {i}: {result}")

    print("\nZero-shot batch execution verified.")

    print("\n=== TEST 8: Grouped layout with multiple images ===")

    cat_imgs = ["./demo/images/ex1.jpg"]
    dog_imgs = ["./demo/images/ex2.jpg"]
    lion_imgs = ["./demo/images/ex3.jpg"]

    prompter.system_prompt = (
        "You are a helpful assistant. You will be shown several groups of animal images. "
        "Each group corresponds to a different animal class. Your job is to compare the "
        "groups and describe visual features that distinguish them."
    )

    groups = [
        ("CAT", cat_imgs),
        ("DOG", dog_imgs),
        ("LION", lion_imgs),
    ]

    grouped_messages = prompter.build_grouped_image_messages(
        groups,
        extra_instruction=(
            "Now carefully compare the groups. Identify patterns that appear "
            "in each animal class and explain how to tell them apart. Think step by step."
        ),
        detail="auto",
    )
    out_md_path = "./prompt_exports/test8_grouped_animals.md"
    out_abs = prompter.export_messages_markdown(
        grouped_messages,
        out_md_path=out_md_path,
        save_images=True,
        images_dirname="images",
    )
    print(f"\nTEST 8 grouped prompt exported to: {out_abs}")

    exit()

    grouped_result = prompter.get_completion(grouped_messages)
    print("\nGrouped layout result:")
    print(grouped_result)

    # -------------------------------------------------------------------------
    # TEST 9 — Export prompt with no examples
    print("\n=== TEST 9: Export zero-shot prompt to Markdown ===")
    zero_shot_query = Prompt(
        user={"question": default_question, "context": default_context},
        img_path="./demo/images/query.jpg",
    )
    zero_shot_md_path = "./prompt_exports/test9_zero_shot.md"
    zero_shot_md_abs = prompter.export_prompt_markdown(
        examples=[],  # no few-shot examples
        query=zero_shot_query,
        out_md_path=zero_shot_md_path,
        save_images=True,
        images_dirname="images",
    )
    print(f"TEST 9 zero-shot prompt exported to: {zero_shot_md_abs}")   