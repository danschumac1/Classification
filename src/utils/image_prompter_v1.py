from abc import ABC, abstractmethod
import base64
from dataclasses import dataclass, field
import json
import mimetypes
from typing import Any, Dict, Optional

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
    img_path: Optional[str | list[str]]] = None
    assistant: Optional[Dict[str, Any]] = None
    img_detail:str ="auto"
    mime: Optional[str] = field(init=False, default=None)
    b64: Optional[str] = field(init=False, default=None)
    list_of_msgs: list[dict[str, str]] = field(init=False)
    str_representation: str = field(init=False)

    def __post_init__(self) -> None:
        assert self.img_detail in ["low", "high", "auto"], "detail must be 'low', 'high', or 'auto'"

        # Attach image(s) as OpenAI-style `image_url` blocks inside the user dict.
        if self.img_path:
            # Normalize to a list of paths
            if isinstance(self.img_path, list):
                paths: list[str] = self.img_path
            else:
                paths = [self.img_path]

            img_blocks: list[dict[str, Any]] = []
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
        pretty: list[Dict[str, Any]] = []

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


class ImagePrompter(ABC):

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def format_prompt(self, image_path: str, prompt: str) -> str:
        ...

    @abstractmethod
    def get_completion(self, prompt: list[dict[str, Any]] | list[list[dict[str, Any]]]) -> str:
        ...

    def export_prompt_markdown(
        self, 
        examples: list["Prompt"],
        query: "Prompt",
        out_md_path: str,
        *,
        save_images: bool = True,
        images_dirname: str = "images"):
        ...